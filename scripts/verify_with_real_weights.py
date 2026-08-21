#!/usr/bin/env python3
"""실제 SmolVLM2 사전학습 가중치로, 지금까지 합성(장난감) 백본으로만 검증했던 세 가지 실험을
순서대로 재확인한다:

  1. 레이어 중요도 (scripts/layer_importance.py와 같은 방법론) — 자연 이미지 + 실제 로봇
     지시문 텍스트로 SmolLM2 32개 레이어의 코사인 유사도 기반 중요도 점수를 계산하고,
     SmolVLA가 기본으로 스킵하는 뒤쪽 16개 레이어(16~31번)와의 overlap을 계산한다.
  2. GradNorm (lerobot.policies.smolvla.ard.GradNormLambdas) — scripts/train_ard.py의
     --use-gradnorm 학습 루프를 몇 스텝만 재현해서 lambda_smooth/force/traj가 실제 SmolVLM2
     백본 위에서도 장난감 백본과 같은 방향으로 움직이는지 본다. 실제 LeRobotDataset이 없으면
     (--dataset-repo-id 안 주면) 무작위 액션을 supervision으로 쓴다 — 그러면 loss 절대값
     자체는 의미가 없고, "lambda가 실제로 움직이는 방향/패턴"만 비교 대상이다.
  3. 비전 토큰 프루닝 (lerobot.policies.smolvla.token_pruning) — 같은 자연 이미지 + 지시문으로
     64개 비전 토큰의 태스크 관련성 점수 분포를 뽑고, 장난감 백본(표준편차가 평균의 0.06%로
     거의 균일)과 달리 유의미한 편차가 있는지, 상위 점수 토큰이 이미지의 어느 위치에 있는지
     그리드 히트맵으로 확인한다.

세 작업 모두 SmolVLM2 가중치가 필요한데, 다운로드/디스크에서 읽어오는 게 제일 느린 부분이라
**딱 한 번만** `AutoModelForImageTextToText.from_pretrained()`를 실제로 호출하고, 그 결과를
in-memory `copy.deepcopy()`로 복제해서 세 작업에 나눠준다(각 작업이 레이어 수를 다르게
자르거나 GradNorm으로 가중치를 업데이트하므로, 서로 영향 안 주게 독립된 사본을 쓴다 — 복제는
디스크/네트워크 접근이 아니라 순수 메모리 복사라 빠르다). `AutoProcessor.from_pretrained()`도
마찬가지로 한 번만 호출해서 공유한다(토크나이저는 읽기 전용으로만 쓰므로 복제 불필요).

Colab GPU 셀 예시 (기본 옵션 그대로, 순서대로 세 작업 전부 실행):
    !git clone <이 레포 URL> ARD-VLA
    %cd ARD-VLA
    !pip install -e "third_party/lerobot[smolvla,peft]"
    !python scripts/verify_with_real_weights.py

특정 작업만 돌리려면 --tasks로 고르면 된다:
    !python scripts/verify_with_real_weights.py --tasks layer_importance token_pruning

기본 이미지/지시문은 HuggingFace 공식 예제에서 흔히 쓰는 COCO 데모 이미지
(고양이 두 마리 + 리모컨 두 개, 안정적으로 접근 가능한 고정 URL)와
"Pick up the black remote control next to the cat."다. 실제 로봇 작업 사진/지시문이 있으면
--image-url(또는 --image-path)과 --instruction으로 바꿔서 쓸 수 있다.

주의 — 이 스크립트는 GPU/Hugging Face Hub 접근이 없는 샌드박스에서 작성되어 실제로 돌려보지
못했다. 대신 각 함수가 부르는 실제 코드 경로(embed_image/embed_language_tokens,
compute_task_relevance_scores/select_tokens, ard.py의 GradNorm 업데이트 순서)는 이번 세션에서
이미 합성 백본으로 여러 번 실행 검증된 것들을 그대로 재사용했고, 이 스크립트 자체의 배선(캐싱
로더가 실제로 다운로드를 한 번만 하는지, 세 작업이 순서대로 에러 없이 도는지)은 별도의 합성
소형 SmolVLM 백본 몽키패치 프로브로 오프라인 검증했다(스크립트에는 포함되지 않은 검증용
스크립트 — Hub 접근이 없는 개발 환경에서만 필요한 것이라 정식 파일로 남기지 않았다).
"""

import argparse
import copy
import gc
import io
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import layer_importance as layer_importance_mod  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla import smolvlm_with_expert as swe  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad  # noqa: E402
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel  # noqa: E402
from lerobot.policies.smolvla.token_pruning import compute_task_relevance_scores, select_tokens  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_INSTRUCTION = "Pick up the black remote control next to the cat."


# --------------------------------------------------------------------------
# 다운로드/로드는 한 번만: AutoModelForImageTextToText.from_pretrained과
# AutoProcessor.from_pretrained을 캐싱 래퍼로 감싸서, SmolVLMWithExpertModel을
# 몇 번을 새로 만들든 실제 네트워크/디스크 접근은 딱 한 번만 일어나게 한다.
# --------------------------------------------------------------------------
class CachedLoader:
    def __init__(self):
        self._model_cache: dict[str, object] = {}
        self._processor_cache: dict[str, object] = {}
        self._real_model_from_pretrained = swe.AutoModelForImageTextToText.from_pretrained
        self._real_processor_from_pretrained = swe.AutoProcessor.from_pretrained

    def install(self) -> None:
        swe.AutoModelForImageTextToText.from_pretrained = self._cached_model_from_pretrained
        swe.AutoProcessor.from_pretrained = self._cached_processor_from_pretrained

    def uninstall(self) -> None:
        swe.AutoModelForImageTextToText.from_pretrained = self._real_model_from_pretrained
        swe.AutoProcessor.from_pretrained = self._real_processor_from_pretrained

    def _cached_model_from_pretrained(self, model_id, *a, **k):
        if model_id not in self._model_cache:
            logging.info("실제 Hugging Face Hub에서 %s 가중치를 로드합니다 (이번 실행에서 딱 한 번만) ...", model_id)
            self._model_cache[model_id] = self._real_model_from_pretrained(model_id, *a, **k)
        else:
            logging.info("%s는 이미 로드되어 있어 메모리에서 복제만 합니다 (재다운로드 없음).", model_id)
        return copy.deepcopy(self._model_cache[model_id])

    def _cached_processor_from_pretrained(self, model_id, *a, **k):
        if model_id not in self._processor_cache:
            self._processor_cache[model_id] = self._real_processor_from_pretrained(model_id, *a, **k)
        return self._processor_cache[model_id]  # 읽기 전용(토크나이즈)으로만 쓰므로 공유해도 안전


def load_image_01(spec: str) -> torch.Tensor:
    """spec이 http(s)://로 시작하면 다운로드, 아니면 로컬 파일로 연다.
    (1, 3, H, W) float 텐서, [0, 1] 범위로 반환한다 (리사이즈/패딩/정규화 전, 원본 크기 그대로)."""
    from PIL import Image

    if spec.startswith("http://") or spec.startswith("https://"):
        import requests

        resp = requests.get(spec, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(spec)
    img = img.convert("RGB")

    import numpy as np

    arr = np.asarray(img).astype("float32") / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def siglip_preprocess(img_01: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """SmolVLAPolicy.prepare_images()와 동일한 전처리: resize_with_pad로 (width, height)에
    맞추고, [0, 1] -> [-1, 1]로 정규화한다 (SigLIP이 기대하는 입력 범위)."""
    img = resize_with_pad(img_01, width, height, pad_value=0)
    img = img * 2.0 - 1.0
    return img


def tokenize_instruction(processor, instruction: str, max_length: int, device: torch.device):
    tokenizer = processor.tokenizer
    enc = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device).bool()


# --------------------------------------------------------------------------
# 1. 레이어 중요도
# --------------------------------------------------------------------------
def task_layer_importance(args, device, real_image_01, lang_tokens, lang_mask) -> None:
    print("\n" + "=" * 70)
    print("[1/3] 레이어 중요도 재확인 (실제 SmolVLM2 가중치)")
    print("=" * 70)

    # num_vlm_layers를 넘기지 않으면(기본 -1) 트림 로직이 건너뛰어져서 원본 레이어 수(보통 32)
    # 그대로 로드된다 — layer_importance.py와 동일한 방식.
    vlm_expert = SmolVLMWithExpertModel(model_id=args.vlm_model_name, load_vlm_weights=True, device=str(device))
    vlm_expert.to(device)
    vlm_expert.eval()

    text_model = vlm_expert.get_vlm_model().text_model
    original_num_layers = len(text_model.layers)
    logging.info("원본 SmolLM2 레이어 수: %d", original_num_layers)

    width, height = args.image_width, args.image_height
    image_embeds = []
    for _ in range(args.cameras):
        img = siglip_preprocess(real_image_01, width, height).to(device)
        image_embeds.append(vlm_expert.embed_image(img))
    image_embeds = torch.cat(image_embeds, dim=1)
    lang_embeds = vlm_expert.embed_language_tokens(lang_tokens)

    inputs_embeds = torch.cat([image_embeds, lang_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    logging.info(
        "대표 시퀀스 길이: %d (실제 이미지 토큰 %d x %d카메라 + 실제 지시문 토큰 %d)",
        inputs_embeds.shape[1], image_embeds.shape[1] // args.cameras, args.cameras, lang_embeds.shape[1],
    )

    captured = layer_importance_mod.capture_layer_io(text_model, inputs_embeds, attention_mask)
    rows = layer_importance_mod.compute_importance_scores(captured)

    layer_importance_mod.print_importance_table(rows)
    layer_importance_mod.print_comparison(rows, args.num_vlm_layers, original_num_layers, args.layer_importance_bottom_n)

    ranked = sorted(rows, key=lambda r: r["importance_score"])
    top2_most_important = sorted(ranked, key=lambda r: r["importance_score"], reverse=True)[:2]
    print(f"\n가장 중요한 레이어 상위 2개: {[r['layer'] for r in top2_most_important]}")
    print(
        "지난번 장난감 백본 결과 참고치: overlap 31.2%, 레이어 30/31이 가장 중요 — 위 실제 결과와 "
        "방향(뒤쪽 레이어가 대체로 더 중요하다는 경향, overlap 정도)을 직접 비교해보세요."
    )

    del vlm_expert, image_embeds, lang_embeds, inputs_embeds, captured
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 2. GradNorm
# --------------------------------------------------------------------------
def build_dummy_action_batch(policy: SmolVLAPolicy, args, real_image_01, lang_tokens, lang_mask, batch_size: int, device: torch.device) -> dict:
    """실제 LeRobotDataset이 없을 때(--dataset-repo-id 안 줬을 때) 쓰는 대체 배치 — 이미지/지시문은
    실제(같은 이미지/텍스트를 배치 전체에 복제), state/action은 무작위. GradNorm의 lambda가
    "실제로 방향성 있게 움직이는지"만 보는 용도라, action의 절대값 자체는 의미가 없다."""
    config = policy.config
    batch = {}
    for key in config.image_features:
        # policy.forward() -> prepare_images()가 resize_with_pad + [0,1]->[-1,1] 정규화를 자체
        # 수행하므로, 여기서는 [0,1] 범위로 크기만 맞춰서 넘긴다 (siglip_preprocess의 *2-1은 안 함).
        img = resize_with_pad(real_image_01, args.image_width, args.image_height, pad_value=0)
        batch[key] = img.expand(batch_size, -1, -1, -1).to(device)
    batch[OBS_STATE] = torch.randn(batch_size, args.state_dim, device=device)
    batch[ACTION] = torch.randn(batch_size, config.chunk_size, args.action_dim, device=device)
    batch[OBS_LANGUAGE_TOKENS] = lang_tokens.expand(batch_size, -1).to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask.expand(batch_size, -1).to(device)
    return batch


def task_gradnorm(args, device, real_image_01, lang_tokens, lang_mask) -> None:
    print("\n" + "=" * 70)
    print("[2/3] GradNorm 재확인 (실제 SmolVLM2 가중치)")
    print("=" * 70)

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
    }
    for i in range(args.cameras):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_height, args.image_width)
        )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))}

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,
        num_vlm_layers=args.num_vlm_layers,
        tokenizer_max_length=args.lang_seq_len,
        use_ard=True,
        ard_arm_dim=args.ard_arm_dim,
        use_gradnorm=True,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        device=str(device),
    )
    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.train()

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    gradnorm_optimizer = torch.optim.Adam([policy.model.ard_gradnorm.weights], lr=config.gradnorm_lr)

    if args.dataset_repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader

        dataset = LeRobotDataset(args.dataset_repo_id)
        dataloader = DataLoader(dataset, batch_size=args.gradnorm_batch_size, shuffle=True, drop_last=True)
        data_iter = iter(dataloader)
        logging.info("실제 데이터셋(%s) 사용 — action target이 진짜입니다.", args.dataset_repo_id)
    else:
        data_iter = None
        logging.warning(
            "--dataset-repo-id가 없어 무작위 action을 supervision으로 씁니다 — loss 절대값은 "
            "의미 없고, lambda_smooth/force/traj가 스텝별로 움직이는 방향/패턴만 비교하세요."
        )

    rows = []
    for step in range(1, args.gradnorm_steps + 1):
        if data_iter is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        else:
            batch = build_dummy_action_batch(policy, args, real_image_01, lang_tokens, lang_mask, args.gradnorm_batch_size, device)

        loss, loss_dict = policy.forward(batch)
        grad_loss = loss_dict.pop("ard_grad_loss_tensor", None)

        if grad_loss is not None:
            gradnorm_optimizer.zero_grad()
            grad_loss.backward(inputs=[policy.model.ard_gradnorm.weights], retain_graph=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
        optimizer.step()

        if grad_loss is not None:
            gradnorm_optimizer.step()
            policy.model.ard_gradnorm.renormalize()

        rows.append(
            {
                "step": step,
                "loss": loss_dict["loss"],
                "smooth": loss_dict["ard_smooth_loss"],
                "force": loss_dict["ard_force_loss"],
                "traj": loss_dict["ard_traj_loss"],
                "lambda_smooth": loss_dict["ard_lambda_smooth"],
                "lambda_force": loss_dict["ard_lambda_force"],
                "lambda_traj": loss_dict["ard_lambda_traj"],
            }
        )

    print(f"\n{'step':>4} | {'loss':>8} | {'lambda_smooth':>13} | {'lambda_force':>12} | {'lambda_traj':>11}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['step']:>4} | {r['loss']:>8.4f} | {r['lambda_smooth']:>13.4f} | "
            f"{r['lambda_force']:>12.4f} | {r['lambda_traj']:>11.4f}"
        )

    first, last = rows[0], rows[-1]
    print(
        f"\nlambda_smooth: {first['lambda_smooth']:.2f} -> {last['lambda_smooth']:.2f}   "
        f"lambda_force: {first['lambda_force']:.2f} -> {last['lambda_force']:.2f}   "
        f"lambda_traj: {first['lambda_traj']:.2f} -> {last['lambda_traj']:.2f}"
    )
    print(
        "지난번 장난감 백본 결과 참고치 (8스텝): lambda_smooth 1.00->1.33, lambda_force 1.00->1.00(거의 고정), "
        "lambda_traj 1.00->0.67 — 위 실제 결과와 방향(어느 lambda가 커지고 작아지는지)을 비교해보세요."
    )

    del policy, optimizer, gradnorm_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 3. 비전 토큰 프루닝
# --------------------------------------------------------------------------
def task_token_pruning(args, device, real_image_01, lang_tokens, lang_mask) -> None:
    print("\n" + "=" * 70)
    print("[3/3] 비전 토큰 프루닝 재확인 (실제 SmolVLM2 가중치)")
    print("=" * 70)

    vlm_expert = SmolVLMWithExpertModel(
        model_id=args.vlm_model_name, load_vlm_weights=True, num_vlm_layers=args.num_vlm_layers, device=str(device)
    )
    vlm_expert.to(device)
    vlm_expert.eval()

    img = siglip_preprocess(real_image_01, args.image_width, args.image_height).to(device)
    with torch.no_grad():
        img_emb = vlm_expert.embed_image(img)  # (1, N, hidden)
        lang_emb = vlm_expert.embed_language_tokens(lang_tokens)  # (1, L, hidden)
        relevance = compute_task_relevance_scores(img_emb, lang_emb, language_mask=lang_mask)  # (1, N)

    scores = relevance[0].float().cpu()
    n_tokens = scores.shape[0]
    mean, std = scores.mean().item(), scores.std().item()
    print(f"\n비전 토큰 개수: {n_tokens}")
    print(f"relevance score: mean={mean:.6f}  std={std:.6f}  (std/mean={100 * std / mean:.2f}%)")
    print(f"min={scores.min().item():.6f}  max={scores.max().item():.6f}")
    print(
        "지난번 장난감 백본 결과 참고치: std/mean ≈ 0.06% (사실상 균일) — 위 실제 결과의 std/mean이 "
        "이보다 뚜렷하게 크다면, 사전학습된 표현이 실제로 토큰 간 관련성 차이를 구분하고 있다는 뜻입니다."
    )

    order = torch.argsort(scores, descending=True)
    k_key = args.token_pruning_k_key
    print(f"\n관련성 상위 {k_key}개(핵심 세트) 토큰 인덱스와 점수:")
    for idx in order[:k_key].tolist():
        print(f"  토큰 #{idx:>3}: {scores[idx].item():.6f}")

    _selected_tokens, selected_indices = select_tokens(
        img_emb, relevance, k_final=args.token_pruning_k_final, k_key=k_key
    )
    print(f"\nselect_tokens()가 최종적으로 고른 {args.token_pruning_k_final}개 토큰 인덱스: {sorted(selected_indices[0].tolist())}")

    grid_side = round(n_tokens**0.5)
    if grid_side * grid_side == n_tokens:
        _save_relevance_heatmap(scores, grid_side, order[:k_key].tolist(), real_image_01, args)
    else:
        logging.warning(
            "비전 토큰 개수(%d)가 정사각형 그리드로 안 떨어져서(%d^2 != %d) 이미지 위 히트맵 오버레이는 건너뜁니다 "
            "— 위 점수 표만으로 확인해주세요.",
            n_tokens, grid_side, n_tokens,
        )

    del vlm_expert, img_emb, lang_emb, relevance
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _save_relevance_heatmap(scores: torch.Tensor, grid_side: int, core_indices: list[int], real_image_01: torch.Tensor, args) -> None:
    """relevance score를 (grid_side, grid_side) 히트맵으로 만들어 원본 이미지 위에 겹쳐 그린다.
    상위 k_key(핵심 세트) 토큰이 실제로 지시문과 관련된 물체 위치에 있는지 눈으로 확인하는 용도."""
    import matplotlib.pyplot as plt
    import numpy as np

    heatmap = scores.reshape(grid_side, grid_side).numpy()
    core_mask = np.zeros(grid_side * grid_side, dtype=bool)
    core_mask[core_indices] = True
    core_mask = core_mask.reshape(grid_side, grid_side)

    img_np = real_image_01[0].permute(1, 2, 0).numpy()

    # matplotlib 기본 폰트(DejaVu Sans)가 한글 글리프를 지원하지 않아 제목/축만 영어로 쓴다
    # (콘솔 출력/주석 등 다른 텍스트는 이 레포 관례대로 한국어 유지 — 여기는 렌더링 결과물이라
    # 별도 한글 폰트 설치 없이도 깨지지 않게 하는 게 우선).
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(img_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    im = axes[1].imshow(heatmap, cmap="inferno", alpha=0.55, extent=(0, img_np.shape[1], img_np.shape[0], 0))
    for r in range(grid_side):
        for c in range(grid_side):
            if core_mask[r, c]:
                axes[1].add_patch(
                    plt.Rectangle(
                        (c * img_np.shape[1] / grid_side, r * img_np.shape[0] / grid_side),
                        img_np.shape[1] / grid_side, img_np.shape[0] / grid_side,
                        fill=False, edgecolor="cyan", linewidth=2,
                    )
                )
    axes[1].set_title(f"Task-relevance heatmap ({grid_side}x{grid_side} token grid, cyan = core set)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f'Instruction: "{args.instruction}"', fontsize=11)
    fig.tight_layout()

    out_path = Path(args.output_dir) / "token_relevance_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n관련성 히트맵 저장: {out_path}")


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["layer_importance", "gradnorm", "token_pruning"],
        default=["layer_importance", "gradnorm", "token_pruning"],
        help="어떤 작업을 (순서대로) 돌릴지. 기본은 셋 다.",
    )
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="http(s):// URL. --image-path가 주어지면 무시된다.")
    parser.add_argument("--image-path", default=None, help="로컬 이미지 파일 경로 (주어지면 --image-url 대신 이걸 쓴다)")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image-width", type=int, default=512, help="SmolVLAConfig.resize_imgs_with_padding 기본값과 동일")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--lang-seq-len", type=int, default=48, help="config.tokenizer_max_length 기본값과 동일")
    parser.add_argument("--num-vlm-layers", type=int, default=16, help="SmolVLA 기본값 — layer_importance는 원본 32개 전부를 별도로 로드하므로 영향 없음")
    parser.add_argument("--cameras", type=int, default=3, help="top + 좌손목 + 우손목 (레이어 중요도/GradNorm 배치 구성용)")
    parser.add_argument("--device", default=None, help="'cuda' 또는 'cpu'. 기본은 자동 감지")
    parser.add_argument("--output-dir", default="outputs/verify_real_weights")

    li_group = parser.add_argument_group("1. 레이어 중요도")
    li_group.add_argument("--layer-importance-bottom-n", type=int, default=None, help="기본: 원본 - num_vlm_layers")

    gn_group = parser.add_argument_group("2. GradNorm")
    gn_group.add_argument("--dataset-repo-id", default=None, help="주면 실제 LeRobotDataset을 쓴다. 안 주면 무작위 action(방향성만 비교 목적).")
    gn_group.add_argument("--gradnorm-steps", type=int, default=8, help="지난번 장난감 백본 실험과 동일하게 기본 8스텝")
    gn_group.add_argument("--gradnorm-batch-size", type=int, default=4)
    gn_group.add_argument("--gradnorm-alpha", type=float, default=1.5)
    gn_group.add_argument("--gradnorm-lr", type=float, default=0.025)
    gn_group.add_argument("--ard-arm-dim", type=int, default=7)
    gn_group.add_argument("--action-dim", type=int, default=14, help="bimanual: 7 left + 7 right")
    gn_group.add_argument("--state-dim", type=int, default=14)
    gn_group.add_argument("--chunk-size", type=int, default=50)

    tp_group = parser.add_argument_group("3. 비전 토큰 프루닝")
    tp_group.add_argument("--token-pruning-k-final", type=int, default=32)
    tp_group.add_argument("--token-pruning-k-key", type=int, default=6)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logging.warning("GPU가 없어 CPU로 실행합니다 — SmolVLM2-500M 순전파라 많이 느릴 수 있습니다.")

    logging.info("설정: vlm_model_name=%s tasks=%s device=%s", args.vlm_model_name, args.tasks, device)

    image_spec = args.image_path if args.image_path else args.image_url
    logging.info("이미지 로드: %s", image_spec)
    real_image_01 = load_image_01(image_spec)  # (1, 3, H_orig, W_orig), [0, 1] — 아직 리사이즈 전

    loader = CachedLoader()
    loader.install()
    try:
        # 프로세서(토크나이저)는 아무 SmolVLMWithExpertModel이나 하나 먼저 만들어서 얻는 대신,
        # 실제 다운로드 캐싱이 걸려 있는 상태에서 첫 작업이 알아서 로드해줄 것이므로 여기서는
        # 지시문 토큰화에 필요한 processor만 미리 한 번 얻어둔다 (모델은 아직 안 만듦).
        processor = loader._cached_processor_from_pretrained(args.vlm_model_name)
        lang_tokens, lang_mask = tokenize_instruction(processor, args.instruction, args.lang_seq_len, device)
        logging.info('지시문: "%s" -> 토큰 %d개 (유효 %d개)', args.instruction, lang_tokens.shape[1], int(lang_mask.sum().item()))

        if "layer_importance" in args.tasks:
            task_layer_importance(args, device, real_image_01, lang_tokens, lang_mask)
        if "gradnorm" in args.tasks:
            task_gradnorm(args, device, real_image_01, lang_tokens, lang_mask)
        if "token_pruning" in args.tasks:
            task_token_pruning(args, device, real_image_01, lang_tokens, lang_mask)
    finally:
        loader.uninstall()

    print("\n" + "=" * 70)
    print("전체 완료. 위 각 절의 '지난번 장난감 백본 결과 참고치'와 비교해서 방향이 같은지 확인하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
