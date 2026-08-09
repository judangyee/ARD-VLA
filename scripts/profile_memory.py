#!/usr/bin/env python3
"""SmolVLA(+ARD) 메모리 프로파일링 — LoRA + bf16 + gradient checkpointing을 켠 상태에서
배치 사이즈별 forward+backward 최대 GPU 메모리 사용량을 측정한다.

Colab/Kaggle 노트북 셀에 그대로 실행할 수 있도록 단일 파일로 짰다. GPU가 필요하다
(torch.cuda.max_memory_allocated 기반). 더미 이미지/언어 토큰으로 배치를 만들기 때문에
실제 데이터셋 없이 바로 돌아가지만, 그만큼 "숫자가 맞다"가 아니라 "메모리가 얼마나 드는지"만
알려준다.

기본 실행은 SmolVLA의 레이어 프루닝(`num_vlm_layers`로 SmolLM2를 앞쪽 몇 개 레이어만 쓰도록
자르는 것) 적용 여부를 둘 다 프로파일링해서 표를 두 개 낸다 — "적용 O"는 기본 설정
(`--num-vlm-layers`, 기본 16)대로 자른 모델, "적용 X"는 원본 SmolLM2 레이어 수를 그대로 쓰는
모델이다. 원본 레이어 수 쪽은 action expert도 같이 커지기 때문에(`num_expert_layers`가 기본
`-1`이라 VLM 레이어 수를 따라감) 훨씬 무겁고 OOM이 더 빨리 날 수 있다. `--layer-pruning-mode
pruned`나 `unpruned`를 주면 그중 하나만 돌려서 시간을 아낄 수 있다.

`--vlm-layer-indices`를 주면 "앞쪽 N개"라는 기본 규칙 대신 임의의 원본 레이어 인덱스 조합을
그대로 써서 ARD-VLA를 빌드하고, "기본(pruned)" vs "사용자 지정(custom)" 두 표를 비교
출력한다 (예: `scripts/layer_importance.py`가 코사인 유사도 기준으로 골라준 레이어들 —
레이어 개수가 같으면 메모리 자체는 거의 그대로 나올 것으로 예상되지만, 그 구성으로 실제로
문제없이 빌드/학습되는지 확인하는 용도다). 이때는 `--layer-pruning-mode`가 무시된다.

Colab/Kaggle 셀 예시:
    !git clone <이 레포 URL> ARD-VLA
    %cd ARD-VLA
    !pip install -e "third_party/lerobot[smolvla,peft]"
    !python scripts/profile_memory.py

주의 — 이 스크립트는 이 프로젝트를 만든 샌드박스에 GPU와 Hugging Face Hub 접근이 둘 다
없어서 실제로 실행해보지 못했다. 아래는 검증한 것과 못 한 것이다:
  - gradient checkpointing 래핑 패턴(리스트-오브-텐서 + None + non-tensor 인자 혼합,
    torch.utils.checkpoint.checkpoint(..., use_reentrant=False))은 가짜 레이어로 순전파/역전파
    결과가 체크포인팅 유무와 정확히 일치하는지 별도로 검증했다 (`smolvlm_with_expert.py`에
    새로 추가한 기능).
  - `wrap_with_peft()`가 서브모듈을 in-place로 교체하는 peft의 표준 동작(문서화된 동작이며
    직접 실행 검증은 못 함)에 의존한다.
  - 배치 딕셔너리 키/shape(OBS_STATE, ACTION, OBS_LANGUAGE_TOKENS 등)는 modeling_smolvla.py의
    prepare_state/prepare_action/forward 코드를 직접 읽고 맞춘 것이라 로직상 맞아야 하지만,
    실제 SmolVLM2 토크나이저 vocab_size 등 Hub에서만 받아지는 값에 의존하는 부분은 실행 전까지
    확신할 수 없다.
  - 처음 실행하면 배치 사이즈 1부터 먼저 통과하는지 확인하고, 이상하면 --steps나 --cameras 등을
    줄여서 재현 범위를 좁혀보는 걸 권장한다.
"""

import argparse
import gc
import logging

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--chunk-size", type=int, default=50, help="action chunk length (horizon)")
    parser.add_argument("--action-dim", type=int, default=14, help="bimanual: 7 left + 7 right")
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--ard-arm-dim", type=int, default=7)
    parser.add_argument("--cameras", type=int, default=3, help="top + 좌손목 + 우손목")
    parser.add_argument("--image-size", type=int, default=128, help="더미 이미지 한 변 길이 (resize_imgs_with_padding이 어차피 재조정함)")
    parser.add_argument("--lang-seq-len", type=int, default=48, help="config.tokenizer_max_length 기본값과 동일")
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="LoRA 끄고 프로파일링")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false", help="bf16 autocast 끄기")
    parser.add_argument("--no-grad-checkpoint", dest="use_grad_checkpoint", action="store_false")
    parser.add_argument("--no-ard", dest="use_ard", action="store_false", help="ARD 없이 베이스 SmolVLA만 프로파일링")
    parser.add_argument(
        "--num-vlm-layers",
        type=int,
        default=16,
        help="레이어 프루닝 적용 시 사용할 num_vlm_layers 값 (SmolVLAConfig 기본값과 동일)",
    )
    parser.add_argument(
        "--layer-pruning-mode",
        choices=["both", "pruned", "unpruned"],
        default="both",
        help=(
            "'both'(기본)면 --num-vlm-layers로 자른 모델과 원본 레이어 수 그대로인 모델을 "
            "둘 다 프로파일링해서 표를 두 개 출력한다. 'pruned'/'unpruned'면 그 중 하나만 실행한다. "
            "--vlm-layer-indices가 주어지면 이 옵션은 무시된다."
        ),
    )
    parser.add_argument(
        "--vlm-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "'앞에서부터 --num-vlm-layers개'라는 기본 규칙 대신, 이 원본 레이어 인덱스 조합을 "
            "그대로 써서 ARD-VLA를 빌드한다 (예: scripts/layer_importance.py가 코사인 유사도 "
            "기준으로 골라준 레이어들). 주어지면 --layer-pruning-mode 대신 '기본(pruned)' vs "
            "'사용자 지정(custom)' 두 표를 비교해서 출력한다."
        ),
    )
    parser.set_defaults(use_lora=True, use_bf16=True, use_grad_checkpoint=True, use_ard=True)
    return parser.parse_args()


def build_policy(args, mode: str) -> SmolVLAPolicy:
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
    }
    for i in range(args.cameras):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))}

    # mode="pruned": 기본 규칙(앞에서부터 num_vlm_layers개). mode="unpruned": 트림 없이 원본
    # 레이어 수 그대로(num_vlm_layers<=0이면 smolvlm_with_expert.py의 트림 로직이 아예
    # 건너뛰어짐 — action expert도 num_expert_layers=-1(기본)이라 VLM 레이어 수를 따라가며
    # 함께 커진다). mode="custom": --vlm-layer-indices로 받은 임의의 레이어 인덱스 조합을
    # 그대로 사용 (예: scripts/layer_importance.py의 중요도 기준 선택 결과).
    if mode == "custom":
        num_vlm_layers = args.num_vlm_layers
        vlm_layer_indices = args.vlm_layer_indices
    elif mode == "pruned":
        num_vlm_layers = args.num_vlm_layers
        vlm_layer_indices = None
    else:
        num_vlm_layers = 0
        vlm_layer_indices = None

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,  # 실제 사전학습 백본 — 메모리 프로파일링은 실제 가중치 기준이어야 의미 있음
        pretrained_path=args.vlm_model_name if args.use_lora else None,  # PEFT의 "from-scratch 경고" 통과용 — VLM 백본은 실제로 사전학습 가중치를 받으므로 사실과 부합
        num_vlm_layers=num_vlm_layers,
        vlm_layer_indices=vlm_layer_indices,
        use_ard=args.use_ard,
        ard_arm_dim=args.ard_arm_dim,
        tokenizer_max_length=args.lang_seq_len,
        device="cuda",
    )
    policy = SmolVLAPolicy(config)
    policy.to("cuda")
    return policy


def apply_lora(policy: SmolVLAPolicy, args) -> SmolVLAPolicy:
    """LoRA를 씌운다. wrap_with_peft()는 대상 서브모듈을 in-place로 교체하므로(peft 표준 동작),
    호출 뒤에도 원래 `policy` 참조를 계속 forward/backward에 써도 된다 — 반환값은 peft 자체
    부기(trainable params 집계 등)를 위해서만 보관한다.
    """
    peft_model = policy.wrap_with_peft(peft_cli_overrides={"r": args.lora_r, "lora_alpha": args.lora_alpha})
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    if args.use_ard:
        # wrap_with_peft()는 시작할 때 전체 파라미터를 얼리고(requires_grad=False) LoRA 어댑터만
        # 풀어준다 — ARD 기본 target_modules에는 stabilizer_head/actuator_head가 없어서, 그대로
        # 두면 ARD head가 통째로 얼어붙는다. 실제 ARD 파인튜닝 시나리오와 맞추려면 다시 풀어줘야 한다.
        n = 0
        for p in policy.model.ard_heads.parameters():
            p.requires_grad_(True)
            n += p.numel()
        logging.info("LoRA 적용 후 AsymmetricResidualHeads %d개 파라미터를 다시 학습 가능하게 풀었습니다.", n)
    return peft_model


def build_dummy_batch(policy: SmolVLAPolicy, args, batch_size: int, device: torch.device) -> dict:
    config = policy.config
    batch = {}
    for key in config.image_features:
        batch[key] = torch.rand(batch_size, 3, args.image_size, args.image_size, device=device)
    batch[OBS_STATE] = torch.randn(batch_size, args.state_dim, device=device)
    batch[ACTION] = torch.randn(batch_size, args.chunk_size, args.action_dim, device=device)

    vocab_size = policy.model.vlm_with_expert.vlm.config.text_config.vocab_size
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(
        low=0, high=vocab_size, size=(batch_size, args.lang_seq_len), device=device
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        batch_size, args.lang_seq_len, dtype=torch.bool, device=device
    )
    return batch


def profile_one_batch_size(policy, run_forward_backward, batch_size: int) -> float:
    """forward+backward 1회를 돌리고 최대 할당 메모리(GB)를 반환한다."""
    policy.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    run_forward_backward(batch_size)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024**3)


def run_layer_pruning_sweep(args, mode: str) -> tuple[str, list[tuple[int, float | None, str | None]]]:
    """레이어 선택 모드 하나에 대해 정책을 새로 만들고 배치 사이즈 스윕을 전부 돈다.

    mode마다 모델 구조 자체가 다르므로(action expert 레이어 수까지 달라짐) 정책을 공유할 수
    없어 매번 새로 빌드한다 — 끝나면 다음 모드를 위해 GPU 메모리를 명시적으로 비운다.
    """
    device = torch.device("cuda")
    policy = build_policy(args, mode=mode)
    actual_num_vlm_layers = policy.model.vlm_with_expert.num_vlm_layers
    if mode == "pruned":
        label = f"기본 규칙: 앞쪽 num_vlm_layers={actual_num_vlm_layers}개"
    elif mode == "unpruned":
        label = f"레이어 프루닝 적용 X / 원본 레이어 수 그대로 (num_vlm_layers={actual_num_vlm_layers})"
    else:
        label = f"사용자 지정 레이어 인덱스 (n={actual_num_vlm_layers}): {sorted(args.vlm_layer_indices)}"
    logging.info("=== %s ===", label)

    if args.use_grad_checkpoint:
        # LoRA를 씌우기 전에 켠다 — gradient_checkpointing은 vlm_with_expert 자체의 플래그라
        # 순서는 상관없지만, 다른 토글들과 일관되게 "정책 준비" 단계에서 전부 끝내둔다.
        policy.model.vlm_with_expert.gradient_checkpointing_enable()
        logging.info("gradient checkpointing 활성화 (VLAFlowMatching.forward에서만 적용, 추론 경로는 영향 없음)")

    policy.train()

    if args.use_lora:
        apply_lora(policy, args)
    else:
        for p in policy.parameters():
            p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logging.info("학습 가능 파라미터: %s / 전체 %s (%.2f%%)", f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total)

    autocast_dtype = torch.bfloat16 if args.use_bf16 else None

    def run_forward_backward(batch_size: int) -> None:
        batch = build_dummy_batch(policy, args, batch_size, device)
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss, _ = policy.forward(batch)
        else:
            loss, _ = policy.forward(batch)
        loss.backward()

    results = []
    for batch_size in args.batch_sizes:
        try:
            peak_gb = profile_one_batch_size(policy, run_forward_backward, batch_size)
            results.append((batch_size, peak_gb, None))
            logging.info("batch_size=%d -> peak %.2f GB", batch_size, peak_gb)
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            results.append((batch_size, None, "OOM"))
            logging.warning("batch_size=%d -> OOM (%s)", batch_size, str(e).splitlines()[0])
            torch.cuda.empty_cache()

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return label, results


def print_results_table(label: str, results: list[tuple[int, float | None, str | None]]) -> None:
    print()
    print(f"=== {label} ===")
    print(f"{'batch_size':>10} | {'peak memory (GB)':>16}")
    print("-" * 30)
    for batch_size, peak_gb, note in results:
        value = f"{peak_gb:.2f}" if peak_gb is not None else (note or "-")
        print(f"{batch_size:>10} | {value:>16}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU가 없습니다 — 이 스크립트는 torch.cuda.max_memory_allocated 기반이라 GPU가 필수입니다. "
            "Colab/Kaggle에서 런타임을 GPU로 설정했는지 확인하세요."
        )

    logging.info(
        "설정: LoRA=%s(r=%d) bf16=%s grad_checkpoint=%s ARD=%s chunk_size=%d action_dim=%d cameras=%d "
        "layer_pruning_mode=%s vlm_layer_indices=%s",
        args.use_lora, args.lora_r, args.use_bf16, args.use_grad_checkpoint, args.use_ard,
        args.chunk_size, args.action_dim, args.cameras, args.layer_pruning_mode, args.vlm_layer_indices,
    )

    if args.vlm_layer_indices is not None:
        # 사용자 지정 레이어 인덱스가 주어지면, --layer-pruning-mode는 무시하고 항상
        # "기본(앞쪽 num_vlm_layers개)" vs "사용자 지정 인덱스"를 비교한다.
        modes_to_run = ["pruned", "custom"]
    else:
        modes_to_run = []
        if args.layer_pruning_mode in ("both", "pruned"):
            modes_to_run.append("pruned")
        if args.layer_pruning_mode in ("both", "unpruned"):
            modes_to_run.append("unpruned")

    all_results = [run_layer_pruning_sweep(args, mode=mode) for mode in modes_to_run]

    for label, results in all_results:
        print_results_table(label, results)


if __name__ == "__main__":
    main()
