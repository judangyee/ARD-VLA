#!/usr/bin/env python3
"""SmolLM2 백본의 각 레이어가 얼마나 '중요한지' 측정해서, SmolVLA가 기본으로 스킵하는 16개
레이어 선택(뒤쪽 절반을 통째로 버리는 것)이 최적인지 확인한다.

측정 방법 (ShortGPT류 "레이어 중복성" 분석과 동일한 아이디어):
  1. 대표 샘플 배치를 (더미 이미지 + 더미 언어 토큰으로) 만들어서, SmolVLA가 실제로 쓰는
     `embed_image()`/`embed_language_tokens()`를 통해 임베딩한 뒤 이어붙인다.
  2. 이 시퀀스를 SmolLM2 백본(`text_model`)에 통과시키면서, forward hook으로 각 레이어의
     "입력 hidden state"와 "출력 hidden state"를 그대로 뽑는다 — `num_vlm_layers`로
     자르지 않은 원본 전체 레이어(보통 32개)를 대상으로 한다.
  3. 레이어별로 (배치 전체 x 토큰 위치 전체에 대해 평균한) 코사인 유사도를 구한다.
  4. 중요도 점수 = 1 - 평균 코사인 유사도. 레이어가 입력을 거의 안 바꾸면(유사도≈1) 그
     레이어는 있으나 마나 하다는 뜻이라 중요도가 낮다고 본다.
  5. 중요도 점수 오름차순(=가장 안 중요한 레이어부터)으로 순위를 매긴다.

비교: SmolVLA가 실제로 어떤 레이어를 스킵하는지는
`third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`의
`SmolVLMWithExpertModel.__init__`에서 결정된다:
    if num_vlm_layers > 0:
        self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
즉 "중요도"와는 무관하게 그냥 **앞쪽 `num_vlm_layers`(기본 16, `configuration_smolvla.py`의
`SmolVLAConfig.num_vlm_layers`)개만 남기고 뒤쪽을 통째로 버리는" 방식이다. 이 스크립트는 그
"뒤쪽 절반을 버린다"는 선택과, 코사인 유사도 기준 "중요도 하위 N개(N=버려지는 레이어 수)"가
실제로 얼마나 겹치는지 계산한다.

Colab/Kaggle 셀 예시:
    !git clone <이 레포 URL> ARD-VLA
    %cd ARD-VLA
    !pip install -e "third_party/lerobot[smolvla,peft]"
    !python scripts/layer_importance.py

주의 — 이 스크립트는 GPU와 Hugging Face Hub 접근이 없는 샌드박스에서 작성돼서 실제로
돌려보지 못했다. 검증한 것과 못 한 것:
  - forward hook으로 각 레이어의 입력/출력을 뽑는 방식은 실제 설치된 `transformers`
    라이브러리(4.57.6)의 `LlamaDecoderLayer.forward()` 소스를 직접 읽고 맞췄다 — 이
    레이어는 `hidden_states`를 첫 번째 위치 인자로 받고, 튜플이 아니라 텐서 하나를 그대로
    반환한다(`return hidden_states`, 44번째 줄 근방 참고). SmolVLM2의 text_model은
    `AutoModel.from_config(config.text_config)`으로 만들어지는데, SmolLM2는 Llama 계열이라
    이 클래스가 그대로 쓰인다. 다만 실제 SmolVLM2 config를 받아본 적은 없어서, 만약 다른
    아키텍처(예: 튜플을 반환하는 구현)로 바뀌면 훅의 출력 파싱 부분(`output[0] if
    isinstance(output, tuple) else output`)이 그 경우까지는 커버하지만, 그 외의 시그니처
    차이는 실행 전까지 확신할 수 없다.
  - `GradientCheckpointingLayer.__call__`이 `nn.Module.__call__`을 오버라이드하지만,
    gradient checkpointing이 꺼져있고(`self.training=False`) `super().__call__()`으로
    위임하는 경로를 그대로 타므로 `register_forward_hook`이 정상 동작해야 한다 — 이것도
    소스를 읽고 확인했지만 실제 모델로 실행 검증은 못 했다.
  - 대표 배치는 SmolVLA가 실제로 만드는 멀티모달 prefix(이미지 토큰 + 언어 토큰을 이어붙인
    것)를 흉내 낸 것이지, cross-attention/action expert가 끼어드는 실제 학습 시점의 정확한
    forward 경로는 아니다 — 순수하게 "SmolLM2 레이어 하나하나가 표현을 얼마나 바꾸는지"만
    보려고 일부러 단순화했다.
"""

import argparse
import logging

import torch
import torch.nn.functional as F

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--num-vlm-layers",
        type=int,
        default=16,
        help="SmolVLA가 현재 실제로 사용 중인 num_vlm_layers 값 (SmolVLAConfig 기본값과 동일) — "
        "비교 기준으로만 쓰이고, 이 스크립트가 분석하는 모델 자체는 항상 원본 레이어 수 그대로 로드한다.",
    )
    parser.add_argument("--bottom-n", type=int, default=None, help="중요도 하위 몇 개를 뽑을지 (기본: 원본 - num_vlm_layers, 즉 SmolVLA가 실제로 버리는 개수와 동일)")
    parser.add_argument("--batch-size", type=int, default=4, help="대표 샘플 배치 크기")
    parser.add_argument("--cameras", type=int, default=3, help="top + 좌손목 + 우손목")
    parser.add_argument("--image-size", type=int, default=128, help="더미 이미지 한 변 길이")
    parser.add_argument("--lang-seq-len", type=int, default=48, help="config.tokenizer_max_length 기본값과 동일")
    parser.add_argument("--device", default=None, help="'cuda' 또는 'cpu'. 기본은 자동 감지 (GPU 있으면 GPU, 없으면 CPU로 느리게 실행)")
    return parser.parse_args()


def build_representative_inputs(vlm_expert: SmolVLMWithExpertModel, args, device: torch.device):
    """SmolVLA가 실제로 만드는 멀티모달 prefix(이미지 임베딩 + 언어 토큰 임베딩)를 흉내 낸
    더미 시퀀스를 만든다."""
    image_embeds = []
    for _ in range(args.cameras):
        dummy_image = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
        image_embeds.append(vlm_expert.embed_image(dummy_image))
    image_embeds = torch.cat(image_embeds, dim=1)

    vocab_size = vlm_expert.vlm.config.text_config.vocab_size
    lang_tokens = torch.randint(low=0, high=vocab_size, size=(args.batch_size, args.lang_seq_len), device=device)
    lang_embeds = vlm_expert.embed_language_tokens(lang_tokens)

    inputs_embeds = torch.cat([image_embeds, lang_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attention_mask


def capture_layer_io(text_model, inputs_embeds, attention_mask) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """각 디코더 레이어에 forward hook을 걸어서 (입력 hidden state, 출력 hidden state)를
    CPU float32 텐서로 뽑아온다. 원본 텐서는 곧바로 버려서(hook 안에서 .detach().float().cpu())
    32개 레이어치 activation을 GPU에 계속 들고 있지 않도록 한다."""
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            hidden_in = args[0] if args else kwargs.get("hidden_states")
            hidden_out = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = (hidden_in.detach().float().cpu(), hidden_out.detach().float().cpu())

        return hook

    for layer_idx, layer in enumerate(text_model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx), with_kwargs=True))

    try:
        with torch.no_grad():
            text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    return captured


def compute_importance_scores(captured: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> list[dict]:
    rows = []
    for layer_idx in sorted(captured):
        hidden_in, hidden_out = captured[layer_idx]
        # (B, seq_len, hidden) -> (B, seq_len) 코사인 유사도, 배치+토큰 전체 평균
        cos_sim = F.cosine_similarity(hidden_in, hidden_out, dim=-1)
        avg_cos_sim = cos_sim.mean().item()
        importance_score = 1.0 - avg_cos_sim
        rows.append({"layer": layer_idx, "cosine_similarity": avg_cos_sim, "importance_score": importance_score})
    return rows


def print_importance_table(rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda r: r["importance_score"])
    print()
    print("레이어별 중요도 점수 (중요도 점수 오름차순 = 가장 안 중요한 레이어부터)")
    print(f"{'순위':>4} | {'레이어 번호':>10} | {'코사인 유사도':>14} | {'중요도 점수':>12}")
    print("-" * 50)
    for rank, row in enumerate(ranked, start=1):
        print(f"{rank:>4} | {row['layer']:>10} | {row['cosine_similarity']:>14.4f} | {row['importance_score']:>12.4f}")


def print_comparison(rows: list[dict], num_vlm_layers: int, original_num_layers: int, bottom_n: int | None) -> None:
    skip_count = bottom_n if bottom_n is not None else max(original_num_layers - num_vlm_layers, 0)

    ranked = sorted(rows, key=lambda r: r["importance_score"])
    predicted_skip = {row["layer"] for row in ranked[:skip_count]}
    actual_skip = set(range(num_vlm_layers, original_num_layers))

    overlap = predicted_skip & actual_skip
    overlap_ratio = (len(overlap) / skip_count * 100) if skip_count else 0.0
    only_actual = sorted(actual_skip - predicted_skip)  # SmolVLA는 버리지만 중요도 기준으론 안 버려도 될 레이어
    only_predicted = sorted(predicted_skip - actual_skip)  # 중요도 기준으론 버릴 만한데 SmolVLA가 유지 중인 레이어

    print()
    print("=== SmolVLA의 실제 레이어 스킵 vs 코사인 유사도 기준 중요도 하위 레이어 비교 ===")
    print(
        f"SmolVLA 스킵 결정 위치: smolvlm_with_expert.py의 "
        f"`text_model.layers[:num_vlm_layers]` (config.num_vlm_layers={num_vlm_layers}, "
        f"원본 레이어 수={original_num_layers})"
    )
    print(f"SmolVLA가 실제로 스킵 중인 레이어 ({len(actual_skip)}개): {sorted(actual_skip)}")
    print(f"중요도 하위 {skip_count}개 레이어 (코사인 유사도 기준 스킵 후보): {sorted(predicted_skip)}")
    print(f"겹치는 레이어: {len(overlap)} / {skip_count} (overlap 비율: {overlap_ratio:.1f}%)")
    print(f"SmolVLA만 스킵 중 (중요도 기준으론 안 버려도 될 수 있는 레이어): {only_actual if only_actual else '없음'}")
    print(f"중요도 기준으론 스킵 후보인데 SmolVLA가 유지 중인 레이어: {only_predicted if only_predicted else '없음'}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logging.warning("GPU가 없어 CPU로 실행합니다 — SmolVLM2-500M 순전파 1회라 느리긴 해도 끝나긴 합니다.")

    logging.info(
        "설정: vlm_model_name=%s num_vlm_layers(SmolVLA 기준)=%d batch_size=%d cameras=%d device=%s",
        args.vlm_model_name, args.num_vlm_layers, args.batch_size, args.cameras, device,
    )

    # num_vlm_layers를 넘기지 않으면(기본값 -1) 트림 로직이 건너뛰어져서 원본 레이어 수 그대로 로드된다.
    vlm_expert = SmolVLMWithExpertModel(model_id=args.vlm_model_name, load_vlm_weights=True, device=str(device))
    vlm_expert.to(device)
    vlm_expert.eval()

    text_model = vlm_expert.get_vlm_model().text_model
    original_num_layers = len(text_model.layers)
    logging.info("원본 SmolLM2 레이어 수: %d", original_num_layers)

    if args.num_vlm_layers > original_num_layers:
        raise SystemExit(
            f"--num-vlm-layers({args.num_vlm_layers})가 원본 레이어 수({original_num_layers})보다 큽니다."
        )

    inputs_embeds, attention_mask = build_representative_inputs(vlm_expert, args, device)
    logging.info("대표 배치 시퀀스 길이: %d (이미지 토큰 + 언어 토큰 %d)", inputs_embeds.shape[1], args.lang_seq_len)

    captured = capture_layer_io(text_model, inputs_embeds, attention_mask)
    rows = compute_importance_scores(captured)

    print_importance_table(rows)
    print_comparison(rows, args.num_vlm_layers, original_num_layers, args.bottom_n)


if __name__ == "__main__":
    main()
