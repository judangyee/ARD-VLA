#!/usr/bin/env python3
"""SmolVLA(+ARD) 메모리 프로파일링 — LoRA + bf16 + gradient checkpointing을 켠 상태에서
배치 사이즈별 forward+backward 최대 GPU 메모리 사용량을 측정한다.

Colab/Kaggle 노트북 셀에 그대로 실행할 수 있도록 단일 파일로 짰다. GPU가 필요하다
(torch.cuda.max_memory_allocated 기반). 더미 이미지/언어 토큰으로 배치를 만들기 때문에
실제 데이터셋 없이 바로 돌아가지만, 그만큼 "숫자가 맞다"가 아니라 "메모리가 얼마나 드는지"만
알려준다.

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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
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
    parser.set_defaults(use_lora=True, use_bf16=True, use_grad_checkpoint=True, use_ard=True)
    return parser.parse_args()


def build_policy(args) -> SmolVLAPolicy:
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
    }
    for i in range(args.cameras):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))}

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,  # 실제 사전학습 백본 — 메모리 프로파일링은 실제 가중치 기준이어야 의미 있음
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU가 없습니다 — 이 스크립트는 torch.cuda.max_memory_allocated 기반이라 GPU가 필수입니다. "
            "Colab/Kaggle에서 런타임을 GPU로 설정했는지 확인하세요."
        )
    device = torch.device("cuda")

    logging.info(
        "설정: LoRA=%s(r=%d) bf16=%s grad_checkpoint=%s ARD=%s chunk_size=%d action_dim=%d cameras=%d",
        args.use_lora, args.lora_r, args.use_bf16, args.use_grad_checkpoint, args.use_ard,
        args.chunk_size, args.action_dim, args.cameras,
    )

    policy = build_policy(args)

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

    print()
    print(f"{'batch_size':>10} | {'peak memory (GB)':>16}")
    print("-" * 30)
    for batch_size, peak_gb, note in results:
        value = f"{peak_gb:.2f}" if peak_gb is not None else (note or "-")
        print(f"{batch_size:>10} | {value:>16}")


if __name__ == "__main__":
    main()
