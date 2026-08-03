#!/usr/bin/env python3
"""기본 SmolVLA(단일팔 7 DoF, 원본 액션 헤드) vs ARD-VLA(bimanual 14 DoF,
AsymmetricResidualHeads 적용)를 같은 배치 사이즈로 나란히 비교 프로파일링한다.

`scripts/profile_memory.py`를 참고해서 만들었다 — LoRA + bf16 autocast + gradient
checkpointing이 기본으로 켜져있는 것도 동일하다. 차이는 "레이어 프루닝 O/X"가 아니라
"기본 SmolVLA vs ARD-VLA" 두 모델 설정을 비교한다는 점이다. 비교 항목:
  - 전체 파라미터 수
  - LoRA 학습 대상(trainable) 파라미터 수
  - 배치 사이즈별([1, 4, 16, 32] 기본값) forward+backward 최대 메모리(GB)
  - 배치 사이즈별 forward pass 시간(ms) — backward는 시간에서 제외, 메모리에는 포함
    (실제 학습 스텝의 메모리 최고점은 backward에서 나오는 게 보통이라 메모리는 계속
    forward+backward 기준으로 재고, 시간만 forward 단독으로 잰다)

두 변형은 액션/상태 차원과 ARD 사용 여부가 달라서 모델 구조 자체가 다르다 — 정책을 공유할 수
없어 변형마다 새로 빌드하고, 끝나면 GPU 메모리를 비우고 다음 변형으로 넘어간다.

Colab/Kaggle 셀 예시:
    !git clone <이 레포 URL> ARD-VLA
    %cd ARD-VLA
    !pip install -e "third_party/lerobot[smolvla,peft]"
    !python scripts/compare_smolvla_ard.py

주의 — 이 스크립트는 `profile_memory.py`와 마찬가지로 GPU와 Hugging Face Hub 접근이 없는
샌드박스에서 작성돼서, 로직은 `profile_memory.py`(실제 Colab GPU에서 검증됨)와 동일한
패턴을 그대로 따랐지만 이 스크립트 자체는 실행해보지 못했다. forward 단독 타이밍은 별도
warmup 없이 첫 호출을 그대로 재기 때문에, 배치 사이즈 1의 첫 결과는 CUDA 커널
초기화/캐싱 비용이 섞여 실제보다 느리게 나올 수 있다 — 여러 번 돌려서 재현되는 추세만
비교하는 걸 권장한다. Colab에서 `pip install`이나 `wrap_with_peft()` 관련 이슈가 나오면
README의 "메모리 프로파일링" 절에 정리된 두 가지 Colab 이슈(torch 재설치 후 런타임 재시작,
peft-torchao 버전 충돌)를 먼저 확인하자.
"""

import argparse
import gc
import logging
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

VARIANTS = {
    "base_smolvla": {
        "label": "기본 SmolVLA (단일팔 7 DoF, 원본 액션 헤드)",
        "short_label": "기본 SmolVLA",
        "state_dim": 7,
        "action_dim": 7,
        "use_ard": False,
    },
    "ard_vla": {
        "label": "ARD-VLA (bimanual 14 DoF, AsymmetricResidualHeads)",
        "short_label": "ARD-VLA",
        "state_dim": 14,
        "action_dim": 14,
        "use_ard": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--chunk-size", type=int, default=50, help="action chunk length (horizon)")
    parser.add_argument(
        "--ard-arm-dim", type=int, default=7, help="ARD-VLA 변형에서 팔 하나당 자유도 (2*ard_arm_dim=action_dim)"
    )
    parser.add_argument("--cameras", type=int, default=3, help="top + 좌손목 + 우손목")
    parser.add_argument("--image-size", type=int, default=128, help="더미 이미지 한 변 길이 (resize_imgs_with_padding이 어차피 재조정함)")
    parser.add_argument("--lang-seq-len", type=int, default=48, help="config.tokenizer_max_length 기본값과 동일")
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="LoRA 끄고 프로파일링")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false", help="bf16 autocast 끄기")
    parser.add_argument("--no-grad-checkpoint", dest="use_grad_checkpoint", action="store_false")
    parser.add_argument(
        "--variants",
        choices=["both", "base", "ard"],
        default="both",
        help="'both'(기본)면 두 변형을 모두 돌려서 비교표를 낸다. 'base'/'ard'면 그중 하나만 실행한다.",
    )
    parser.set_defaults(use_lora=True, use_bf16=True, use_grad_checkpoint=True)
    return parser.parse_args()


def build_policy_for_variant(args, variant_key: str) -> SmolVLAPolicy:
    variant = VARIANTS[variant_key]
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(variant["state_dim"],)),
    }
    for i in range(args.cameras):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(variant["action_dim"],))}

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,  # 실제 사전학습 백본 — 메모리/시간 비교는 실제 가중치 기준이어야 의미 있음
        pretrained_path=args.vlm_model_name if args.use_lora else None,  # PEFT의 "from-scratch 경고" 통과용
        use_ard=variant["use_ard"],
        ard_arm_dim=args.ard_arm_dim,
        tokenizer_max_length=args.lang_seq_len,
        device="cuda",
    )
    policy = SmolVLAPolicy(config)
    policy.to("cuda")
    return policy


def apply_lora(policy: SmolVLAPolicy, args, use_ard: bool) -> None:
    """LoRA를 씌운다. use_ard=True인 변형만 AsymmetricResidualHeads가 존재하므로, 그 경우에만
    ARD head를 명시적으로 다시 학습 가능하게 풀어준다 (wrap_with_peft()의 기본 타겟에는 ARD head가
    없어서, 그대로 두면 통째로 얼어붙는다).
    """
    peft_model = policy.wrap_with_peft(peft_cli_overrides={"r": args.lora_r, "lora_alpha": args.lora_alpha})
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    if use_ard:
        n = 0
        for p in policy.model.ard_heads.parameters():
            p.requires_grad_(True)
            n += p.numel()
        logging.info("LoRA 적용 후 AsymmetricResidualHeads %d개 파라미터를 다시 학습 가능하게 풀었습니다.", n)


def build_dummy_batch(policy: SmolVLAPolicy, args, variant: dict, batch_size: int, device: torch.device) -> dict:
    config = policy.config
    batch = {}
    for key in config.image_features:
        batch[key] = torch.rand(batch_size, 3, args.image_size, args.image_size, device=device)
    batch[OBS_STATE] = torch.randn(batch_size, variant["state_dim"], device=device)
    batch[ACTION] = torch.randn(batch_size, args.chunk_size, variant["action_dim"], device=device)

    vocab_size = policy.model.vlm_with_expert.vlm.config.text_config.vocab_size
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(
        low=0, high=vocab_size, size=(batch_size, args.lang_seq_len), device=device
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        batch_size, args.lang_seq_len, dtype=torch.bool, device=device
    )
    return batch


def profile_one_batch_size(
    policy: SmolVLAPolicy, args, variant: dict, batch_size: int, device: torch.device, autocast_dtype
) -> tuple[float, float]:
    """forward+backward 1회를 돌리고 (forward 단독 시간(ms), 최대 할당 메모리(GB))를 반환한다."""
    policy.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch = build_dummy_batch(policy, args, variant, batch_size, device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if autocast_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            loss, _ = policy.forward(batch)
    else:
        loss, _ = policy.forward(batch)
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1000

    loss.backward()
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return forward_ms, peak_gb


def run_variant_sweep(args, variant_key: str):
    variant = VARIANTS[variant_key]
    device = torch.device("cuda")
    logging.info("=== %s ===", variant["label"])

    policy = build_policy_for_variant(args, variant_key)
    total_params = sum(p.numel() for p in policy.parameters())

    if args.use_grad_checkpoint:
        policy.model.vlm_with_expert.gradient_checkpointing_enable()

    policy.train()

    if args.use_lora:
        apply_lora(policy, args, use_ard=variant["use_ard"])
    else:
        for p in policy.parameters():
            p.requires_grad_(True)

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(
        "전체 파라미터: %s / 학습 가능: %s (%.2f%%)",
        f"{total_params:,}", f"{trainable_params:,}", 100 * trainable_params / total_params,
    )

    autocast_dtype = torch.bfloat16 if args.use_bf16 else None

    results = []
    for batch_size in args.batch_sizes:
        try:
            forward_ms, peak_gb = profile_one_batch_size(policy, args, variant, batch_size, device, autocast_dtype)
            results.append((batch_size, peak_gb, forward_ms, None))
            logging.info(
                "[%s] batch_size=%d -> peak %.2f GB, forward %.1f ms",
                variant["short_label"], batch_size, peak_gb, forward_ms,
            )
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            results.append((batch_size, None, None, "OOM"))
            logging.warning("[%s] batch_size=%d -> OOM (%s)", variant["short_label"], batch_size, str(e).splitlines()[0])
            torch.cuda.empty_cache()

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return variant, total_params, trainable_params, results


def print_param_summary(all_results) -> None:
    print()
    print(f"{'변형':50s} | {'전체 파라미터 수':>16s} | {'LoRA 학습 대상 파라미터 수':>26s}")
    print("-" * 100)
    for variant, total_params, trainable_params, _ in all_results:
        pct = 100 * trainable_params / total_params if total_params else 0.0
        print(f"{variant['label']:50s} | {total_params:16,d} | {trainable_params:16,d} ({pct:5.2f}%)")


def print_comparison_table(all_results, batch_sizes) -> None:
    print()
    col_width = 22
    header = f"{'batch_size':>10}"
    for variant, *_rest in all_results:
        header += f" | {variant['short_label'] + ' peak GB':>{col_width}} | {variant['short_label'] + ' fwd ms':>{col_width}}"
    print(header)
    print("-" * len(header))

    for i, batch_size in enumerate(batch_sizes):
        row = f"{batch_size:>10}"
        for _variant, _total, _trainable, results in all_results:
            _bs, peak_gb, forward_ms, note = results[i]
            gb_str = f"{peak_gb:.2f}" if peak_gb is not None else (note or "-")
            ms_str = f"{forward_ms:.1f}" if forward_ms is not None else (note or "-")
            row += f" | {gb_str:>{col_width}} | {ms_str:>{col_width}}"
        print(row)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU가 없습니다 — 이 스크립트는 torch.cuda.max_memory_allocated 기반이라 GPU가 필수입니다. "
            "Colab/Kaggle에서 런타임을 GPU로 설정했는지 확인하세요."
        )

    variant_keys = []
    if args.variants in ("both", "base"):
        variant_keys.append("base_smolvla")
    if args.variants in ("both", "ard"):
        variant_keys.append("ard_vla")

    logging.info(
        "설정: LoRA=%s(r=%d) bf16=%s grad_checkpoint=%s batch_sizes=%s variants=%s",
        args.use_lora, args.lora_r, args.use_bf16, args.use_grad_checkpoint, args.batch_sizes, args.variants,
    )

    all_results = [run_variant_sweep(args, key) for key in variant_keys]

    print_param_summary(all_results)
    print_comparison_table(all_results, args.batch_sizes)


if __name__ == "__main__":
    main()
