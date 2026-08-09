#!/usr/bin/env python3
"""QLoRA 스타일 백본 양자화(`lerobot.policies.smolvla.quantization`)에 대한 오프라인 단위 테스트.

GPU 없이 검증 가능한 부분(설정 검증, BitsAndBytesConfig 구성, 메모리 이론값 계산, LoRA 타겟
정규식)은 전부 여기서 실제로 실행해서 확인한다. 4bit/8bit 양자화 forward/backward 자체는
bitsandbytes가 CUDA 커널에 의존해서 이 샌드박스(GPU 없음)에서는 검증할 수 없다 — 대신
`test_known_cpu_limitations()`가 그 한계를 실제로 재현해서 기록해둔다 (bitsandbytes가 설치돼
있을 때만 실행되고, 없으면 건너뛴다).
"""

import sys

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.quantization import (
    BITS_PER_PARAM,
    build_bnb_config,
    estimate_backbone_memory_bytes,
    summarize_quantization_savings,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_build_bnb_config():
    check("quantization=None -> build_bnb_config는 None", build_bnb_config(None) is None)

    cfg8 = build_bnb_config("8bit")
    check("8bit config: load_in_8bit=True", cfg8.load_in_8bit is True)
    check("8bit config: load_in_4bit=False", cfg8.load_in_4bit is False)

    cfg4 = build_bnb_config(
        "4bit", bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16"
    )
    check("4bit config: load_in_4bit=True", cfg4.load_in_4bit is True)
    check("4bit config: quant_type이 그대로 전달된다", cfg4.bnb_4bit_quant_type == "nf4")
    check("4bit config: double_quant이 그대로 전달된다", cfg4.bnb_4bit_use_double_quant is True)
    check("4bit config: compute_dtype이 문자열->torch.dtype으로 변환된다", cfg4.bnb_4bit_compute_dtype == torch.bfloat16)

    try:
        build_bnb_config("2bit")
        check("잘못된 quantization 값을 거부한다", False)
    except ValueError:
        check("잘못된 quantization 값을 거부한다", True)


def test_config_validation():
    cfg = SmolVLAConfig(quantization="4bit", load_vlm_weights=True)
    check("SmolVLAConfig(quantization='4bit')가 정상 생성된다", cfg.quantization == "4bit")

    try:
        SmolVLAConfig(quantization="4bit", load_vlm_weights=False)
        check("load_vlm_weights=False와 quantization 동시 사용을 거부한다", False)
    except ValueError:
        check("load_vlm_weights=False와 quantization 동시 사용을 거부한다", True)

    try:
        SmolVLAConfig(quantization="16bit")
        check("잘못된 quantization 문자열을 거부한다", False)
    except ValueError:
        check("잘못된 quantization 문자열을 거부한다", True)

    try:
        SmolVLAConfig(quantization="4bit", load_vlm_weights=True, bnb_4bit_compute_dtype="int8")
        check("잘못된 bnb_4bit_compute_dtype을 거부한다", False)
    except ValueError:
        check("잘못된 bnb_4bit_compute_dtype을 거부한다", True)


def test_memory_estimation():
    n = 450_000_000  # SmolVLM2 백본 근사 파라미터 수
    b_bf16 = estimate_backbone_memory_bytes(n, "bf16")
    b_8bit = estimate_backbone_memory_bytes(n, "8bit")
    b_4bit = estimate_backbone_memory_bytes(n, "4bit")
    b_4bit_dq = estimate_backbone_memory_bytes(n, "4bit_double_quant")

    check("8bit이 bf16보다 작다", b_8bit < b_bf16)
    check("4bit이 8bit보다 작다", b_4bit < b_8bit)
    check("double quant이 double quant 없는 4bit보다 작거나 같다", b_4bit_dq <= b_4bit)
    check(
        "8bit이 bf16 대비 정확히 절반이다 (8/16 bits)",
        abs(b_8bit / b_bf16 - 0.5) < 1e-9,
        detail=f"ratio={b_8bit / b_bf16}",
    )

    summary = summarize_quantization_savings(n, baseline="bf16")
    check("summarize_quantization_savings가 BITS_PER_PARAM의 모든 dtype을 포함한다", set(summary.keys()) == set(BITS_PER_PARAM))
    check("baseline(bf16) 자체의 절감률은 0이다", abs(summary["bf16"]["reduction_vs_baseline"]) < 1e-9)
    check(
        "4bit(double quant)의 bf16 대비 절감률이 70% 이상이다",
        summary["4bit_double_quant"]["reduction_vs_baseline"] > 0.70,
        detail=f"{summary['4bit_double_quant']['reduction_vs_baseline']:.4f}",
    )


def test_lora_target_regex_includes_quantized_backbone():
    """_get_default_peft_targets()가 quantization 활성화 시 VLM 백본의 attention projection도
    타겟에 포함시키는지, modeling_smolvla.py를 import하지 않고(Hub 필요 없음) 동일한 정규식
    구성 로직을 그대로 재현해서 검증한다."""
    import re

    common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
    base = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
    quantized = base[:-1] + r"|model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|v)_proj)"

    check(
        "양자화 없으면 VLM 백본 레이어는 LoRA 타겟이 아니다",
        re.search(base, "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj") is None,
    )
    check(
        "양자화 켜면 VLM 백본의 q/v_proj가 LoRA 타겟에 포함된다",
        re.search(quantized, "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj") is not None,
    )
    check(
        "양자화 켜도 k_proj/mlp는 타겟이 아니다 (q/v만)",
        re.search(quantized, "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj") is None,
    )
    check(
        "양자화 켜도 vision_model은 타겟이 아니다 (text_model만)",
        re.search(quantized, "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj") is None,
    )
    check(
        "기존 action expert/shim 타겟은 양자화 여부와 무관하게 그대로 유지된다",
        re.search(quantized, "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj") is not None
        and re.search(quantized, "model.action_out_proj") is not None,
    )


def test_known_cpu_limitations():
    """bitsandbytes 4bit/8bit이 CPU에서 실제로 어떻게 동작(실패/애매하게 동작)하는지 직접
    재현해서 기록해둔다 — 레이어를 일부러 어떤 device로도 옮기지 않고 그대로 쓴다 (기본
    device가 CPU이므로). bitsandbytes가 설치돼 있지 않으면 건너뛴다."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("[SKIP] bitsandbytes가 설치돼 있지 않아 CPU 한계 재현 테스트를 건너뜁니다.")
        return

    # 1) Linear4bit는 CPU에서 forward 자체가 안 된다 (.cuda() 없이는 양자화 상태 초기화가 안 됨).
    #    QLoRA는 정확히는 4bit(NF4)를 가리키므로, 이게 가장 확실하고 중요한 제약이다.
    try:
        layer4 = bnb.nn.Linear4bit(16, 8, bias=False, compute_dtype=torch.float32)
        layer4(torch.randn(2, 16))
        check("Linear4bit은 CPU에서 forward가 실패해야 한다 (알려진 제약)", False, detail="예상과 달리 성공함")
    except AssertionError:
        check("Linear4bit은 CPU에서 forward가 실패해야 한다 (알려진 제약)", True)

    # 2) Linear8bitLt는 CPU에서 forward가 에러 없이 돌고, 첫 forward 이후 weight.dtype이
    #    실제로 int8로 바뀌며, 같은 초기 가중치의 fp32 nn.Linear와 출력이 (작지만) 다르다 --
    #    "전혀 양자화 안 됨"이라고 단정할 근거는 아니다. 하지만 GPU 경로에서 양자화 검증에
    #    쓰이는 weight.SCB/weight.CB(스케일/양자화 텐서)는 CPU에서 계속 None이라, 표준적인
    #    방법으로 "제대로 양자화됐다"고 확인할 수도 없다 -- 이 테스트는 그 애매한 상태 자체를
    #    기록해두는 것이지, 8bit이 CPU에서 신뢰할 만하다고 주장하는 게 아니다.
    torch.manual_seed(0)
    ref = torch.nn.Linear(16, 8, bias=False)
    layer8 = bnb.nn.Linear8bitLt(16, 8, bias=False, has_fp16_weights=False)
    with torch.no_grad():
        layer8.weight.data.copy_(ref.weight.data)

    weight_dtype_before = dict(layer8.named_parameters())["weight"].dtype
    x = torch.randn(4, 16)
    with torch.no_grad():
        out_ref = ref(x)
        out_bnb = layer8(x)
    weight_dtype_after = dict(layer8.named_parameters())["weight"].dtype
    max_abs_diff = (out_ref - out_bnb).abs().max().item()

    check("Linear8bitLt: forward 전 weight는 fp32다 (아직 양자화 안 됨)", weight_dtype_before == torch.float32)
    check(
        "Linear8bitLt: 첫 forward 후 weight.dtype이 int8로 바뀐다",
        weight_dtype_after == torch.int8,
        detail=f"실제 dtype={weight_dtype_after}",
    )
    check(
        "Linear8bitLt: 같은 가중치의 fp32 nn.Linear와 출력이 완전히 같지는 않다 (양자화 노이즈로 보임)",
        max_abs_diff > 1e-6,
        detail=f"max_abs_diff={max_abs_diff}",
    )
    check(
        "Linear8bitLt: 그런데 GPU 경로의 표준 양자화 검증 속성(SCB/CB)은 CPU에서 None이다 "
        "(그래서 '진짜 제대로' 양자화됐다고 확신할 수는 없음 -- 결론은 여전히 '검증 불가')",
        layer8.weight.SCB is None,
    )


def test_real_gpu_behavior():
    """CUDA가 실제로 있으면(이 샌드박스에서는 항상 없음, GPU 환경에서만 의미가 있다) 레이어를
    진짜 .cuda()로 옮겨서 bitsandbytes의 실제 목적지 경로를 확인한다 — 위 CPU 테스트와 대구를
    이룬다: 거기서 "안 되거나 애매했던" 것들이 진짜 GPU에서는 실제로 되는지 직접 확인."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("[SKIP] bitsandbytes가 설치돼 있지 않아 GPU 동작 테스트를 건너뜁니다.")
        return
    if not torch.cuda.is_available():
        print("[SKIP] CUDA GPU가 없어 GPU 동작 테스트를 건너뜁니다.")
        return

    device = torch.device("cuda")

    # 1) Linear4bit: GPU에서는 forward가 실제로 성공하고, gradient도 입력까지 흘러야 한다
    #    (LoRA 어댑터가 옆에 붙어서 학습되는 실제 QLoRA 상황을 흉내).
    layer4 = bnb.nn.Linear4bit(16, 8, bias=False, compute_dtype=torch.float16).to(device)
    x4 = torch.randn(2, 16, device=device, requires_grad=True)
    out4 = layer4(x4)
    check(
        "GPU: Linear4bit forward가 실제로 성공한다 (CPU에서는 AssertionError였음)",
        out4.shape == (2, 8) and torch.isfinite(out4).all().item(),
    )
    out4.sum().backward()
    check(
        "GPU: Linear4bit을 통과해 입력까지 gradient가 흐른다",
        x4.grad is not None and x4.grad.abs().sum().item() > 0,
    )

    # 2) Linear8bitLt: GPU에서는 SCB(양자화 스케일)가 실제로 채워지는지 (CPU에서는 계속 None이었음).
    #    순서 중요: 원하는 가중치를 먼저 CPU에서 채워 넣고, 그 다음에 .to(device)해야
    #    quantize가 그 값 기준으로 일어난다 (반대로 하면 무작위 초기값이 양자화돼버림).
    torch.manual_seed(0)
    ref = torch.nn.Linear(16, 8, bias=False)
    layer8 = bnb.nn.Linear8bitLt(16, 8, bias=False, has_fp16_weights=False)
    with torch.no_grad():
        layer8.weight.data.copy_(ref.weight.data)
    layer8 = layer8.to(device)

    check(
        "GPU: Linear8bitLt는 .to(device) 시점에 바로 양자화된다 (SCB가 채워짐)",
        layer8.weight.SCB is not None,
    )
    out8 = layer8(torch.randn(4, 16, device=device))
    check("GPU: Linear8bitLt forward가 정상적으로 돈다", out8.shape == (4, 8) and torch.isfinite(out8).all().item())


def main():
    test_build_bnb_config()
    test_config_validation()
    test_memory_estimation()
    test_lora_target_regex_includes_quantized_backbone()
    test_known_cpu_limitations()
    test_real_gpu_behavior()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)}개 항목 실패: {FAILURES}")
        sys.exit(1)
    print("[OK] 양자화(QLoRA) 단위 테스트 전체 통과.")
    if torch.cuda.is_available():
        print("CUDA GPU가 감지되어 test_real_gpu_behavior()가 실제로 4bit/8bit forward/backward를 실행했습니다.")
    else:
        print(
            "참고: 실제 4bit/8bit forward/backward 자체(진짜 양자화된 행렬곱)는 bitsandbytes가 CUDA "
            "커널에 의존해서 GPU가 없는 이 환경에서는 검증할 수 없습니다 — test_known_cpu_limitations()가 "
            "그 한계 자체를 재현해서 기록해뒀습니다. GPU가 있는 환경에서 돌리면 test_real_gpu_behavior()가 "
            "자동으로 실제 검증까지 수행합니다."
        )


if __name__ == "__main__":
    main()
