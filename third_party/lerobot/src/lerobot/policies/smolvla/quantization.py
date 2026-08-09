# Copyright 2026 The ARD-VLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QLoRA 스타일 백본 양자화(bitsandbytes 기반) 지원.

VLM 백본을 4bit/8bit로 양자화해서 얼리고, LoRA 어댑터만 원래 정밀도로 학습하는 QLoRA
(Dettmers et al., 2023) 패턴을 SmolVLA에 적용한다. 핵심 함수:
  - `build_bnb_config()`: SmolVLAConfig의 양자화 필드로부터 `transformers.BitsAndBytesConfig`를
    만든다 (`smolvlm_with_expert.py`가 `AutoModelForImageTextToText.from_pretrained(...,
    quantization_config=...)`에 넘긴다).
  - `estimate_backbone_memory_bytes()` / `summarize_quantization_savings()`: 실제로 양자화를
    돌려보지 않고도 파라미터 수만으로 dtype별 메모리를 추정하는 순수 계산 함수 — 아래 "중요한
    제약" 때문에 이 레포 샌드박스에서 direct 실측이 불가능해서 만들었다.

**중요한 제약 (이 프로젝트를 만든 샌드박스에서 직접 확인함)**: bitsandbytes의 4bit/8bit
양자화는 원래 CUDA 커널에 의존하도록 설계됐다.
  - `bnb.nn.Linear4bit`은 CPU에서 forward를 시도하면 즉시
    `AssertionError: FP4 quantization state not initialized. Please call .cuda() ...`로
    **확실하게, 재현 가능하게** 실패한다 — 이건 명확하다. 최신 bitsandbytes(0.50+)는
    `kernels-community`에서 커널을 받아와 CPU 4bit forward를 지원하려는 시도가 있지만, 그러려면
    Hugging Face Hub에서 원격 커널을 내려받아야 한다 — 이 레포 샌드박스는 Hub 접근이 막혀 있어서
    이 경로도 막힌다. **QLoRA는 정확히는 4bit(NF4) 방식을 가리키므로, 이게 가장 중요한 제약이다.**
  - `bnb.nn.Linear8bitLt`는 더 애매하다: CPU에서 forward가 에러 없이 돌고, 첫 forward 호출
    *이후에* `weight.dtype`이 실제로 `torch.int8`로 바뀌며, 같은 초기 가중치의 fp32
    `nn.Linear`와 비교하면 작지만 0이 아닌 출력 차이가 난다(직접 측정: 최대 절대오차
    ≈0.0047 — 진짜 양자화 노이즈처럼 보이는 크기). 그런데 정작 bitsandbytes가 GPU 경로에서
    양자화 검증용으로 노출하는 `weight.SCB`/`weight.CB`(스케일/양자화 텐서) 속성은 CPU
    경로에서 계속 `None`이다 — 즉 "진짜 양자화가 맞는지"를 표준적인 방법으로 확인할 수가
    없다. 이건 이 bitsandbytes 버전의 실험적 CPU 백엔드가 GPU와 다른 내부 경로를 타고 있다는
    뜻으로 보이며, **신뢰하고 쓸 근거가 아니다** — 8bit 쪽은 "확실히 안 된다"도 "확실히
    된다"도 아니고, 검증 불가능한 회색지대라고 보는 게 정확하다.

그래서 이 모듈의 함수들은 (1) 설정값 → `BitsAndBytesConfig` 객체 변환처럼 GPU가 필요 없는
부분과, (2) 순수 산수인 메모리 추정만 이 샌드박스에서 실행 검증했다. 실제 4bit/8bit forward/
backward 자체(특히 QLoRA의 핵심인 4bit)는 진짜 GPU 환경에서 처음 검증해야 한다 —
`scripts/test_quantization.py`의 "알려진 CPU 제약" 절이 위 현상들을 실제로 재현해서 기록해둔다.
"""

import torch

QUANTIZATION_CHOICES = (None, "4bit", "8bit")

# bitsandbytes 방식별 파라미터당 비트 수 추정치.
#   - "8bit"(LLM.int8(), Dettmers et al., 2022): 텐서/행 단위 스케일 계수 오버헤드가 <1% 수준이라
#     무시하고 8 bits/param으로 근사.
#   - "4bit"(NF4, double quantization 없이): block_size=64 기준, 블록당 fp32 absmax 1개
#     = 32 bits / 64 params = 0.5 bits/param 오버헤드 → 4 + 0.5 = 4.5 bits/param.
#   - "4bit_double_quant": QLoRA 논문(Dettmers et al., 2023) Table 상의 실측 수치 —
#     2차 양자화로 블록당 오버헤드를 0.5 → 0.127 bits/param까지 줄여서 4.127 bits/param.
BITS_PER_PARAM = {
    "fp32": 32.0,
    "bf16": 16.0,
    "fp16": 16.0,
    "8bit": 8.0,
    "4bit": 4.5,
    "4bit_double_quant": 4.127,
}


def build_bnb_config(
    quantization: str | None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
):
    """SmolVLAConfig의 양자화 필드로부터 `transformers.BitsAndBytesConfig`를 만든다.
    `quantization`이 None이면 None을 반환한다(양자화 안 함, 기존 동작 그대로).

    참고: 이 함수는 bitsandbytes가 실제로 설치/동작하지 않아도 호출할 수 있다 —
    `BitsAndBytesConfig`는 순수 설정 객체라, 실제로 양자화를 적용하는 시점
    (`AutoModelForImageTextToText.from_pretrained(..., quantization_config=...)`)에서만
    bitsandbytes가 필요하다.
    """
    if quantization is None:
        return None
    if quantization not in ("4bit", "8bit"):
        raise ValueError(f"quantization은 None, '4bit', '8bit' 중 하나여야 합니다. 받은 값: {quantization!r}")

    from transformers import BitsAndBytesConfig

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
    )


def estimate_backbone_memory_bytes(num_params: int, dtype: str) -> float:
    """`dtype`(BITS_PER_PARAM의 키)으로 `num_params`개 파라미터를 저장하는 데 필요한 바이트 수
    추정치. 활성화(activation)/옵티마이저 상태/LoRA 어댑터 자체의 메모리는 포함하지 않는다 —
    "정지된 백본 가중치 저장 용량"만 계산한다."""
    if dtype not in BITS_PER_PARAM:
        raise ValueError(f"알 수 없는 dtype: {dtype!r}. 선택지: {list(BITS_PER_PARAM)}")
    return num_params * BITS_PER_PARAM[dtype] / 8.0


def summarize_quantization_savings(num_params: int, baseline: str = "bf16") -> dict[str, dict[str, float]]:
    """BITS_PER_PARAM에 있는 모든 dtype에 대해 (바이트 수, `baseline` 대비 절감률)을 계산한다."""
    baseline_bytes = estimate_backbone_memory_bytes(num_params, baseline)
    summary = {}
    for dtype in BITS_PER_PARAM:
        num_bytes = estimate_backbone_memory_bytes(num_params, dtype)
        summary[dtype] = {
            "bytes": num_bytes,
            "mb": num_bytes / (1024**2),
            "gb": num_bytes / (1024**3),
            "reduction_vs_baseline": 1.0 - num_bytes / baseline_bytes,
        }
    return summary
