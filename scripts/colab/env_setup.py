"""GPU 감지 + dtype/배치 크기 자동 선택.

무료 Colab은 어떤 GPU가 배정될지 보장이 없다(T4 16GB가 가장 흔하지만 A100/L4/
CPU-only도 나올 수 있음) — 매번 실행 시점에 실제로 꽂힌 GPU를 코드로 확인하고
분기한다. 특히 **bf16 하드웨어 지원 여부**를 `torch.cuda.is_bf16_supported()`의
`including_emulation` 인자로 정확히 구분한다 — 이 인자를 기본값(`True`)으로 두면
네이티브 지원이 없는 GPU(T4 등, Ampere 이전 Turing 아키텍처)에서도 "에뮬레이션으로는
가능"이라는 이유로 `True`가 나올 수 있어서(PyTorch 문서 기준 동작, 이 레포에서 GPU로
직접 재현하지는 못했다 — 미검증, 아래 `detect_gpu()` docstring 참고), 실제 하드웨어
가속 여부만 보려면 `including_emulation=False`로 호출해야 한다.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

AmpMode = Literal["bf16", "fp16_scaled", "none"]


@dataclasses.dataclass
class GpuInfo:
    available: bool
    name: str | None = None
    vram_gb: float | None = None
    compute_capability: tuple[int, int] | None = None
    bf16_native: bool = False


@dataclasses.dataclass
class DtypeChoice:
    dtype: torch.dtype
    amp_mode: AmpMode
    reason: str


@dataclasses.dataclass
class ComputeEnv:
    gpu: GpuInfo
    dtype_choice: DtypeChoice
    batch_size: int
    batch_size_source: Literal["manual", "auto"]

    def summary(self) -> str:
        g = self.gpu
        lines = [
            f"GPU: {'없음(CPU)' if not g.available else g.name}",
        ]
        if g.available:
            lines.append(f"  VRAM: {g.vram_gb:.1f} GB" if g.vram_gb is not None else "  VRAM: 조회 실패")
            lines.append(f"  compute capability: {g.compute_capability}")
            lines.append(f"  bf16 네이티브 지원: {g.bf16_native}")
        lines.append(
            f"dtype 선택: {self.dtype_choice.dtype} (amp_mode={self.dtype_choice.amp_mode}) — "
            f"{self.dtype_choice.reason}"
        )
        lines.append(f"batch_size: {self.batch_size} ({self.batch_size_source})")
        return "\n".join(lines)


def detect_gpu() -> GpuInfo:
    """현재 프로세스가 보는 GPU 정보를 조회한다. GPU가 없으면 `available=False`만
    채워진 `GpuInfo`를 돌려준다(예외를 던지지 않음 — CPU 전용 환경도 정상 경로다).

    `bf16_native`는 `torch.cuda.is_bf16_supported(including_emulation=False)`로
    확인한다 — **이 호출 하나가 T4(네이티브 bf16 없음)와 A100/L4(네이티브 bf16 있음)를
    가르는 지점**이다. 이 함수 자체는 실제 GPU가 있는 이 샌드박스 밖(Colab)에서
    최종 확인해야 한다 — 이 레포 샌드박스에는 GPU가 없어(`torch.cuda.is_available()
    == False`, 확인됨) 분기 자체는 유닛 테스트로 몽키패치해서 검증했지만 실제 T4 위에서
    돌려본 적은 없다(미검증, `docs/colab_free_tier.md` 참고)."""
    if not torch.cuda.is_available():
        return GpuInfo(available=False)

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    capability = torch.cuda.get_device_capability(0)
    bf16_native = torch.cuda.is_bf16_supported(including_emulation=False)
    return GpuInfo(
        available=True,
        name=name,
        vram_gb=vram_gb,
        compute_capability=capability,
        bf16_native=bf16_native,
    )


def select_dtype(gpu: GpuInfo, force: str | None = None) -> DtypeChoice:
    """GPU 정보로부터 학습에 쓸 dtype/AMP 모드를 고른다.

    - GPU 없음(CPU): fp32. fp16/bf16은 CPU에서 지원이 느리거나 불완전해서
      스모크 테스트조차 불안정해질 수 있다는 일반적으로 알려진 사실에 근거한다
      (이 레포에서 CPU fp16을 실측 비교하지는 않았다 — 미검증이지만, 애초에
      CPU 학습은 스모크 검증용일 뿐 실제 학습 대상이 아니므로 안전한 쪽을 택한다).
    - bf16 네이티브 지원(Ampere+, compute capability >= 8): bf16. fp16과 달리
      지수 비트가 fp32와 같아 `GradScaler` 같은 손실 스케일링이 필요 없고,
      그만큼 구현이 단순하고 안정적이다(bf16의 일반적으로 알려진 장점 — 이
      레포에서 실측 비교는 하지 않았다, 미검증).
    - bf16 미지원(T4 등 Turing, compute capability < 8): fp16 autocast +
      `GradScaler`. fp16 가중치를 그대로 쓰면(스케일링 없이) 작은 gradient가
      언더플로로 0이 되는 문제가 알려져 있어(일반적으로 알려진 사실, 미검증),
      autocast(fp16 연산)+GradScaler(손실 스케일링) 조합을 쓴다 — PyTorch가
      공식으로 권장하는 혼합정밀도 패턴.

    `force`("bf16"/"fp16"/"fp32")를 주면 위 자동 판단을 무시하고 그 dtype을
    강제한다(디버깅/비교용) — 다만 GPU가 bf16을 네이티브로 지원하지 않는데
    "bf16"을 강제하면 PyTorch가 소프트웨어 에뮬레이션으로 느리게 돌릴 수 있다는
    경고를 reason에 남긴다.
    """
    if force is not None:
        force = force.lower()
        if force == "bf16":
            reason = "사용자가 --dtype bf16으로 강제 지정함"
            if gpu.available and not gpu.bf16_native:
                reason += " (경고: 이 GPU는 bf16 네이티브 지원이 없어 에뮬레이션으로 느릴 수 있음)"
            return DtypeChoice(torch.bfloat16, "bf16", reason)
        if force == "fp16":
            return DtypeChoice(torch.float32, "fp16_scaled", "사용자가 --dtype fp16으로 강제 지정함(autocast+GradScaler)")
        if force == "fp32":
            return DtypeChoice(torch.float32, "none", "사용자가 --dtype fp32로 강제 지정함")
        raise ValueError(f"알 수 없는 --dtype 값: {force!r} (bf16/fp16/fp32 중 하나)")

    if not gpu.available:
        return DtypeChoice(torch.float32, "none", "GPU 없음(CPU) — fp32가 가장 안전")

    if gpu.bf16_native:
        return DtypeChoice(
            torch.bfloat16,
            "bf16",
            f"GPU({gpu.name}, compute capability {gpu.compute_capability})가 bf16을 네이티브로 지원함"
            " — GradScaler 없이 바로 사용",
        )

    return DtypeChoice(
        torch.float32,
        "fp16_scaled",
        f"GPU({gpu.name}, compute capability {gpu.compute_capability})가 bf16 네이티브 미지원"
        "(T4류 Turing 추정) — fp16 autocast + GradScaler 사용",
    )


# vram_gb -> 배치 크기. **실측하지 않은 추정치**(SmolVLA + 카메라 3대 + chunk_size=20
# 기준으로 "OOM 없이 도는 쪽"을 보수적으로 추정한 것일 뿐, 실제로 이 크기에서 돌려본
# 적이 없다 — 첫 실행에서 OOM이 나면 더 낮춰야 한다). 반드시 `docs/colab_free_tier.md`의
# "알려진 한계"에서 이 추정치가 미검증임을 같이 언급한다.
_AUTO_BATCH_SIZE_TABLE: list[tuple[float, int]] = [
    (10.0, 1),
    (20.0, 2),
    (40.0, 4),
]
_AUTO_BATCH_SIZE_DEFAULT = 8


def select_batch_size(gpu: GpuInfo, manual: int | None = None) -> tuple[int, Literal["manual", "auto"]]:
    if manual is not None:
        if manual <= 0:
            raise ValueError(f"--batch-size는 양수여야 합니다: {manual}")
        return manual, "manual"

    if not gpu.available or gpu.vram_gb is None:
        return 1, "auto"

    for threshold, size in _AUTO_BATCH_SIZE_TABLE:
        if gpu.vram_gb < threshold:
            return size, "auto"
    return _AUTO_BATCH_SIZE_DEFAULT, "auto"


def detect_compute_env(manual_batch_size: int | None = None, force_dtype: str | None = None) -> ComputeEnv:
    gpu = detect_gpu()
    dtype_choice = select_dtype(gpu, force=force_dtype)
    batch_size, source = select_batch_size(gpu, manual=manual_batch_size)
    return ComputeEnv(gpu=gpu, dtype_choice=dtype_choice, batch_size=batch_size, batch_size_source=source)
