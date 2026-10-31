"""scripts/colab/env_setup.py — GPU 감지/dtype/배치 크기 분기 로직 테스트.

실제 GPU가 없는 샌드박스에서도 `torch.cuda.*`를 몽키패치해서 T4(bf16 미지원)/
A100류(bf16 지원) 두 경우를 둘 다 재현해 검증한다 — 실제 GPU 위에서 돌려본 적은
없다(미확인, docs/colab_free_tier.md 참고). 이 테스트가 보장하는 건 "그 정보가 주어졌을
때 분기 로직이 올바른가"이다."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import env_setup  # noqa: E402


def test_detect_gpu_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu = env_setup.detect_gpu()
    assert gpu.available is False
    assert gpu.name is None


def test_detect_gpu_t4_like_no_native_bf16(monkeypatch):
    """T4(Turing, compute capability 7.5)는 bf16 네이티브 지원이 없다 —
    `is_bf16_supported(including_emulation=False)`가 False를 내야 한다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))

    class _Props:
        total_memory = 16 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: _Props())

    def fake_bf16_supported(including_emulation=True):
        return including_emulation  # 에뮬레이션 포함이면 True, 네이티브만이면 False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fake_bf16_supported)

    gpu = env_setup.detect_gpu()
    assert gpu.available is True
    assert gpu.name == "Tesla T4"
    assert abs(gpu.vram_gb - 16.0) < 1e-6
    assert gpu.compute_capability == (7, 5)
    assert gpu.bf16_native is False  # including_emulation=False로 호출했는지가 핵심


def test_detect_gpu_a100_like_native_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "A100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (8, 0))

    class _Props:
        total_memory = 40 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: _Props())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)

    gpu = env_setup.detect_gpu()
    assert gpu.bf16_native is True


def test_select_dtype_cpu_is_fp32():
    gpu = env_setup.GpuInfo(available=False)
    choice = env_setup.select_dtype(gpu)
    assert choice.dtype == torch.float32
    assert choice.amp_mode == "none"


def test_select_dtype_t4_uses_fp16_scaled():
    gpu = env_setup.GpuInfo(available=True, name="Tesla T4", vram_gb=16.0, compute_capability=(7, 5), bf16_native=False)
    choice = env_setup.select_dtype(gpu)
    assert choice.amp_mode == "fp16_scaled"
    assert "T4" in choice.reason or "미지원" in choice.reason


def test_select_dtype_a100_uses_bf16():
    gpu = env_setup.GpuInfo(available=True, name="A100", vram_gb=40.0, compute_capability=(8, 0), bf16_native=True)
    choice = env_setup.select_dtype(gpu)
    assert choice.dtype == torch.bfloat16
    assert choice.amp_mode == "bf16"


def test_select_dtype_force_overrides_auto_detection():
    gpu = env_setup.GpuInfo(available=True, name="Tesla T4", vram_gb=16.0, compute_capability=(7, 5), bf16_native=False)
    choice = env_setup.select_dtype(gpu, force="fp32")
    assert choice.amp_mode == "none"
    assert choice.dtype == torch.float32


def test_select_dtype_invalid_force_raises():
    gpu = env_setup.GpuInfo(available=False)
    try:
        env_setup.select_dtype(gpu, force="not_a_dtype")
        assert False, "should have raised"
    except ValueError:
        pass


def test_select_batch_size_manual_overrides_auto():
    gpu = env_setup.GpuInfo(available=True, vram_gb=16.0)
    size, source = env_setup.select_batch_size(gpu, manual=5)
    assert (size, source) == (5, "manual")


def test_select_batch_size_auto_scales_with_vram():
    small = env_setup.select_batch_size(env_setup.GpuInfo(available=True, vram_gb=8.0))
    mid = env_setup.select_batch_size(env_setup.GpuInfo(available=True, vram_gb=16.0))
    large = env_setup.select_batch_size(env_setup.GpuInfo(available=True, vram_gb=80.0))
    assert small[0] <= mid[0] <= large[0]
    assert small[1] == mid[1] == large[1] == "auto"


def test_select_batch_size_cpu_defaults_to_one():
    size, source = env_setup.select_batch_size(env_setup.GpuInfo(available=False))
    assert (size, source) == (1, "auto")


def test_detect_compute_env_end_to_end(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = env_setup.detect_compute_env()
    assert env.gpu.available is False
    assert env.dtype_choice.amp_mode == "none"
    assert env.batch_size == 1
    assert "GPU" in env.summary()
