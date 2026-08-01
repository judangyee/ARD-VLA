#!/usr/bin/env python3
"""Smoke-test the SmolVLA research environment.

Verifies that torch, transformers, lerobot, and the SmolVLA policy classes
import cleanly and that device selection falls back to CPU gracefully when
no GPU/accelerator is present. Does not download any pretrained weights, so
it works fully offline.
"""

import sys


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch
    except ImportError as exc:
        fail(f"could not import torch: {exc}")
    print(f"torch: {torch.__version__}")

    try:
        import transformers
    except ImportError as exc:
        fail(f"could not import transformers: {exc}")
    print(f"transformers: {transformers.__version__}")

    try:
        import lerobot
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.utils import auto_select_torch_device
    except ImportError as exc:
        fail(f"could not import lerobot / SmolVLA modules: {exc}")
    print(f"lerobot: {lerobot.__version__}")
    print(f"SmolVLAPolicy: {SmolVLAPolicy.__module__}.{SmolVLAPolicy.__qualname__}")

    cuda_available = torch.cuda.is_available()
    device = auto_select_torch_device()
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"auto_select_torch_device(): {device}")

    if not cuda_available and device.type != "cpu":
        fail(f"no GPU available but auto-selected device was '{device}', expected 'cpu'")

    try:
        config = SmolVLAConfig()
    except Exception as exc:  # noqa: BLE001
        fail(f"failed to construct SmolVLAConfig on a GPU-less machine: {exc}")
    print(f"SmolVLAConfig() constructed OK (device={config.device})")

    print("\n[OK] Environment imports cleanly and falls back to CPU without a GPU.")


if __name__ == "__main__":
    main()
