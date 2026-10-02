"""SmolVLA 연구 환경이 깨끗하게 import되고 GPU 없이도 CPU로 잘 떨어지는지 확인한다.

scripts/check_env.py의 pytest 버전 — scripts/check_env.py는 pip install 직후 사람이 직접
`python scripts/check_env.py`로 돌려보는 독립 스모크 스크립트로 그대로 남겨뒀고(README
"Verifying the environment" 절 참고), 이 파일은 같은 점검을 pytest/CI에도 포함시키기 위한
것이다. 사전학습 가중치를 전혀 받지 않아서 완전히 오프라인으로 동작한다.
"""


def test_imports_cleanly():
    import lerobot
    import torch
    import transformers
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    assert torch.__version__
    assert transformers.__version__
    assert lerobot.__version__
    assert SmolVLAPolicy.__module__ == "lerobot.policies.smolvla.modeling_smolvla"


def test_device_falls_back_to_cpu_without_gpu():
    import torch
    from lerobot.utils.utils import auto_select_torch_device

    device = auto_select_torch_device()
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_smolvla_config_constructs_on_cpu_only_machine():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig()
    assert config.device in ("cpu", "cuda", "mps")
