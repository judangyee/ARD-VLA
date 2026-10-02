"""pytest 공용 fixture.

Hugging Face Hub 접근 없이 SmolVLAPolicy 전체(forward/backward 포함한 실제 코드 경로)를
오프라인으로 검증하기 위한 핵심 장치: `AutoConfig.from_pretrained`/
`AutoProcessor.from_pretrained`를 아주 작은 합성 SmolVLM 설정으로 몽키패치한다
(`load_vlm_weights=False`라서 `AutoModelForImageTextToText.from_pretrained`는 애초에
호출되지 않는다). pytest의 `monkeypatch` fixture를 쓰므로 각 테스트가 끝나면 자동으로
원복된다 — 테스트 간 전역 상태 오염이 없다.
"""

from __future__ import annotations

import torch
import transformers
from transformers import SmolVLMConfig, SmolVLMVisionConfig

import pytest

FAKE_MODEL_ID = "fake/tiny-smolvlm"


def _build_tiny_smolvlm_config() -> SmolVLMConfig:
    """실제 Hub 다운로드 없이, CPU에서도 forward+backward가 몇 초 안에 끝나는 아주 작은
    SmolVLM config를 만든다 (vision/text 백본 둘 다 hidden_size=32, 레이어 2개)."""
    vision_config = SmolVLMVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=64,
        patch_size=16,
    )
    return SmolVLMConfig(
        vision_config=vision_config,
        text_config={
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 256,
            "max_position_embeddings": 128,
        },
        scale_factor=2,
        image_token_id=57,
    )


class _StubTokenizer:
    fake_image_token_id = 1
    global_image_token_id = 2


class _StubProcessor:
    tokenizer = _StubTokenizer()


@pytest.fixture
def tiny_smolvlm_config() -> SmolVLMConfig:
    return _build_tiny_smolvlm_config()


@pytest.fixture
def tiny_smolvlm_monkeypatch(monkeypatch, tiny_smolvlm_config):
    """AutoConfig.from_pretrained / AutoProcessor.from_pretrained를 몽키패치한다. 이 fixture를
    요청하는 테스트는 그 안에서 `SmolVLAPolicy(config)`를 생성할 수 있다(Hub 접근 없이)."""
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: tiny_smolvlm_config)
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _StubProcessor())
    )
    return tiny_smolvlm_config


@pytest.fixture
def build_tiny_policy(tiny_smolvlm_monkeypatch):
    """SmolVLAConfig 오버라이드를 받아 작은 SmolVLAPolicy를 만드는 팩토리 함수를 반환한다.

    사용 예:
        def test_something(build_tiny_policy):
            policy = build_tiny_policy(use_ard=True, ard_use_force_head=True)
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    def _build(arm_dim: int = 7, num_vlm_layers: int = 2, chunk_size: int = 8, **config_overrides):
        input_features = {
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2 * arm_dim,)),
        }
        output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2 * arm_dim,))}
        config_kwargs = dict(
            input_features=input_features,
            output_features=output_features,
            device="cpu",
            vlm_model_name=FAKE_MODEL_ID,
            load_vlm_weights=False,
            num_vlm_layers=num_vlm_layers,
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            max_state_dim=2 * arm_dim,
            max_action_dim=2 * arm_dim,
        )
        config_kwargs.update(config_overrides)
        config = SmolVLAConfig(**config_kwargs)
        return SmolVLAPolicy(config)

    return _build


@pytest.fixture
def build_tiny_batch():
    """합성 배치를 만드는 팩토리 함수를 반환한다."""
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    def _build(batch_size: int = 2, arm_dim: int = 7, chunk: int = 8, lang_len: int = 6) -> dict:
        return {
            OBS_IMAGE: torch.rand(batch_size, 3, 64, 64),
            OBS_STATE: torch.randn(batch_size, 2 * arm_dim),
            ACTION: torch.randn(batch_size, chunk, 2 * arm_dim),
            OBS_LANGUAGE_TOKENS: torch.randint(0, 200, (batch_size, lang_len)),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, lang_len, dtype=torch.bool),
        }

    return _build
