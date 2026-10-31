"""`--smoke` 모드/통합 테스트에서 쓰는 아주 작은 합성 SmolVLM — Hugging Face Hub 접근
없이 `SmolVLAPolicy` 전체(forward/backward/predict_action_chunk)를 오프라인으로 돌리기
위한 몽키패치. `tests/conftest.py`의 `tiny_smolvlm_monkeypatch`/`build_tiny_policy`와
완전히 같은 패턴이다 — pytest의 `monkeypatch` fixture 대신 `unittest.mock.patch`로
구현해서 pytest 밖(실제 `--smoke` CLI 실행)에서도 쓸 수 있게 한 것뿐이다. 두 구현을
갈라두는 이유: 테스트 fixture를 프로덕션 스크립트가 import하는 건 방향이 거꾸로이고
(테스트가 스크립트에 의존해야 함), 로직 자체는 20줄 안팎이라 중복 비용이 낮다."""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import torch

FAKE_MODEL_ID = "fake/tiny-smolvlm"


def _build_tiny_smolvlm_config():
    from transformers import SmolVLMConfig, SmolVLMVisionConfig

    vision_config = SmolVLMVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_channels=3, image_size=64, patch_size=16,
    )
    return SmolVLMConfig(
        vision_config=vision_config,
        text_config={
            "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "vocab_size": 256, "max_position_embeddings": 128,
        },
        scale_factor=2, image_token_id=57,
    )


class _StubTokenizer:
    fake_image_token_id = 1
    global_image_token_id = 2


class _StubProcessor:
    tokenizer = _StubTokenizer()


class _CallableStubTokenizer:
    """`TokenizerProcessorStep._tokenize_text()`가 실제로 **호출**하는
    `AutoTokenizer.from_pretrained(...)`의 대체물 — `tests/test_configurable_ard.py`가
    쓰는 `object()` 스텁은 파이프라인을 "구성"만 해볼 때는 충분했지만, 실제로
    `preprocessor(batch)`를 호출해 텍스트를 토크나이즈하려고 하면
    `TypeError: 'object' object is not callable`가 난다(이 스크립트를 실제로 돌려서
    확인함) — 그래서 호출 가능한 최소 스텁을 따로 둔다. 길이/내용은 의미 없고
    고정 shape의 더미 텐서만 반환한다."""

    pad_token_id = 0

    def __call__(self, text, max_length=None, truncation=True, padding="max_length", padding_side="right", return_tensors="pt"):
        if isinstance(text, str):
            text = [text]
        length = max_length or 16
        n = len(text)
        return {
            "input_ids": torch.zeros(n, length, dtype=torch.long),
            "attention_mask": torch.ones(n, length, dtype=torch.long),
        }


@contextmanager
def tiny_smolvlm_patch():
    """`AutoConfig`/`AutoProcessor`뿐 아니라 `AutoTokenizer`도 몽키패치한다 —
    `make_smolvla_pre_post_processors()`가 만드는 `TokenizerProcessorStep`이
    생성 시점에 `AutoTokenizer.from_pretrained(config.vlm_model_name)`을 호출하기
    때문이다(`tests/test_configurable_ard.py`가 이미 같은 패턴으로 몽키패치해서
    확인한 지점 — `object()` 스텁이면 충분하다, 토크나이저를 실제로 호출하는
    경로까지는 이 스모크가 건드리지 않는다)."""
    import transformers

    tiny_config = _build_tiny_smolvlm_config()
    with (
        mock.patch.object(transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: tiny_config)),
        mock.patch.object(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _StubProcessor())),
        mock.patch.object(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: _CallableStubTokenizer())),
    ):
        yield tiny_config
