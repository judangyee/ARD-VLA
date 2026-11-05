"""`scripts/policy_interface.py`(RoCo 평가용 정책 실행 어댑터)의 인터페이스 테스트 —
가짜(합성) 환경/데이터로 `RoCoPolicyAdapter.act()`와 `replay_episode()`가 끝까지 도는지
확인한다. 공식 RoCo 환경과는 통신하지 않는다(docs/policy_interface.md의 "미확인 항목" 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import policy_interface  # noqa: E402


def _build_offline_tokenizer():
    """네트워크 없는 즉석 토크나이저 — tests/test_configurable_ard.py의
    `_build_offline_tokenizer`와 같은 패턴(파일을 가로질러 import하지 않고 각 테스트 파일이
    자급자족하도록 각자 둔다)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "insert": 2, "the": 3, "bolt": 4, "into": 5, "board": 6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


def _build_roco_shaped_policy(monkeypatch, **overrides):
    """RoCo 차원(state 44, action 14, 카메라 3대, ard_arm_dim=7)에 맞춘 작은 합성 정책
    + 전/후처리 파이프라인(실제 동작하는 오프라인 토크나이저 포함)을 만든다. 이 레포의
    `tests/conftest.py::build_tiny_policy` fixture는 카메라 1대/대칭 state 차원을 가정해서
    맞지 않으므로, 이 테스트 파일은 `SmolVLAConfig`/`SmolVLAPolicy`를 직접 만든다."""
    import transformers
    from transformers import SmolVLMConfig, SmolVLMVisionConfig

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    tiny_cfg = SmolVLMConfig(
        vision_config=SmolVLMVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_channels=3, image_size=64, patch_size=16,
        ),
        text_config={
            "model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 256, "max_position_embeddings": 128,
        },
        scale_factor=2, image_token_id=57,
    )

    class _StubTok:
        fake_image_token_id = 1
        global_image_token_id = 2

    class _StubProc:
        tokenizer = _StubTok()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: tiny_cfg))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _StubProc()))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: _build_offline_tokenizer())
    )

    input_features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in policy_interface.DEFAULT_CAMERA_KEYS.values()}
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(policy_interface.STATE_DIM,))
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(policy_interface.ACTION_DIM,))}

    config_kwargs = dict(
        input_features=input_features, output_features=output_features, device="cpu",
        vlm_model_name="fake/tiny-smolvlm", load_vlm_weights=False, num_vlm_layers=2,
        max_action_dim=policy_interface.ACTION_DIM, max_state_dim=policy_interface.STATE_DIM,
        use_ard=True, ard_arm_dim=7, ard_gripper_dim=1, action_space="ee",
        chunk_size=6, n_action_steps=2, num_steps=2,
    )
    config_kwargs.update(overrides)
    config = SmolVLAConfig(**config_kwargs)
    policy = SmolVLAPolicy(config)
    policy.eval()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config)
    return policy, preprocessor, postprocessor


def _fake_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        role: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for role in policy_interface.DEFAULT_CAMERA_KEYS
    }


def test_adapter_act_returns_14dim_continuous_action(monkeypatch):
    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = policy_interface.RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    adapter.reset()

    state = np.random.default_rng(1).normal(size=(policy_interface.STATE_DIM,)).astype(np.float32)
    action = adapter.act(images=_fake_images(), state=state, task="insert the bolt into the board")

    assert action.shape == (policy_interface.ACTION_DIM,)
    assert torch.isfinite(action).all()
    assert action.dtype == torch.float32


def test_adapter_reset_clears_action_queue_between_episodes(monkeypatch):
    """reset()을 부르면 이전 에피소드에서 쌓인 액션 큐가 비워져서, 새 에피소드 첫 act()가
    새로 forward를 한다(큐에서 그냥 pop하는 게 아니라) — n_action_steps=2라 3번째 호출부터는
    원래 큐에서 꺼내 쓰므로, reset 유무로 호출 패턴을 구분한다."""
    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = policy_interface.RoCoPolicyAdapter(policy, preprocessor, postprocessor)

    forward_calls = {"n": 0}
    original = policy.model.forward

    def counting_forward(*args, **kwargs):
        forward_calls["n"] += 1
        return original(*args, **kwargs)

    # select_action -> _get_action_chunk -> ... -> model.sample_actions (추론 경로, forward 아님)
    # 를 세야 하므로 sample_actions를 센다.
    original_sample = policy.model.sample_actions
    sample_calls = {"n": 0}

    def counting_sample(*args, **kwargs):
        sample_calls["n"] += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(policy.model, "sample_actions", counting_sample)

    state = np.zeros(policy_interface.STATE_DIM, dtype=np.float32)
    images = _fake_images()

    adapter.reset()
    adapter.act(images=images, state=state, task="pick up the part")  # 큐 비어있음 -> forward 1회, n_action_steps=2개 채움
    assert sample_calls["n"] == 1
    adapter.act(images=images, state=state, task="pick up the part")  # 큐에서 pop만 함
    assert sample_calls["n"] == 1

    adapter.reset()  # 큐가 다시 비워진다
    adapter.act(images=images, state=state, task="pick up the part")
    assert sample_calls["n"] == 2  # reset 덕에 새로 forward했다


def test_act_rejects_wrong_state_shape(monkeypatch):
    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = policy_interface.RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    adapter.reset()

    try:
        adapter.act(images=_fake_images(), state=np.zeros(10, dtype=np.float32), task="x")
        raise AssertionError("잘못된 state shape인데 에러가 안 났다")
    except ValueError:
        pass


def test_replay_episode_computes_per_dim_l1_against_ground_truth(monkeypatch):
    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = policy_interface.RoCoPolicyAdapter(policy, preprocessor, postprocessor)

    from lerobot.utils.constants import OBS_STATE

    rng = np.random.default_rng(2)
    n_frames = 5
    frames = []
    for _ in range(n_frames):
        frame = {role_key: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for role_key in policy_interface.DEFAULT_CAMERA_KEYS.values()}
        frame[OBS_STATE] = rng.normal(size=(policy_interface.STATE_DIM,)).astype(np.float32)
        frame["action"] = rng.normal(size=(policy_interface.ACTION_DIM,)).astype(np.float32)
        frame["task"] = "insert the bolt into the board"
        frames.append(frame)

    result = policy_interface.replay_episode(adapter, frames)

    assert result["per_dim_l1"].shape == (policy_interface.ACTION_DIM,)
    assert result["n_frames"] == n_frames
    assert np.isfinite(result["per_dim_l1"]).all()
    assert result["mean_l1"] >= 0
    assert result["predicted_actions"].shape == (n_frames, policy_interface.ACTION_DIM)
    assert result["true_actions"].shape == (n_frames, policy_interface.ACTION_DIM)


def test_replay_episode_rejects_empty_frame_list(monkeypatch):
    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = policy_interface.RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    try:
        policy_interface.replay_episode(adapter, [])
        raise AssertionError("빈 frames인데 에러가 안 났다")
    except ValueError:
        pass
