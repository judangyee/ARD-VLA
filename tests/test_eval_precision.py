"""`scripts/eval_precision.py`의 접촉 구간 자동 분리/mm 환산/팔별 분리 로직을 합성
데이터로 검증한다. 실제 LeRobotDataset 로딩(`load_episode_frames`)은 Hub 접근이 필요해
이 샌드박스에서 검증 불가 — 순수 수학 함수들과, tiny 합성 정책으로 끝까지 돈
`compute_precision_report()`만 검증한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_precision  # noqa: E402


def test_detect_gripper_change_frame_picks_largest_delta():
    gripper = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.9])
    assert eval_precision.detect_gripper_change_frame(gripper) == 3


def test_detect_min_height_frame_picks_lowest_z():
    z = np.array([0.5, 0.3, 0.05, 0.4])
    assert eval_precision.detect_min_height_frame(z) == 2


def test_select_contact_arm_picks_larger_gripper_variation():
    true_actions = np.zeros((6, 14), dtype=np.float32)
    true_actions[:, 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # left gripper: 변화 큼
    true_actions[:, 13] = [0.2, 0.2, 0.2, 0.21, 0.21, 0.21]  # right gripper: 변화 작음
    assert eval_precision.select_contact_arm(true_actions) == "left"

    true_actions2 = np.zeros((6, 14), dtype=np.float32)
    true_actions2[:, 13] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    true_actions2[:, 6] = [0.2, 0.2, 0.2, 0.21, 0.21, 0.21]
    assert eval_precision.select_contact_arm(true_actions2) == "right"


def test_select_contact_window_centers_on_detected_event_and_stays_in_bounds():
    true_actions = np.zeros((20, 14), dtype=np.float32)
    true_actions[10:, 6] = 1.0  # left gripper가 프레임 10에서 바뀜

    start, end = eval_precision.select_contact_window(true_actions, "left", "gripper_change", window_frames=4)
    assert start < 11 <= end  # 이벤트 프레임(10번 변화 -> 감지는 11번)이 구간 안에 들어감
    assert 0 <= start < end <= 20

    # 경계 근처(이벤트가 끝자락)에서도 범위를 벗어나지 않는지
    true_actions_edge = np.zeros((20, 14), dtype=np.float32)
    true_actions_edge[19, 6] = 1.0
    start_e, end_e = eval_precision.select_contact_window(true_actions_edge, "left", "gripper_change", window_frames=10)
    assert 0 <= start_e < end_e <= 20


def test_arm_errors_computes_mm_rad_and_gripper_l1():
    pred = np.zeros((3, 14), dtype=np.float32)
    true = np.zeros((3, 14), dtype=np.float32)
    true[:, 0] = 0.01  # left x: 10mm 오차
    true[:, 3] = 0.1   # left rx: 0.1rad 오차
    true[:, 6] = 0.5   # left gripper: 0.5 오차

    errs = eval_precision.arm_errors(pred, true, "left")
    assert abs(errs["pos_l1_mm"] - (0.01 * 1000 / 3)) < 1e-4
    assert abs(errs["rot_l1_rad"] - (0.1 / 3)) < 1e-6
    assert abs(errs["gripper_l1"] - 0.5) < 1e-6  # 그리퍼는 채널 1개라 행(3프레임) 평균만
    assert errs["n_frames"] == 3

    right_errs = eval_precision.arm_errors(pred, true, "right")
    assert right_errs["pos_l1_mm"] == 0.0
    assert right_errs["rot_l1_rad"] == 0.0
    assert right_errs["gripper_l1"] == 0.0


def _build_offline_tokenizer():
    """tests/test_policy_interface.py와 동일한 패턴."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "insert": 2, "the": 3, "bolt": 4, "into": 5, "board": 6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


def _build_roco_shaped_policy(monkeypatch):
    """tests/test_policy_interface.py::_build_roco_shaped_policy와 동일(state 44/action 14/
    카메라 3대/ard_arm_dim=7) — 각 테스트 파일이 자급자족하도록 복제."""
    import transformers
    from transformers import SmolVLMConfig, SmolVLMVisionConfig

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE
    from policy_interface import ACTION_DIM, DEFAULT_CAMERA_KEYS, STATE_DIM

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

    input_features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in DEFAULT_CAMERA_KEYS.values()}
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}

    config = SmolVLAConfig(
        input_features=input_features, output_features=output_features, device="cpu",
        vlm_model_name="fake/tiny-smolvlm", load_vlm_weights=False, num_vlm_layers=2,
        max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM,
        use_ard=True, ard_arm_dim=7, ard_gripper_dim=1, action_space="ee",
        chunk_size=6, n_action_steps=2, num_steps=2,
    )
    policy = SmolVLAPolicy(config)
    policy.eval()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config)
    return policy, preprocessor, postprocessor


def _fake_frames(n_frames: int) -> list[dict]:
    from lerobot.utils.constants import OBS_STATE
    from policy_interface import ACTION_DIM, DEFAULT_CAMERA_KEYS, STATE_DIM

    rng = np.random.default_rng(3)
    frames = []
    for _ in range(n_frames):
        frame = {role_key: rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for role_key in DEFAULT_CAMERA_KEYS.values()}
        frame[OBS_STATE] = rng.normal(size=(STATE_DIM,)).astype(np.float32)
        frame["action"] = rng.normal(size=(ACTION_DIM,)).astype(np.float32)
        frame["task"] = "insert the bolt into the board"
        frames.append(frame)
    return frames


def test_compute_precision_report_end_to_end_on_tiny_policy(monkeypatch):
    from policy_interface import RoCoPolicyAdapter

    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    frames = _fake_frames(10)

    report = eval_precision.compute_precision_report(adapter, frames, window_frames=4)

    assert report["n_frames_total"] == 10
    assert report["contact_arm"] in ("left", "right")
    assert 0 <= report["window"][0] < report["window"][1] <= 10
    for group in ("full_episode", "contact_window"):
        for arm in ("left", "right"):
            errs = report[group][arm]
            assert errs["pos_l1_mm"] >= 0
            assert errs["rot_l1_rad"] >= 0
            assert errs["gripper_l1"] >= 0
    assert report["full_episode"]["left"]["n_frames"] == 10
    assert report["contact_window"]["left"]["n_frames"] == report["window"][1] - report["window"][0]

    import json
    json.loads(json.dumps(report))


def test_window_criterion_min_height_uses_z_channel(monkeypatch):
    from policy_interface import RoCoPolicyAdapter

    policy, preprocessor, postprocessor = _build_roco_shaped_policy(monkeypatch)
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    frames = _fake_frames(10)

    report = eval_precision.compute_precision_report(
        adapter, frames, window_criterion="min_height", contact_arm="left", window_frames=4,
    )
    assert report["window_criterion"] == "min_height"
    assert report["contact_arm"] == "left"
