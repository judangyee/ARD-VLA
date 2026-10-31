"""scripts/colab/schema_check.py — observation.state/action 채널 순서 검증.

핵심 요구사항: "meta/info.json의 이름이 기대 순서와 다르면 즉시 중단"을
`SchemaMismatchError`로 구현했는지 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import schema_check  # noqa: E402

import pytest

ACTION_NAMES_OK = [
    "left_ee_x", "left_ee_y", "left_ee_z", "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
    "right_ee_x", "right_ee_y", "right_ee_z", "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
]


def _state_names_ok() -> list[str]:
    return (
        [f"left_ee_{c}" for c in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
        + [f"right_ee_{c}" for c in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
        + [f"left_joint_pos_{i}" for i in range(7)]
        + [f"right_joint_pos_{i}" for i in range(7)]
        + [f"left_joint_vel_{i}" for i in range(7)]
        + [f"right_joint_vel_{i}" for i in range(7)]
        + ["left_gripper", "right_gripper"]
    )


def test_validate_action_names_accepts_expected_order():
    schema_check.validate_action_names(ACTION_NAMES_OK)  # 예외 없이 통과해야 함


def test_validate_action_names_rejects_wrong_count():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names(ACTION_NAMES_OK[:10])


def test_validate_action_names_rejects_missing_names():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names(None)
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names([])


def test_validate_action_names_rejects_swapped_arms():
    swapped = ACTION_NAMES_OK[7:] + ACTION_NAMES_OK[:7]
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names(swapped)


def test_validate_action_names_rejects_wrong_channel_type():
    bad = list(ACTION_NAMES_OK)
    bad[6] = "left_wrist_angle"  # 그리퍼 위치인데 그리퍼/EE처럼 안 보임
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names(bad)


def test_validate_state_names_accepts_expected_order():
    schema_check.validate_state_names(_state_names_ok())  # 통과해야 함


def test_validate_state_names_rejects_wrong_count():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_state_names(_state_names_ok()[:20])


def test_validate_state_names_rejects_swapped_gripper_order():
    names = _state_names_ok()
    names[42], names[43] = names[43], names[42]
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_state_names(names)


def test_validate_state_names_rejects_swapped_ee_arm_order():
    """EE 블록(0-13) 안에서 왼팔/오른팔 7개씩이 통째로 바뀌면 잡아야 한다."""
    names = _state_names_ok()
    names[0:14] = names[7:14] + names[0:7]
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_state_names(names)


def test_validate_schema_end_to_end_info_dict():
    info = {"features": {"observation.state": {"names": _state_names_ok()}, "action": {"names": ACTION_NAMES_OK}}}
    schema_check.validate_schema(info)  # 통과해야 함


def test_validate_schema_end_to_end_aborts_on_mismatch():
    bad_state = _state_names_ok()
    bad_state[42], bad_state[43] = bad_state[43], bad_state[42]
    info = {"features": {"observation.state": {"names": bad_state}, "action": {"names": ACTION_NAMES_OK}}}
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_schema(info)


def test_validate_schema_aborts_when_features_missing_entirely():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_schema({"features": {}})
