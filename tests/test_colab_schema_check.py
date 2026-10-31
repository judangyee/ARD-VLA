"""scripts/colab/schema_check.py — observation.state/action 채널 이름의 정확한(exact)
비교.

`schema_check.EXPECTED_STATE_NAMES`/`EXPECTED_ACTION_NAMES` 자체를 "실제 info.json"
역할의 테스트 픽스처로 그대로 쓴다(이번 요청에서 그 정확한 이름 규칙이 주어졌으므로,
더는 별도로 손으로 비슷한 이름을 만들 필요가 없다). 핵심 요구사항: "다르면 첫 불일치
위치와 기대값/실제값을 출력하고 즉시 중단"을 `SchemaMismatchError`로 구현했는지
확인한다."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import schema_check  # noqa: E402

import pytest


def _fake_info(state_names=None, action_names=None) -> dict:
    return {
        "features": {
            "observation.state": {"names": state_names if state_names is not None else list(schema_check.EXPECTED_STATE_NAMES)},
            "action": {"names": action_names if action_names is not None else list(schema_check.EXPECTED_ACTION_NAMES)},
        }
    }


def test_expected_name_lists_have_correct_lengths():
    assert len(schema_check.EXPECTED_STATE_NAMES) == 44
    assert len(schema_check.EXPECTED_ACTION_NAMES) == 14


def test_validate_action_names_accepts_exact_expected_order():
    schema_check.validate_action_names(list(schema_check.EXPECTED_ACTION_NAMES))  # 예외 없이 통과해야 함


def test_validate_state_names_accepts_exact_expected_order():
    schema_check.validate_state_names(list(schema_check.EXPECTED_STATE_NAMES))  # 통과해야 함


def test_validate_action_names_rejects_missing_names():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names(None)
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_action_names([])


def test_validate_action_names_rejects_wrong_count():
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_action_names(schema_check.EXPECTED_ACTION_NAMES[:10])
    assert "개수" in str(exc_info.value)


def test_validate_action_names_reports_first_mismatch_with_expected_and_actual():
    names = list(schema_check.EXPECTED_ACTION_NAMES)
    names[7] = "right_ee_WRONG"  # 8번째(인덱스 7) 채널만 틀림
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_action_names(names)
    msg = str(exc_info.value)
    assert "[7]" in msg
    assert "right_ee_x" in msg  # 기대값
    assert "right_ee_WRONG" in msg  # 실제값


def test_validate_action_names_rejects_swapped_arms():
    swapped = schema_check.EXPECTED_ACTION_NAMES[7:] + schema_check.EXPECTED_ACTION_NAMES[:7]
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_action_names(swapped)
    # 왼팔/오른팔 블록을 통째로 바꾸면 인덱스 0에서부터 어긋나야 한다.
    assert "[0]" in str(exc_info.value)


def test_validate_state_names_rejects_wrong_count():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_state_names(schema_check.EXPECTED_STATE_NAMES[:20])


def test_validate_state_names_rejects_swapped_gripper_order():
    names = list(schema_check.EXPECTED_STATE_NAMES)
    names[42], names[43] = names[43], names[42]
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_state_names(names)
    msg = str(exc_info.value)
    assert "[42]" in msg
    assert "left_gripper" in msg
    assert "right_gripper" in msg


def test_validate_state_names_rejects_swapped_ee_arm_order():
    """EE 블록(0-13) 안에서 왼팔/오른팔 7개씩이 통째로 바뀌면 첫 어긋남(인덱스 0)에서
    잡아야 한다."""
    names = list(schema_check.EXPECTED_STATE_NAMES)
    names[0:14] = names[7:14] + names[0:7]
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_state_names(names)
    assert "[0]" in str(exc_info.value)


def test_validate_state_names_rejects_quaternion_component_order_change():
    """qw/qx/qy/qz 순서가 바뀌는 것도(철자 하나만 다른 미세한 차이) 정확히 잡아야 한다."""
    names = list(schema_check.EXPECTED_STATE_NAMES)
    names[3], names[4] = names[4], names[3]  # left_ee_qw <-> left_ee_qx
    with pytest.raises(schema_check.SchemaMismatchError) as exc_info:
        schema_check.validate_state_names(names)
    assert "[3]" in str(exc_info.value)


def test_validate_schema_end_to_end_accepts_expected_fixture():
    schema_check.validate_schema(_fake_info())  # 통과해야 함


def test_validate_schema_end_to_end_aborts_on_reordered_state():
    info = _fake_info()
    names = info["features"]["observation.state"]["names"]
    names[42], names[43] = names[43], names[42]
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_schema(info)


def test_validate_schema_end_to_end_aborts_on_reordered_action():
    info = _fake_info()
    names = info["features"]["action"]["names"]
    names[0], names[7] = names[7], names[0]
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_schema(info)


def test_validate_schema_aborts_when_features_missing_entirely():
    with pytest.raises(schema_check.SchemaMismatchError):
        schema_check.validate_schema({"features": {}})


def test_fixture_helper_does_not_mutate_module_constants():
    """`_fake_info()`가 돌려주는 리스트를 테스트가 제자리에서 바꿔도, 다음 테스트가 쓰는
    `schema_check.EXPECTED_*`가 오염되지 않아야 한다(list(...)로 복사했는지 확인)."""
    info = _fake_info()
    info["features"]["observation.state"]["names"][0] = "오염됨"
    assert schema_check.EXPECTED_STATE_NAMES[0] == "left_ee_x"
    assert copy.deepcopy(schema_check.EXPECTED_STATE_NAMES)[0] == "left_ee_x"
