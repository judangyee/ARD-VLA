"""`meta/info.json`의 `observation.state`/`action` 채널 이름 순서를 기대 순서와 대조한다.

**기대 순서(이번 요청에서 명시됨, `state_select` 적용 후 30채널 기준)**:
`[왼팔 EE 7][오른팔 EE 7][왼팔 관절 위치 7][오른팔 관절 위치 7][왼쪽 그리퍼][오른쪽 그리퍼]`
(관절 속도 14개는 `state_select`로 아예 제외 — `configs/presets/roco_board.yaml`의
`state_select = list(range(28)) + [42, 43]`와 동일).

**정확한 문자열이 아니라 구조(왼팔/오른팔, EE/관절/그리퍼)로 검증하는 이유**: action
14채널의 정확한 이름(`left_ee_x` 등)은 `docs/configurable_ard.md`가 사용자의 실측
집계를 근거로 확인한 값이지만, 그 확인은 "이 순서로 통계를 모았다"는 수준이고
`meta/info.json`에 **그 정확한 문자열이 그대로 박혀 있는지**는 이 샌드박스에서
Hugging Face Hub 접근이 막혀 있어 직접 열어본 적이 없다(미확인). 완전히 다른 로봇/
데이터셋을 잘못 가리키는 경우(예: action이 관절각 기반이거나 팔 순서가 반대인 경우)는
반드시 잡아야 하지만, 글자 하나 다른 철자 변형(`left_ee_x` vs `l_ee_pos_x`)까지
전부 실패로 처리하면 오히려 멀쩍이 맞는 데이터를 거부할 위험이 있다 — 그래서 "왼팔/
오른팔"과 "EE/관절/그리퍼" 두 축의 부분 문자열만 검사한다. 이 판정 로직 자체가
실제 RoCo info.json으로 검증된 적은 없다(미검증) — 처음 실행할 때 로그에 실제
이름을 그대로 찍어서 사람이 한 번 더 눈으로 확인할 수 있게 한다.
"""

from __future__ import annotations

import re

# state_select: raw 44차원 중 관절 속도(28-41)를 뺀 30채널 — configs/presets/roco_board.yaml과 동일.
STATE_SELECT = list(range(28)) + [42, 43]

EXPECTED_ACTION_NAMES = [
    "left_ee_x", "left_ee_y", "left_ee_z", "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
    "right_ee_x", "right_ee_y", "right_ee_z", "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
]

_LEFT_RE = re.compile(r"\bleft\b|^l_|_l_|left", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\bright\b|^r_|_r_|right", re.IGNORECASE)
_EE_RE = re.compile(r"ee|pos|quat|xyz|rot", re.IGNORECASE)
_JOINT_RE = re.compile(r"joint", re.IGNORECASE)
_GRIPPER_RE = re.compile(r"gripper|grip", re.IGNORECASE)


class SchemaMismatchError(Exception):
    """observation.state/action 채널 이름이 기대 순서와 다를 때 던진다 — 호출자는 이걸
    받으면 학습을 시작하지 말고 즉시 중단해야 한다."""


def _side(name: str) -> str | None:
    is_left = bool(_LEFT_RE.search(name))
    is_right = bool(_RIGHT_RE.search(name))
    if is_left and not is_right:
        return "left"
    if is_right and not is_left:
        return "right"
    return None


def validate_action_names(names: list[str] | None) -> None:
    if not names:
        raise SchemaMismatchError("action 피처에 채널 이름(names)이 없습니다 — 기대 순서를 확인할 수 없어 중단합니다.")
    if len(names) != len(EXPECTED_ACTION_NAMES):
        raise SchemaMismatchError(
            f"action 채널 개수가 기대값(14)과 다릅니다: 실제 {len(names)}개 {names!r}"
        )
    for i, name in enumerate(names):
        expected_side = "left" if i < 7 else "right"
        actual_side = _side(name)
        if actual_side != expected_side:
            raise SchemaMismatchError(
                f"action[{i}]={name!r}의 팔 쪽(왼/오)을 '{expected_side}'로 기대했는데 "
                f"판정 결과는 {actual_side!r}입니다 — 기대 순서: {EXPECTED_ACTION_NAMES}, 실제: {names}"
            )
    # EE/그리퍼 타입 체크: 팔당 앞 6개는 EE(xyz+rot), 마지막 1개는 그리퍼.
    for arm_start in (0, 7):
        for i in range(arm_start, arm_start + 6):
            if not _EE_RE.search(names[i]):
                raise SchemaMismatchError(f"action[{i}]={names[i]!r}이 EE 위치/회전 채널처럼 보이지 않습니다.")
        gripper_idx = arm_start + 6
        if not _GRIPPER_RE.search(names[gripper_idx]):
            raise SchemaMismatchError(f"action[{gripper_idx}]={names[gripper_idx]!r}이 그리퍼 채널처럼 보이지 않습니다.")


def validate_state_names(names: list[str] | None, state_select: list[int] = STATE_SELECT) -> None:
    if not names:
        raise SchemaMismatchError("observation.state 피처에 채널 이름(names)이 없습니다 — 중단합니다.")
    if len(names) != 44:
        raise SchemaMismatchError(f"observation.state 채널 개수가 기대값(44)과 다릅니다: 실제 {len(names)}개")

    try:
        selected = [names[i] for i in state_select]
    except IndexError as e:
        raise SchemaMismatchError(f"state_select 인덱스가 실제 채널 개수를 벗어납니다: {e}") from e

    # 선택된 30채널 = [왼팔 EE 7][오른팔 EE 7][왼팔 관절위치 7][오른팔 관절위치 7][왼쪽 그리퍼][오른쪽 그리퍼]
    blocks = [
        (0, 7, "left", _EE_RE, "EE"),
        (7, 14, "right", _EE_RE, "EE"),
        (14, 21, "left", _JOINT_RE, "관절 위치"),
        (21, 28, "right", _JOINT_RE, "관절 위치"),
        (28, 29, "left", _GRIPPER_RE, "그리퍼"),
        (29, 30, "right", _GRIPPER_RE, "그리퍼"),
    ]
    for start, end, expected_side, type_re, label in blocks:
        for i in range(start, end):
            name = selected[i]
            actual_side = _side(name)
            if actual_side != expected_side:
                raise SchemaMismatchError(
                    f"state_select 결과 {i}번째 채널({name!r}, raw 인덱스 {state_select[i]})의 팔 쪽을 "
                    f"'{expected_side}'로 기대했는데 판정 결과는 {actual_side!r}입니다 — "
                    f"기대 순서: [왼팔 EE 7][오른팔 EE 7][왼팔 관절위치 7][오른팔 관절위치 7]"
                    f"[왼쪽 그리퍼][오른쪽 그리퍼], 실제 선택된 이름: {selected}"
                )
            if not type_re.search(name):
                raise SchemaMismatchError(
                    f"state_select 결과 {i}번째 채널({name!r})이 '{label}' 채널처럼 보이지 않습니다."
                )


def validate_schema(info: dict) -> None:
    """`meta/info.json`을 그대로 로드한 dict를 받아 `features.observation.state`/
    `features.action`의 `names`를 검증한다. 하나라도 어긋나면 `SchemaMismatchError`를
    던진다 — 호출자(실행 스크립트)는 이 예외를 잡아 즉시 종료해야 한다."""
    features = info.get("features", {})
    action_names = (features.get("action") or {}).get("names")
    state_names = (features.get("observation.state") or {}).get("names")
    validate_action_names(action_names)
    validate_state_names(state_names)
