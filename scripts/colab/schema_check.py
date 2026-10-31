"""`meta/info.json`의 `observation.state`/`action` 채널 이름을 **정확한 문자열**로
비교한다.

**이전 버전(되돌림)과 바뀐 점**: 이전엔 실제 info.json의 정확한 문자열을 몰라서
"left/right" + "ee/pos/quat"·"joint"·"gripper" 부분 문자열만 보는 휴리스틱이었다.
이제 실제 이름 규칙이 주어져서(이번 요청) 더는 추측할 필요가 없다 — 44개 state/14개
action 이름을 정확한 리스트로 하드코딩하고, 완전 일치(exact match)만 통과시킨다.
다르면 **첫 번째로 어긋나는 위치**와 그 자리의 기대값/실제값을 담아
`SchemaMismatchError`를 던진다(호출자는 즉시 중단해야 한다).

state 이름 규칙(라운드 2026-10-31 확정):
    [f"{s}_ee_{a}" for s in ("left","right") for a in ("x","y","z","qw","qx","qy","qz")]
    + [f"{s}_jpos_{i}" for s in ("left","right") for i in range(7)]
    + [f"{s}_jvel_{i}" for s in ("left","right") for i in range(7)]
    + ["left_gripper", "right_gripper"]
action 이름 규칙은 `docs/configurable_ard.md`가 이미 확인한 14개 그대로다.
"""

from __future__ import annotations

# state_select: raw 44차원 중 관절 속도(28-41)를 뺀 30채널 — configs/presets/roco_board.yaml과 동일.
STATE_SELECT = list(range(28)) + [42, 43]

EXPECTED_STATE_NAMES = (
    [f"{side}_ee_{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"{side}_jpos_{i}" for side in ("left", "right") for i in range(7)]
    + [f"{side}_jvel_{i}" for side in ("left", "right") for i in range(7)]
    + ["left_gripper", "right_gripper"]
)

EXPECTED_ACTION_NAMES = [
    "left_ee_x", "left_ee_y", "left_ee_z", "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
    "right_ee_x", "right_ee_y", "right_ee_z", "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
]

assert len(EXPECTED_STATE_NAMES) == 44
assert len(EXPECTED_ACTION_NAMES) == 14


class SchemaMismatchError(Exception):
    """observation.state/action 채널 이름이 기대 순서와 다를 때 던진다 — 호출자는 이걸
    받으면 학습을 시작하지 말고 즉시 중단해야 한다."""


def _diff_names(expected: list[str], actual: list[str] | None, label: str) -> None:
    if not actual:
        raise SchemaMismatchError(f"{label} 피처에 채널 이름(names)이 없습니다 — 기대 순서를 확인할 수 없어 중단합니다.")
    if len(expected) != len(actual):
        raise SchemaMismatchError(
            f"{label} 채널 개수가 기대값({len(expected)})과 다릅니다: 실제 {len(actual)}개 — {actual!r}"
        )
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        if e != a:
            raise SchemaMismatchError(
                f"{label}[{i}]에서 첫 불일치 발견: 기대={e!r}, 실제={a!r}"
            )


def validate_action_names(names: list[str] | None) -> None:
    _diff_names(EXPECTED_ACTION_NAMES, names, "action")


def validate_state_names(names: list[str] | None) -> None:
    _diff_names(EXPECTED_STATE_NAMES, names, "observation.state")


def validate_schema(info: dict) -> None:
    """`meta/info.json`을 그대로 로드한 dict를 받아 `features.observation.state`/
    `features.action`의 `names`를 검증한다. 하나라도 어긋나면 `SchemaMismatchError`를
    던진다 — 호출자(실행 스크립트)는 이 예외를 잡아 즉시 종료해야 한다."""
    features = info.get("features", {})
    action_names = (features.get("action") or {}).get("names")
    state_names = (features.get("observation.state") or {}).get("names")
    validate_action_names(action_names)
    validate_state_names(state_names)
