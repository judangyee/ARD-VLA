"""`lerobot.policies.smolvla.decision_logic`에 대한 순수 Python 단위 테스트 — torch를
전혀 쓰지 않는다(이 모듈 자체가 torch 의존성이 없다는 걸 확인하는 것도 테스트 목적 중 하나).
"""

from __future__ import annotations

import pytest

from lerobot.policies.smolvla.decision_logic import (
    DecisionKind,
    DecisionLogic,
    EpisodeState,
)

TASK_GRAPH = {
    "pick": {"next": "insert", "recoveries": ["regrasp"], "max_retries": 2},
    "insert": {"next": None, "recoveries": ["realign", "regrasp"], "max_retries": 2},
}
OPTION_VOCAB = ["regrasp", "realign", "unrelated_option"]


def _logic(**overrides) -> DecisionLogic:
    kwargs = dict(task_graph=TASK_GRAPH, option_vocab=OPTION_VOCAB, tau_success=0.7, tau_fail=0.3)
    kwargs.update(overrides)
    return DecisionLogic(**kwargs)


def test_tau_fail_cannot_exceed_tau_success():
    with pytest.raises(ValueError, match="tau_fail"):
        DecisionLogic(task_graph=TASK_GRAPH, tau_success=0.3, tau_fail=0.7)


def test_three_threshold_zones():
    logic = _logic()
    state = EpisodeState(current_node="pick")

    advance = logic.decide({"success_prob": 0.9}, state)
    assert advance.kind == DecisionKind.ADVANCE
    assert advance.target == "insert"
    assert state.current_node == "insert"
    assert state.completed == ["pick"]

    state2 = EpisodeState(current_node="pick")
    recheck = logic.decide({"success_prob": 0.5}, state2)
    assert recheck.kind == DecisionKind.RECHECK
    assert state2.recheck_counts["pick"] == 1

    state3 = EpisodeState(current_node="pick")
    fail = logic.decide({"success_prob": 0.1}, state3)
    assert fail.kind in (DecisionKind.RETRY, DecisionKind.RECOVER)
    assert state3.retry_counts["pick"] == 1


def test_advance_to_terminal_node_has_none_target():
    logic = _logic()
    state = EpisodeState(current_node="insert")
    decision = logic.decide({"success_prob": 0.95}, state)
    assert decision.kind == DecisionKind.ADVANCE
    assert decision.target is None


def test_retry_limit_exceeded_stops():
    logic = _logic()
    state = EpisodeState(current_node="pick")
    # failure_probs 없음 -> rule table이 None -> RETRY를 max_retries(2)번까지 반복
    d1 = logic.decide({"success_prob": 0.1}, state)
    assert d1.kind == DecisionKind.RETRY
    d2 = logic.decide({"success_prob": 0.1}, state)
    assert d2.kind == DecisionKind.RETRY
    d3 = logic.decide({"success_prob": 0.1}, state)
    assert d3.kind == DecisionKind.STOP
    assert "재시도 한도" in d3.reason


def test_recheck_limit_exceeded_stops():
    logic = _logic(max_recheck=1)
    state = EpisodeState(current_node="pick")
    d1 = logic.decide({"success_prob": 0.5}, state)
    assert d1.kind == DecisionKind.RECHECK
    d2 = logic.decide({"success_prob": 0.5}, state)
    assert d2.kind == DecisionKind.STOP
    assert "재관측 한도" in d2.reason


def test_rule_table_used_when_selector_absent():
    """selector_probs가 decisions에 없으면(use_selector=False와 동일한 상황) failure_probs
    기반 rule table만으로 복구 동작을 고른다."""
    logic = _logic(failure_type_to_recovery={0: "realign"})
    state = EpisodeState(current_node="insert")
    decision = logic.decide({"success_prob": 0.1, "failure_probs": [0.9, 0.1]}, state)
    assert decision.kind == DecisionKind.RECOVER
    assert decision.target == "realign"


def test_rule_table_retry_when_no_failure_type():
    logic = _logic()
    state = EpisodeState(current_node="insert")
    decision = logic.decide({"success_prob": 0.1}, state)  # failure_probs 없음
    assert decision.kind == DecisionKind.RETRY


def test_disallowed_selector_choice_is_rejected_and_falls_back_to_rule_table():
    """Selector가 현재 노드에서 허용되지 않은 옵션("unrelated_option", 인덱스 2)을 고르면
    안전 규칙이 거부하고 rule table(failure_probs 기반)로 대체해야 한다."""
    logic = _logic(failure_type_to_recovery={1: "regrasp"})
    state = EpisodeState(current_node="pick")  # pick의 recoveries는 ["regrasp"]뿐
    decisions = {
        "success_prob": 0.1,
        "selector_probs": [0.0, 0.0, 1.0],  # argmax=2 -> "unrelated_option", pick에서는 허용 안 됨
        "failure_probs": [0.1, 0.9],  # argmax=1 -> "regrasp" (허용됨)
    }
    decision = logic.decide(decisions, state)
    assert decision.kind == DecisionKind.RECOVER
    assert decision.target == "regrasp"


def test_allowed_selector_choice_is_used_directly():
    logic = _logic()
    state = EpisodeState(current_node="insert")  # recoveries = ["realign", "regrasp"]
    decisions = {"success_prob": 0.1, "selector_probs": [0.0, 1.0, 0.0]}  # argmax=1 -> "realign"
    decision = logic.decide(decisions, state)
    assert decision.kind == DecisionKind.RECOVER
    assert decision.target == "realign"


def test_allowed_mask_matches_node_recoveries():
    logic = _logic()
    state = EpisodeState(current_node="pick")
    mask = logic.allowed_mask(state)
    assert mask == [True, False, False]  # option_vocab = [regrasp, realign, unrelated_option]

    state2 = EpisodeState(current_node="insert")
    mask2 = logic.allowed_mask(state2)
    assert mask2 == [True, True, False]


def test_instruction_text_includes_goal_subtask_and_completed():
    logic = _logic()
    state = EpisodeState(current_node="insert", completed=["pick"])
    text = logic.instruction_text("peg in hole 작업을 완료하라", state)
    assert "peg in hole 작업을 완료하라" in text
    assert "insert" in text
    assert "pick" in text


def test_decide_raises_without_success_prob():
    logic = _logic()
    state = EpisodeState(current_node="pick")
    with pytest.raises(ValueError, match="success_prob"):
        logic.decide({}, state)
