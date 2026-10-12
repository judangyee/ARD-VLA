"""`lerobot.policies.smolvla.closed_loop.run_closed_loop`에 대한 통합 테스트 — 가짜 환경
(duck typing) + `conftest.py`의 tiny policy로, Judge 신호만 스크립트로 고정해서(실제 작은
Judge MLP의 랜덤 초기화 출력에 의존하지 않고) "실패 → 남은 청크 버리고 재시도 → 성공 →
다음 노드로 → 종료" 한 번의 전체 패스를 확인한다.
"""

from __future__ import annotations

import torch

from lerobot.policies.smolvla.closed_loop import run_closed_loop
from lerobot.policies.smolvla.decision_logic import DecisionLogic

TASK_GRAPH = {
    "pick": {"next": "place", "recoveries": [], "max_retries": 2},
    "place": {"next": None, "recoveries": [], "max_retries": 2},
}


class FakeEnv:
    """매 청크의 첫 스텝에서 바로 서브태스크가 끝났다고 보는 가장 단순한 가짜 환경."""

    def __init__(self):
        self.executed = []
        self.settled = 0
        self.stopped = False

    def observe(self):
        return {"dummy": 0}

    def execute(self, action):
        self.executed.append(action)

    def subtask_finished(self) -> bool:
        return True

    def settle(self):
        self.settled += 1

    def safe_stop(self):
        self.stopped = True


def test_closed_loop_failure_then_retry_then_success_then_terminate(build_tiny_policy, monkeypatch):
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()

    actions_stub = torch.zeros(1, 4, 1)
    call_count = {"n": 0}
    cancel_calls = []

    def fake_predict(batch, selector_mask=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            decisions = {"success_prob": torch.tensor([0.1])}  # pick 1차 시도 실패
        else:
            decisions = {"success_prob": torch.tensor([0.95])}  # 그 이후는 전부 성공
        return actions_stub, decisions

    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", fake_predict)

    original_cancel = policy.cancel_pending_actions

    def tracking_cancel():
        cancel_calls.append(1)
        return original_cancel()

    monkeypatch.setattr(policy, "cancel_pending_actions", tracking_cancel)

    decision_logic = DecisionLogic(task_graph=TASK_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv()

    log = run_closed_loop(
        policy,
        env,
        decision_logic,
        goal="place the peg",
        start_node="pick",
        preprocessor=lambda obs: {},
        max_iterations=10,
    )

    assert call_count["n"] == 3  # 실패(재시도) + pick 성공(advance) + place 성공(advance, 종료)
    assert len(cancel_calls) == 1  # 실패 1번에서만 cancel_pending_actions 호출

    history = log["decision_history"]
    assert [d["kind"] for d in history] == ["retry", "advance", "advance"]
    assert history[0]["node"] == "pick"
    assert history[1]["target"] == "place"
    assert history[2]["target"] is None

    assert log["termination_reason"] == "task_graph_end"
    assert log["retry_counts"] == {}  # place로 advance하면서 pick의 retry_count는 pop됨
    assert env.settled == 3
    assert not env.stopped
    assert len(log["judge_probabilities"]) == 3


def test_closed_loop_stops_on_retry_limit_exceeded(build_tiny_policy, monkeypatch):
    """계속 실패하면 max_retries를 넘겨 STOP으로 끝나고 env.safe_stop()이 호출돼야 한다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    actions_stub = torch.zeros(1, 4, 1)

    def always_fail(batch, selector_mask=None, **kwargs):
        return actions_stub, {"success_prob": torch.tensor([0.0])}

    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", always_fail)

    decision_logic = DecisionLogic(task_graph=TASK_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv()

    log = run_closed_loop(
        policy, env, decision_logic, goal="place the peg", start_node="pick",
        preprocessor=lambda obs: {}, max_iterations=10,
    )

    assert log["termination_reason"] == "stop"
    assert env.stopped
    assert log["decision_history"][-1]["kind"] == "stop"
