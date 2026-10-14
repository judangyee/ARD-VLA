"""`lerobot.policies.smolvla.closed_loop.run_closed_loop`에 대한 통합 테스트 — 가짜 환경
(duck typing) + `conftest.py`의 tiny policy로, Judge 신호만 스크립트로 고정해서(실제 작은
Judge MLP의 랜덤 초기화 출력에 의존하지 않고) 판정 시점/청크-서브태스크 구분/RECOVER 실행/
역정규화/서브태스크 스텝 예산을 각각 확인한다.
"""

from __future__ import annotations

import torch

import pytest

from lerobot.policies.smolvla.closed_loop import run_closed_loop
from lerobot.policies.smolvla.decision_logic import DecisionLogic


class FakeEnv:
    """observe()가 매 호출마다 늘어나는 `obs_id`를 담아 "판정에 어느 관측이 쓰였는지" 추적할
    수 있게 하고, `finish_after`(settle() 이후 몇 번의 execute()가 쌓여야
    `subtask_finished()`가 True가 되는지)로 "청크 1개=서브태스크 1개"와 "서브태스크가 청크
    여러 개에 걸침"을 둘 다 표현할 수 있는 가짜 환경. `support_recovery=False`면
    `run_recovery` 속성 자체가 없어서 `hasattr(env, "run_recovery")`가 False다."""

    def __init__(self, finish_after: int | None = 1, support_recovery: bool = True):
        self.finish_after = finish_after
        self.obs_count = 0
        self.execute_count = 0
        self.execute_count_since_settle = 0
        self.executed = []
        self.settled = 0
        self.stopped = False
        self.recovered = []
        if support_recovery:
            self.run_recovery = self._run_recovery

    def _run_recovery(self, recovery_id):
        self.recovered.append(recovery_id)

    def observe(self):
        self.obs_count += 1
        return {"obs_id": self.obs_count}

    def execute(self, action):
        self.executed.append(action)
        self.execute_count += 1
        self.execute_count_since_settle += 1

    def subtask_finished(self) -> bool:
        if self.finish_after is None:
            return False
        return self.execute_count_since_settle >= self.finish_after

    def settle(self):
        self.settled += 1
        self.execute_count_since_settle = 0

    def safe_stop(self):
        self.stopped = True


def _make_stub(
    success_by_call: dict[int, float] | None = None,
    default_prob: float = 0.95,
    failure_probs_by_call: dict[int, list[float]] | None = None,
    selector_probs_by_call: dict[int, list[float]] | None = None,
    action_shape: tuple[int, int, int] = (1, 1, 1),
):
    """`predict_action_chunk_with_decisions`를 대신할 스텁. 호출 횟수(1부터 시작)별로
    `success_prob`/`failure_probs`/`selector_probs`를 스크립트로 고정하고, `decisions`에
    `_obs_id`(그 호출이 본 배치의 `obs_id`)를 끼워 넣어 "어느 관측으로 판정했는지" 테스트에서
    바로 확인할 수 있게 한다."""
    call_count = {"n": 0}

    def stub(batch, selector_mask=None, **kwargs):
        call_count["n"] += 1
        n = call_count["n"]
        prob = (success_by_call or {}).get(n, default_prob)
        decisions = {"success_prob": torch.tensor([prob]), "_obs_id": batch.get("obs_id")}
        if failure_probs_by_call and n in failure_probs_by_call:
            decisions["failure_probs"] = torch.tensor([failure_probs_by_call[n]])
        if selector_probs_by_call and n in selector_probs_by_call:
            decisions["selector_probs"] = torch.tensor([selector_probs_by_call[n]])
        actions = torch.zeros(*action_shape)
        return actions, decisions

    return stub, call_count


SINGLE_NODE_GRAPH = {"only": {"next": None, "recoveries": [], "max_retries": 2}}
TWO_NODE_GRAPH = {
    "pick": {"next": "place", "recoveries": [], "max_retries": 2},
    "place": {"next": None, "recoveries": [], "max_retries": 2},
}


def test_closed_loop_failure_then_retry_then_success_then_terminate(build_tiny_policy, monkeypatch):
    """실패(RETRY, 추가 forward 없음) -> 성공(ADVANCE, 다음 iteration에서 재forward) -> 성공
    (ADVANCE, 종료) 전체 패스 — forward_calls/judged 로그와 cancel 호출 횟수까지 정확히 확인."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()

    # 호출 순서: #1 unjudged(최초 pick 시도) #2 judged=FAIL(RETRY) #3 judged=SUCCESS(ADVANCE)
    #           #4 unjudged(place 시도) #5 judged=SUCCESS(ADVANCE, 종료)
    stub, call_count = _make_stub(success_by_call={2: 0.1}, default_prob=0.95)
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    cancel_calls = []
    original_cancel = policy.cancel_pending_actions

    def tracking_cancel():
        cancel_calls.append(1)
        return original_cancel()

    monkeypatch.setattr(policy, "cancel_pending_actions", tracking_cancel)

    decision_logic = DecisionLogic(task_graph=TWO_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=1)

    log = run_closed_loop(
        policy, env, decision_logic, goal="place the peg", start_node="pick", max_iterations=10,
    )

    assert call_count["n"] == 5
    assert len(cancel_calls) == 1  # RETRY 1번에서만 호출

    kinds = [d["kind"] for d in log["decision_history"]]
    assert kinds == ["retry", "advance", "advance"]
    assert log["decision_history"][1]["target"] == "place"
    assert log["decision_history"][2]["target"] is None

    assert log["forward_calls"] == [1, 1, 1, 1, 1]
    assert log["judged"] == [False, True, True, False, True]
    assert log["termination_reason"] == "task_graph_end"
    assert log["retry_counts"] == {}
    assert not env.stopped


def test_closed_loop_stops_on_retry_limit_exceeded(build_tiny_policy, monkeypatch):
    """계속 실패하면 max_retries를 넘겨 STOP으로 끝나고 env.safe_stop()이 호출돼야 한다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(default_prob=0.0)  # 항상 실패
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=TWO_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=1)

    log = run_closed_loop(
        policy, env, decision_logic, goal="place the peg", start_node="pick", max_iterations=10,
    )

    assert log["termination_reason"] == "stop"
    assert env.stopped
    assert log["decision_history"][-1]["kind"] == "stop"


def test_closed_loop_judgment_uses_post_execution_observation(build_tiny_policy, monkeypatch):
    """판정에 쓰이는 obs_id가 최초 관측(1)이 아니라 첫 실행 이후 관측(2)인지 — "한 박자 늦은
    판정" 버그가 고쳐졌는지의 핵심 확인."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(default_prob=0.95)
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=1)

    judged_obs_ids = []
    original_decide = decision_logic.decide

    def tracking_decide(decisions, state):
        judged_obs_ids.append(decisions["_obs_id"])
        return original_decide(decisions, state)

    monkeypatch.setattr(decision_logic, "decide", tracking_decide)

    log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="only", max_iterations=10)

    assert judged_obs_ids == [2]  # obs_id=1(최초 관측)이 아니라 obs_id=2(실행+settle 후 관측)
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_multi_chunk_subtask_defers_judgment_and_settle(build_tiny_policy, monkeypatch):
    """서브태스크가 청크 3개에 걸칠 때, 앞의 두 청크 뒤에는 decide()도 settle()도 호출되지
    않고, 세 번째 청크(=subtask_finished() True) 뒤에야 둘 다 정확히 1번 호출되는지."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(default_prob=0.95)
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=3)

    settled_at_decide = []
    executed_at_decide = []
    original_decide = decision_logic.decide

    def tracking_decide(decisions, state):
        settled_at_decide.append(env.settled)
        executed_at_decide.append(env.execute_count)
        return original_decide(decisions, state)

    monkeypatch.setattr(decision_logic, "decide", tracking_decide)

    log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="only", max_iterations=10)

    assert executed_at_decide == [3]  # decide() 호출 시점에 이미 3번 실행된 뒤였음
    assert settled_at_decide == [1]  # settle()도 그 시점에 정확히 1번만
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_recover_runs_recovery_and_resumes_same_node(build_tiny_policy, monkeypatch):
    """RECOVER 결정이 실제로 env.run_recovery(id)를 부르고, 그 뒤 같은 노드를 다시 시도해서
    결국 성공/종료하는지."""
    policy = build_tiny_policy(use_judge=True, judge_num_failure_types=2, num_steps=2)
    policy.eval()

    stub, _call_count = _make_stub(
        success_by_call={2: 0.1},
        failure_probs_by_call={2: [0.9, 0.1]},  # argmax=0 -> "regrasp"
        default_prob=0.95,
    )
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    graph = {
        "pick": {"next": "done", "recoveries": ["regrasp"], "max_retries": 2},
        "done": {"next": None, "recoveries": [], "max_retries": 2},
    }
    decision_logic = DecisionLogic(
        task_graph=graph, tau_success=0.7, tau_fail=0.3, failure_type_to_recovery={0: "regrasp"},
    )
    env = FakeEnv(finish_after=1)

    log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="pick", max_iterations=20)

    assert env.recovered == ["regrasp"]
    kinds = [d["kind"] for d in log["decision_history"]]
    assert kinds[0] == "recover"
    assert log["decision_history"][0]["target"] == "regrasp"
    assert log["decision_history"][0]["recovery_unsupported"] is False
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_recover_falls_back_to_retry_when_unsupported(build_tiny_policy, monkeypatch):
    """환경에 run_recovery()가 없으면 RECOVER가 경고와 함께 RETRY처럼 처리되고
    (decision_history에 recovery_unsupported=True가 남고), 추가 forward 없이 이번 forward의
    actions로 바로 재시도한다."""
    policy = build_tiny_policy(use_judge=True, judge_num_failure_types=2, num_steps=2)
    policy.eval()

    stub, _call_count = _make_stub(
        success_by_call={2: 0.1, 3: 0.95},
        failure_probs_by_call={2: [0.9, 0.1]},
        default_prob=0.95,
    )
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    graph = {"pick": {"next": None, "recoveries": ["regrasp"], "max_retries": 2}}
    decision_logic = DecisionLogic(
        task_graph=graph, tau_success=0.7, tau_fail=0.3, failure_type_to_recovery={0: "regrasp"},
    )
    env = FakeEnv(finish_after=1, support_recovery=False)
    assert not hasattr(env, "run_recovery")

    with pytest.warns(UserWarning, match="run_recovery"):
        log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="pick", max_iterations=10)

    kinds = [d["kind"] for d in log["decision_history"]]
    assert kinds[0] == "recover"
    assert log["decision_history"][0]["recovery_unsupported"] is True
    assert env.recovered == []  # run_recovery 자체가 없으니 당연히 아무것도 안 쌓임
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_recheck_takes_no_action_and_rejudges(build_tiny_policy, monkeypatch):
    """RECHECK이면 행동 실행 없이 다시 관측·판정하고(env.execute()가 추가로 불리지 않음),
    settle()은 호출된다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(success_by_call={2: 0.5}, default_prob=0.95)
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3, max_recheck=3)
    env = FakeEnv(finish_after=1)

    log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="only", max_iterations=10)

    kinds = [d["kind"] for d in log["decision_history"]]
    assert kinds == ["recheck", "advance"]
    assert env.execute_count == 1  # 최초(unjudged) 실행 1번뿐, RECHECK은 추가 실행을 안 만듦
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_recheck_limit_exceeded_stops(build_tiny_policy, monkeypatch):
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(default_prob=0.5)  # 항상 중간 구간
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3, max_recheck=2)
    env = FakeEnv(finish_after=1)

    log = run_closed_loop(policy, env, decision_logic, goal="g", start_node="only", max_iterations=20)

    assert log["termination_reason"] == "stop"
    assert env.stopped
    assert log["decision_history"][-1]["kind"] == "stop"


def test_closed_loop_postprocessor_applied_before_execution(build_tiny_policy, monkeypatch):
    """postprocessor(역정규화)가 pre_chunk 훅 이후, 실행 직전에 적용되는지 — pre_chunk은
    역정규화 전 값을 받아야 한다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()

    raw_action = torch.tensor([[[1.0]]])

    def stub(batch, selector_mask=None, **kwargs):
        return raw_action.clone(), {"success_prob": torch.tensor([0.95])}

    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=1)

    pre_chunk_actions = []

    def pre_chunk(actions, decisions, state):
        pre_chunk_actions.append(actions.clone())

    run_closed_loop(
        policy, env, decision_logic, goal="g", start_node="only",
        postprocessor=lambda a: a * 1000.0, pre_chunk=pre_chunk, max_iterations=5,
    )

    assert torch.equal(env.executed[0], raw_action[0, 0] * 1000.0)
    assert torch.equal(pre_chunk_actions[0], raw_action)


def test_closed_loop_subtask_step_budget_exceeded_triggers_judgment(build_tiny_policy, monkeypatch):
    """환경이 스스로 subtask_finished()를 True로 만들지 않아도, max_subtask_steps를 넘기면
    종료로 보고 판정으로 넘어가야 한다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    policy.eval()
    stub, _call_count = _make_stub(default_prob=0.95)
    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    env = FakeEnv(finish_after=None)  # 환경은 절대 스스로 안 끝남

    log = run_closed_loop(
        policy, env, decision_logic, goal="g", start_node="only",
        max_subtask_steps=2, max_iterations=10,
    )

    assert env.execute_count == 2  # 예산(2) 정확히 소진된 뒤 멈춤
    assert env.settled == 1
    assert log["termination_reason"] == "task_graph_end"
