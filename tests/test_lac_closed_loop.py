"""ARD-VLA 2.0 폐루프 경로(`predict_action_chunk_with_decisions`/`closed_loop.py`)에 LAC를
연결한 것에 대한 통합 테스트. LAC 모듈 자체(손계산 케이스 등)의 단위 테스트는
`tests/test_lac.py`에, `select_action()` 경로 연결은 `tests/test_ard_integration.py`에
이미 있다 — 여기서는 "같은 `self._lac` 인스턴스를 폐루프 경로가 실제 실행 스텝마다 쓰는지,
서브태스크 전환/재시도·복구 진입 때 리셋되는지, 한 스텝에 두 번 걸리지 않는지"만 다룬다.
"""

from __future__ import annotations

import torch

from lerobot.policies.smolvla.closed_loop import run_closed_loop
from lerobot.policies.smolvla.decision_logic import DecisionLogic
from lerobot.policies.smolvla.lac import LACConfig, LatentAwareController


class FakeEnv:
    """`finish_after`번 execute()가 쌓여야 서브태스크가 끝난다(간단화를 위해 settle() 시
    리셋하지 않고 전체 execute_count로만 판단 — 이 테스트들은 서브태스크당 최대 2스텝만
    쓴다)."""

    def __init__(self, finish_after: int = 1):
        self.finish_after = finish_after
        self.obs_count = 0
        self.execute_count = 0
        self.executed: list[torch.Tensor] = []
        self.settled = 0
        self.stopped = False

    def observe(self):
        self.obs_count += 1
        return {"obs_id": self.obs_count}

    def execute(self, action):
        self.executed.append(action)
        self.execute_count += 1

    def subtask_finished(self) -> bool:
        return self.execute_count % self.finish_after == 0 and self.execute_count > 0

    def settle(self):
        self.settled += 1

    def safe_stop(self):
        self.stopped = True


SINGLE_NODE_GRAPH = {"only": {"next": None, "recoveries": [], "max_retries": 2}}
TWO_NODE_GRAPH = {
    "pick": {"next": "place", "recoveries": [], "max_retries": 2},
    "place": {"next": None, "recoveries": [], "max_retries": 2},
}


def test_predict_action_chunk_with_decisions_lac_mode_none_has_no_role_split_key(build_tiny_policy, build_tiny_batch):
    """lac_mode="none"(기본)이면 decisions에 `_lac_role_split`이 전혀 없어야 한다 — 기존
    sample_actions_with_decisions() 경로를 그대로 타고 있다는 직접적인 증거."""
    arm_dim = 4
    policy = build_tiny_policy(use_judge=True, use_ard=True, ard_arm_dim=arm_dim, arm_dim=arm_dim, num_steps=2)
    policy.eval()
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    with torch.no_grad():
        _, decisions = policy.predict_action_chunk_with_decisions(dict(batch))

    assert "_lac_role_split" not in decisions


def test_predict_action_chunk_with_decisions_lac_mode_none_is_deterministic(build_tiny_policy, build_tiny_batch):
    """`run_closed_loop`가 맨 앞에서 `policy.reset()`을 부르는 것 등 이번 변경이
    lac_mode="none" 경로의 수치를 전혀 바꾸지 않는지 — 같은 시드로 두 번 돌려 bit-for-bit
    동일한지 확인한다."""
    arm_dim = 4
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    def run_once():
        torch.manual_seed(7)
        policy = build_tiny_policy(use_judge=True, use_ard=True, ard_arm_dim=arm_dim, arm_dim=arm_dim, num_steps=2)
        policy.eval()
        env = FakeEnv(finish_after=1)
        decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
        torch.manual_seed(8)
        log = run_closed_loop(
            policy, env, decision_logic, goal="g", start_node="only", max_iterations=10,
            preprocessor=lambda obs: build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6),
        )
        return log, env.executed

    log1, executed1 = run_once()
    log2, executed2 = run_once()

    assert log1["termination_reason"] == log2["termination_reason"]
    assert log1["decision_history"] == log2["decision_history"]
    for a, b in zip(executed1, executed2, strict=True):
        assert torch.equal(a, b)


def test_judge_selector_decisions_independent_of_lac_mode(build_tiny_policy, build_tiny_batch):
    """같은 가중치/같은 noise로 한 번 forward했을 때, Judge의 `success_prob`은
    lac_mode(none vs covla)와 무관하게 완전히 동일해야 한다 — LAC 연결이 Judge/Selector의
    입력·판정 로직을 건드리지 않는지의 핵심 확인."""
    arm_dim = 4
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    torch.manual_seed(42)
    policy_none = build_tiny_policy(use_judge=True, use_ard=True, ard_arm_dim=arm_dim, arm_dim=arm_dim, num_steps=2)
    policy_none.eval()
    noise = policy_none.model.sample_noise(
        (1, policy_none.config.chunk_size, policy_none.config.max_action_dim), torch.device("cpu")
    )
    with torch.no_grad():
        _, decisions_none = policy_none.predict_action_chunk_with_decisions(dict(batch), noise=noise.clone())

    torch.manual_seed(42)
    policy_covla = build_tiny_policy(
        use_judge=True, use_ard=True, ard_arm_dim=arm_dim, arm_dim=arm_dim, num_steps=2, lac_mode="covla",
    )
    policy_covla.eval()
    with torch.no_grad():
        _, decisions_covla = policy_covla.predict_action_chunk_with_decisions(dict(batch), noise=noise.clone())

    assert torch.equal(decisions_none["success_prob"], decisions_covla["success_prob"])


def test_predict_action_chunk_with_decisions_returns_raw_unfiltered_actions(build_tiny_policy, build_tiny_batch):
    """lac_mode가 켜져 있어도 `predict_action_chunk_with_decisions()` 자체는 필터링 전
    raw 액션을 그대로 돌려줘야 한다 — chunk 단위가 아니라 실제 실행 스텝마다(러너의
    `_execute_chunk`에서) 적용하라는 요구사항이라, 여기서 먼저 필터링해버리면 한 스텝에
    두 번 거는 구조가 된다."""
    arm_dim = 4
    policy = build_tiny_policy(
        use_judge=True, use_ard=True, ard_arm_dim=arm_dim, arm_dim=arm_dim, num_steps=2, lac_mode="covla",
    )
    policy.eval()
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    with torch.no_grad():
        actions, decisions = policy.predict_action_chunk_with_decisions(dict(batch))

    role_split = decisions["_lac_role_split"]
    raw_left = role_split["shared_left"] + role_split["residual_left"]
    raw_right = role_split["shared_right"] + role_split["residual_right"]
    assert torch.allclose(actions[..., :arm_dim], raw_left, atol=1e-4)
    assert torch.allclose(actions[..., arm_dim : 2 * arm_dim], raw_right, atol=1e-4)


def test_apply_lac_step_gripper_passthrough(build_tiny_policy):
    """`apply_lac_step()`(select_action/폐루프 두 경로가 공유하는 메서드)이 그리퍼 채널을
    필터링 없이 그대로 통과시키는지."""
    arm_dim = 4
    policy = build_tiny_policy(
        use_ard=True, ard_arm_dim=arm_dim, ard_gripper_dim=1, arm_dim=arm_dim, num_steps=2, lac_mode="covla",
    )
    policy.eval()
    policy.reset()

    action_t = torch.tensor([[1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 9.0]])  # 마지막 채널이 각 팔의 그리퍼
    zeros = torch.zeros(1, arm_dim)
    shared = zeros.clone()
    shared[:, 0] = 1.0
    role_split_t = {"shared_left": shared, "shared_right": shared, "residual_left": zeros, "residual_right": zeros}

    filtered = policy.apply_lac_step(action_t, role_split_t)

    assert filtered[0, 3].item() == 9.0
    assert filtered[0, 7].item() == 9.0


def test_closed_loop_lac_resets_on_episode_subtask_and_retry(build_tiny_policy, monkeypatch):
    """LAC 리셋이 (1) 에피소드 시작, (2) 서브태스크 전환(ADVANCE), (3) 재시도(RETRY) 각각에서
    정확히 한 번씩 일어나는지 — `log_lac_events=True`로 받은 `lac_resets`를 그대로 확인한다."""
    arm_dim = 2
    policy = build_tiny_policy(
        use_judge=True, use_ard=True, ard_arm_dim=arm_dim, ard_gripper_dim=0, arm_dim=arm_dim,
        num_steps=2, lac_mode="covla",
    )
    policy.eval()

    call_count = {"n": 0}

    def stub(batch, selector_mask=None, **kwargs):
        call_count["n"] += 1
        n = call_count["n"]
        actions = torch.zeros(1, 1, 2 * arm_dim)
        prob = 0.1 if n == 2 else 0.95  # pick의 1차 시도만 실패 -> RETRY
        shared = torch.tensor([[1.0, 0.0]]).unsqueeze(1)
        residual = torch.tensor([[0.0, 0.1]]).unsqueeze(1)
        role_split = {
            "shared_left": shared, "shared_right": shared,
            "residual_left": residual, "residual_right": residual,
        }
        return actions, {"success_prob": torch.tensor([prob]), "_lac_role_split": role_split}

    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    env = FakeEnv(finish_after=1)
    decision_logic = DecisionLogic(task_graph=TWO_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)

    log = run_closed_loop(
        policy, env, decision_logic, goal="g", start_node="pick", max_iterations=10, log_lac_events=True,
    )

    # iter0: 판정 없이 pick 1차 시도 실행(forward#1). iter1: 판정(forward#2)=FAIL->RETRY
    # (reset "retry") -> 같은 iteration에서 바로 pick 2차 시도 실행(추가 forward 없음).
    # iter2: 판정(forward#3)=SUCCESS->ADVANCE to place(reset "subtask", 실행 없이 continue).
    # iter3: 판정 없이 place 1차 시도 실행(forward#4). iter4: 판정(forward#5)=SUCCESS->종료.
    reasons = [(r["iteration"], r["reason"]) for r in log["lac_resets"]]
    assert reasons == [(-1, "episode"), (1, "retry"), (2, "subtask")]
    assert log["termination_reason"] == "task_graph_end"


def test_closed_loop_lac_applied_exactly_once_per_step(build_tiny_policy, monkeypatch):
    """러너가 실제로 실행한 두 스텝의 값이, 독립적으로 한 번씩만 LAC를 돌린 참조값과 정확히
    일치하는지 — 만약 어딘가에서 같은 스텝에 LAC가 두 번 걸리면(또는 상태가 중복 진행되면)
    두 번째 스텝부터 참조값과 어긋난다."""
    arm_dim = 2
    policy = build_tiny_policy(
        use_judge=True, use_ard=True, ard_arm_dim=arm_dim, ard_gripper_dim=0, arm_dim=arm_dim,
        num_steps=2, lac_mode="covla",
    )
    policy.eval()

    shared = torch.tensor([[1.0, 0.0]])
    res_step1 = torch.tensor([[0.0, 0.1]])
    res_step2 = torch.tensor([[0.0, 0.2]])
    raw_q_step1 = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    raw_q_step2 = torch.tensor([[7.0, 7.0, 7.0, 7.0]])

    def stub(batch, selector_mask=None, **kwargs):
        actions = torch.stack([raw_q_step1, raw_q_step2], dim=1)
        role_split = {
            "shared_left": torch.stack([shared, shared], dim=1),
            "shared_right": torch.stack([shared, shared], dim=1),
            "residual_left": torch.stack([res_step1, res_step2], dim=1),
            "residual_right": torch.stack([res_step1, res_step2], dim=1),
        }
        return actions, {"success_prob": torch.tensor([0.95]), "_lac_role_split": role_split}

    monkeypatch.setattr(policy, "predict_action_chunk_with_decisions", stub)

    env = FakeEnv(finish_after=2)
    decision_logic = DecisionLogic(task_graph=SINGLE_NODE_GRAPH, tau_success=0.7, tau_fail=0.3)
    run_closed_loop(policy, env, decision_logic, goal="g", start_node="only", max_iterations=5)

    assert len(env.executed) == 2

    reference_lac = LatentAwareController(LACConfig(role_aware=False), joint_dim=2, gripper_dim=0)
    expected_step1, _ = reference_lac.step(shared, shared, res_step1, res_step1, raw_q_step1, stabilizer_is_left=True)
    expected_step2, _ = reference_lac.step(shared, shared, res_step2, res_step2, raw_q_step2, stabilizer_is_left=True)

    assert torch.equal(env.executed[0], expected_step1[0])
    assert torch.equal(env.executed[1], expected_step2[0])
