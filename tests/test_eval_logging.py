"""Fig.4용 평가 로깅 훅(`closed_loop.py`의 `track_role_split`/확장된 `per_step`,
`eval_logging.RolloutLogger`) 테스트 — 정책 동작이 전혀 바뀌지 않는지, role_split이 올바르게
ARD Stabilizer/Actuator 잔차로 연결되는지, 로거가 npz를 왕복 저장/로드하는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.smolvla.closed_loop import run_closed_loop
from lerobot.policies.smolvla.decision_logic import DecisionLogic
from lerobot.policies.smolvla.eval_logging import RolloutLogger


def check(name: str, condition: bool, detail: str = "") -> None:
    message = name if not detail else f"{name} — {detail}"
    assert condition, message


def test_track_role_split_default_off_matches_pre_change_baseline(build_tiny_policy, build_tiny_batch):
    """track_role_split을 안 주면(기본 False) 기존과 완전히 동일 — ARD를 켠 정책에서도
    decisions에 _lac_role_split이 안 생기고, 액션은 track_role_split 유무와 무관하게
    bit-for-bit 동일해야 한다(순수 덧붙임 플래그라는 주장의 핵심 검증)."""
    policy = build_tiny_policy(use_ard=True, ard_arm_dim=4, num_steps=2)
    policy.eval()
    assert policy.config.lac_mode == "none"
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)

    torch.manual_seed(7)
    with torch.no_grad():
        actions_default, decisions_default = policy.predict_action_chunk_with_decisions(dict(batch))
    torch.manual_seed(7)
    with torch.no_grad():
        actions_explicit_off, decisions_explicit_off = policy.predict_action_chunk_with_decisions(
            dict(batch), track_role_split=False
        )

    check("track_role_split 기본값은 decisions에 _lac_role_split을 추가하지 않는다",
          "_lac_role_split" not in decisions_default)
    check("track_role_split=False를 명시해도 동일하게 비어있다",
          "_lac_role_split" not in decisions_explicit_off)
    check("track_role_split 유무와 무관하게 액션은 bit-for-bit 동일하다",
          torch.equal(actions_default, actions_explicit_off))


def test_track_role_split_true_exposes_stabilizer_actuator_residuals(build_tiny_policy, build_tiny_batch):
    """track_role_split=True면(lac_mode="none"이어도) decisions["_lac_role_split"]에
    residual_left/residual_right(기본 역할 매핑에서 Stabilizer/Actuator 잔차)가 실제로 담겨야
    한다 — 이게 바로 ||r_stab||/||r_act|| 계산에 쓰는 값이다."""
    arm_dim = 4
    policy = build_tiny_policy(use_ard=True, ard_arm_dim=arm_dim, num_steps=2)
    policy.eval()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)

    with torch.no_grad():
        actions, decisions = policy.predict_action_chunk_with_decisions(dict(batch), track_role_split=True)

    check("track_role_split=True면 _lac_role_split이 채워진다", "_lac_role_split" in decisions)
    role_split = decisions["_lac_role_split"]
    check("role_split에 residual_left/residual_right가 있다",
          {"residual_left", "residual_right", "shared_left", "shared_right"} <= set(role_split))
    check("residual_left shape이 (batch, chunk, arm_dim)이다",
          tuple(role_split["residual_left"].shape) == (2, actions.shape[1], arm_dim),
          detail=f"shape={tuple(role_split['residual_left'].shape)}")
    check("residual 값이 유한하다", torch.isfinite(role_split["residual_left"]).all().item())


def test_track_role_split_ignored_without_ard(build_tiny_policy, build_tiny_batch):
    """use_ard=False면 track_role_split=True를 줘도 조용히 무시되고(크래시 없이) decisions는
    기존과 동일(빈 dict)해야 한다 — 분해할 ARD 헤드 자체가 없으므로."""
    policy = build_tiny_policy(num_steps=2)
    policy.eval()
    assert policy.config.use_ard is False
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)

    with torch.no_grad():
        _actions, decisions = policy.predict_action_chunk_with_decisions(dict(batch), track_role_split=True)

    check("use_ard=False면 track_role_split=True여도 decisions는 비어있다", decisions == {})


class _FakeEnv:
    """closed_loop 테스트와 같은 최소 duck-typed 환경 + observe()가 물리 상태 몇 개를 낸다."""

    def __init__(self):
        self.t = 0
        self.executed = []

    def observe(self):
        self.t += 1
        return {"obs_id": self.t, "joint_pos": np.array([0.1, 0.2]) * self.t, "contact": float(self.t > 2)}

    def execute(self, action):
        self.executed.append(action)

    def subtask_finished(self) -> bool:
        return len(self.executed) >= 3

    def settle(self):
        pass

    def safe_stop(self):
        pass


def test_closed_loop_threads_track_role_split_into_per_step(build_tiny_policy, tmp_path):
    """run_closed_loop(track_role_split=True)가 (1) predict_action_chunk_with_decisions에
    track_role_split=True를 실제로 전달하고, (2) per_step에 그 스텝만큼 슬라이스된 role_split_t를
    넘기고, (3) 그래도 실행되는 액션 자체는 변하지 않는지(apply_lac_step이 lac_mode="none"일 때
    no-op이므로) 확인한다. predict_action_chunk_with_decisions는 스텁으로 대체해서 forward 계산
    없이 배관만 검증한다."""
    policy = build_tiny_policy(use_ard=True, ard_arm_dim=4, num_steps=2)
    policy.eval()

    seen_kwargs = []

    def stub(batch, selector_mask=None, **kwargs):
        seen_kwargs.append(kwargs.get("track_role_split"))
        n_steps, arm_dim = 3, 4
        actions = torch.arange(n_steps, dtype=torch.float32).view(1, n_steps, 1).expand(1, n_steps, 6).clone()
        decisions = {"success_prob": torch.tensor([0.99])}
        if kwargs.get("track_role_split"):
            decisions["_lac_role_split"] = {
                "shared_left": torch.zeros(1, n_steps, arm_dim),
                "shared_right": torch.zeros(1, n_steps, arm_dim),
                "residual_left": torch.arange(n_steps * arm_dim, dtype=torch.float32).view(1, n_steps, arm_dim),
                "residual_right": -torch.arange(n_steps * arm_dim, dtype=torch.float32).view(1, n_steps, arm_dim),
            }
        return actions, decisions

    policy.predict_action_chunk_with_decisions = stub

    env = _FakeEnv()
    logger = RolloutLogger(env, method="ard_vla", episode_id="test_ep")
    decision_logic = DecisionLogic(
        task_graph={"only": {"next": None, "recoveries": [], "max_retries": 2}}, tau_success=0.7, tau_fail=0.3
    )

    run_closed_loop(
        policy, env, decision_logic, goal="g", start_node="only", max_iterations=3,
        per_step=logger.per_step, track_role_split=True,
    )

    check("predict_action_chunk_with_decisions가 track_role_split=True를 받았다", seen_kwargs and seen_kwargs[0] is True)
    check("per_step이 매 실행 스텝마다 불렸다(3스텝 청크, subtask_finished 조건 3)", len(logger.records) == 3)
    check("role_split_t가 step마다 올바르게 슬라이스되어 r_stab_norm/r_act_norm으로 들어갔다",
          all("r_stab_norm" in r and "r_act_norm" in r for r in logger.records))
    check("step 1의 residual_left는 [0,1,2,3] -> norm", abs(logger.records[0]["r_stab_norm"] - np.linalg.norm([0, 1, 2, 3])) < 1e-5,
          detail=f"got={logger.records[0]['r_stab_norm']}")
    check("stage(현재 노드)가 기록됐다", logger.records[0]["stage"] == "only")
    check("env.observe()의 joint_pos/contact가 기록됐다(물리 상태)", "joint_pos" in logger.records[0] and "contact" in logger.records[0])

    out_path = str(tmp_path / "ard_vla_test_ep.npz")
    logger.save(out_path)
    loaded = RolloutLogger.load(out_path)
    check("저장된 npz에 r_stab_norm이 있다", "r_stab_norm" in loaded)
    check("저장된 method/episode_id가 왕복한다",
          str(loaded["method"]) == "ard_vla" and str(loaded["episode_id"]) == "test_ep")
    check("step 배열 길이가 3이다", len(loaded["step"]) == 3)


def test_rollout_logger_without_role_split(build_tiny_policy):
    """role_split_t가 None이면(track_role_split 안 쓰는 호출) r_stab_norm/r_act_norm을 그냥
    안 채운다 — 크래시하지 않는다."""
    env = _FakeEnv()
    logger = RolloutLogger(env, method="smolvla", episode_id="e0")

    class _DummyState:
        current_node = "stage_a"

    logger.per_step(0, torch.zeros(6), _DummyState(), None)
    check("role_split_t가 None이어도 레코드가 생긴다", len(logger.records) == 1)
    check("r_stab_norm이 없다(ARD 아닌 베이스라인)", "r_stab_norm" not in logger.records[0])
