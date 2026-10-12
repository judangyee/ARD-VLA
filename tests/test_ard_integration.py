"""`tests/test_ard.py`(순수 합성 텐서 단위 테스트)와 달리, 이 파일은 `conftest.py`의
`build_tiny_policy`/`build_tiny_batch` fixture로 실제 `SmolVLAPolicy` 전체(forward/backward/
추론 경로 포함)를 작은 합성 SmolVLM 백본 위에서 생성해 검증한다 — `AutoConfig.from_pretrained`/
`AutoProcessor.from_pretrained`를 몽키패치해서 Hugging Face Hub 접근이 전혀 없다.

여기 있는 두 테스트는 scripts/test_ard.py에 있던 순수 단위 테스트만으로는 확인할 수 없던
것들이다:
  (d) ForceHead가 추론 경로(sample_actions/denoise_step)에서 실제로 전혀 호출되지 않는지
  (f) 오늘 추가된 새 플래그(ard_reg_time_weighting/ard_use_force_head/ard_symmetric)가 전부
      기본값(off)일 때, 전체 정책의 forward+backward 결과가 이 기능들을 추가하기 전 코드
      (commit ae09269)와 bit-for-bit 동일한지
"""

from __future__ import annotations

import torch

import pytest


def test_force_head_not_called_during_inference(build_tiny_policy, build_tiny_batch):
    """ForceHead는 학습 손실(force_loss) 계산에만 쓰이는 순수 보조 출력이라고 설계했다 —
    추론 경로(predict_action_chunk -> VLAFlowMatching.sample_actions -> denoise_step)에서는
    호출되면 안 된다. ForceHead.forward를 "호출되면 즉시 실패"로 바꿔치기해서 직접 확인한다."""
    policy = build_tiny_policy(use_ard=True, ard_use_force_head=True, num_steps=2)
    assert policy.model.ard_force_head is not None

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ForceHead.forward가 추론 경로에서 호출됐다 — 호출되면 안 된다 (순수 학습 전용 보조 출력).")

    policy.model.ard_force_head.forward = _must_not_be_called
    policy.eval()

    batch = build_tiny_batch(batch_size=1)
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)

    assert torch.isfinite(actions).all()


def test_all_new_flags_off_matches_pre_change_baseline(build_tiny_policy, build_tiny_batch):
    """ard_reg_time_weighting="none"(기본), ard_use_force_head=False(기본),
    ard_symmetric=False(기본) — 오늘(2026-10-02) 추가된 세 플래그 전부 기본값(off)일 때,
    고정 시드로 뽑은 loss/loss_dict 값이 이 기능들을 추가하기 전 코드(commit ae09269, 2026-10-02
    세션의 1~3단계 작업 시작 직전)와 정확히 일치해야 한다.

    아래 EXPECTED_* 상수는 commit ae09269 체크아웃(ard.py/modeling_smolvla.py/
    configuration_smolvla.py만 그 커밋 버전으로 임시 교체) 상태에서 이 테스트와 동일한 시드/
    설정으로 정책을 만들고 한 번 forward+backward한 실측값을 그대로 옮겨 적은 것이다 — 두
    커밋(ae09269와 현재 HEAD) 모두에서 직접 돌려서 값이 정확히 같음을 확인했다
    (PROGRESS.md의 4단계 기록 참고).

    `ard_gripper_dim=0`을 명시적으로 넘긴다 — 이 플래그는 ae09269 **이후**(OpenArm 8DoF 작업)에
    추가됐고 기본값이 1(그리퍼를 smooth/traj에서 제외)이라, 명시하지 않으면 ae09269에는 없던
    동작이 섞여 들어가서 이 테스트의 "세 플래그 모두 off면 ae09269와 bit-exact" 취지가
    깨진다. `ard_gripper_dim`의 새 동작 자체는 test_ard.py의 전용 테스트가 확인한다.
    """
    torch.manual_seed(1)
    policy = build_tiny_policy(arm_dim=7, use_ard=True, ard_arm_dim=7, ard_gripper_dim=0)
    policy.train()

    torch.manual_seed(2)
    batch = build_tiny_batch(batch_size=2, arm_dim=7, chunk=8, lang_len=6)

    loss, loss_dict = policy.forward(batch)
    loss.backward()
    grad_norm_sum = sum(p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None)

    EXPECTED_LOSS = 4.929270267486572
    EXPECTED_LOSS_DICT = {
        "ard_actuator_loss": 5.5550971031188965,
        "ard_force_loss": 0.0,
        "ard_pos_loss": 2.204470157623291,
        "ard_smooth_loss": 1.1704199314117432,
        "ard_stabilizer_loss": 3.4690074920654297,
        "ard_traj_loss": 3.444744348526001,
        "loss": 4.929270267486572,
        "losses_after_forward": 2.204470157623291,
        "losses_after_rm_padding": 2.204470157623291,
    }
    EXPECTED_GRAD_NORM_SUM = 168.44144497999514

    assert loss.item() == EXPECTED_LOSS
    for key, expected_value in EXPECTED_LOSS_DICT.items():
        actual = loss_dict[key]
        if hasattr(actual, "item"):
            actual = actual.item()
        assert actual == expected_value, f"{key}: got {actual!r}, expected {expected_value!r}"
    assert grad_norm_sum == EXPECTED_GRAD_NORM_SUM

    # ard_mode는 오늘 새로 추가된 키라 ae09269에는 없었다 — 값만 확인한다(키 자체의 유무는
    # "새 기능 추가"이지 "기존 동작이 bit 단위로 바뀜"이 아니므로 이 비교 대상에서 제외했다).
    assert loss_dict["ard_mode"] == "asymmetric"


def _build_torque_aware_policy(build_tiny_policy, arm_dim: int = 8, torque_dim: int = 16, **overrides):
    """ard_use_joint_torque=True로 관절 토크 입력을 받는 작은 정책을 만든다 — 기본
    input_features(OBS_IMAGE/OBS_STATE)에 ARD_JOINT_TORQUE 피처를 추가하고, concat 후에도
    max_state_dim을 넘지 않도록 넉넉히 넓힌다."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.ard import ARD_JOINT_TORQUE
    from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

    input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2 * arm_dim,)),
        ARD_JOINT_TORQUE: PolicyFeature(type=FeatureType.STATE, shape=(torque_dim,)),
    }
    return build_tiny_policy(
        arm_dim=arm_dim,
        use_ard=True,
        ard_use_joint_torque=True,
        input_features=input_features,
        max_state_dim=2 * arm_dim + torque_dim,
        **overrides,
    )


def test_joint_torque_is_concatenated_before_padding(build_tiny_policy, build_tiny_batch):
    """ard_use_joint_torque=True면 SmolVLAPolicy.prepare_state()가 관절 토크를
    observation.state 뒤에 concat해야 한다 — state 자체는 안 바뀌고, 토크 값을 바꾸면 패딩
    전 벡터의 해당 구간만 바뀌어야 한다."""
    from lerobot.policies.smolvla.ard import ARD_JOINT_TORQUE

    arm_dim, torque_dim = 8, 16
    policy = _build_torque_aware_policy(build_tiny_policy, arm_dim=arm_dim, torque_dim=torque_dim)
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=8, lang_len=6)

    batch_zero = dict(batch)
    batch_zero[ARD_JOINT_TORQUE] = torch.zeros(2, torque_dim)
    batch_one = dict(batch)
    batch_one[ARD_JOINT_TORQUE] = torch.ones(2, torque_dim)

    state_zero = policy.prepare_state(batch_zero)
    state_one = policy.prepare_state(batch_one)

    state_dim = 2 * arm_dim
    assert torch.equal(state_zero[:, :state_dim], state_one[:, :state_dim]), (
        "토크 값이 바뀌어도 원래 state 채널은 그대로여야 한다"
    )
    assert torch.equal(state_zero[:, state_dim : state_dim + torque_dim], torch.zeros(2, torque_dim))
    assert torch.equal(state_one[:, state_dim : state_dim + torque_dim], torch.ones(2, torque_dim))
    # concat 뒤 나머지는 여전히 (max_state_dim까지) 0 패딩이어야 한다.
    assert torch.equal(
        state_zero[:, state_dim + torque_dim :], torch.zeros(2, policy.config.max_state_dim - state_dim - torque_dim)
    )


def test_joint_torque_missing_key_raises(build_tiny_policy, build_tiny_batch):
    """ard_use_joint_torque=True인데 배치에 ARD_JOINT_TORQUE 키가 없으면, 과거처럼 조용히
    무시하지 않고 명확한 ValueError를 내야 한다."""
    policy = _build_torque_aware_policy(build_tiny_policy, arm_dim=8, torque_dim=16)
    batch = build_tiny_batch(batch_size=2, arm_dim=8, chunk=8, lang_len=6)  # ARD_JOINT_TORQUE 없음

    with pytest.raises(ValueError, match="ard_use_joint_torque"):
        policy.prepare_state(batch)


def test_joint_torque_flows_through_full_forward_pass(build_tiny_policy, build_tiny_batch):
    """ard_use_joint_torque=True에서도 전체 forward(학습)/predict_action_chunk(추론) 경로가
    끝까지 정상적으로 돌아가는지(에러 없이, 유한한 값으로) 확인한다."""
    from lerobot.policies.smolvla.ard import ARD_JOINT_TORQUE

    arm_dim, torque_dim = 8, 16
    policy = _build_torque_aware_policy(build_tiny_policy, arm_dim=arm_dim, torque_dim=torque_dim)
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=8, lang_len=6)
    batch[ARD_JOINT_TORQUE] = torch.randn(2, torque_dim)

    policy.train()
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss).all()
    loss.backward()

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(dict(batch))
    assert torch.isfinite(actions).all()


# --- 폐루프 구조(Judge/Selector, predict_action_chunk_with_decisions) 통합 테스트 ---
# 순수 텐서 단위 테스트(Judge 모듈 자체, config 검증)는 tests/test_judge.py에 있다. 여기서는
# 실제 SmolVLAPolicy 전체(prefix forward 1회 공유, use_judge/use_selector가 전부 꺼졌을 때의
# bit-exact 여부)를 확인한다.


def test_judge_flags_off_matches_predict_action_chunk(build_tiny_policy, build_tiny_batch):
    """use_judge=use_selector=False(기본)면 predict_action_chunk_with_decisions의 액션이
    predict_action_chunk과 bit-for-bit 동일해야 하고, decisions는 빈 dict여야 한다."""
    torch.manual_seed(3)
    policy = build_tiny_policy(num_steps=2)
    policy.eval()
    assert policy.model.judge is None

    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)

    torch.manual_seed(4)
    with torch.no_grad():
        actions_plain = policy.predict_action_chunk(dict(batch))
    torch.manual_seed(4)
    with torch.no_grad():
        actions_decisions, decisions = policy.predict_action_chunk_with_decisions(dict(batch))

    assert decisions == {}
    assert torch.equal(actions_plain, actions_decisions)


def test_predict_action_chunk_with_decisions_calls_prefix_forward_exactly_once(build_tiny_policy, build_tiny_batch):
    """prefix forward(fill_kv_cache=True 호출)가 정확히 한 번만 일어나야 한다 — Judge/Selector가
    별도의 prefix forward를 추가로 돌리면 안 된다(stage 2 요구사항)."""
    policy = build_tiny_policy(use_judge=True, judge_num_failure_types=3, judge_use_progress=True, num_steps=2)
    policy.eval()

    original_forward = policy.model.vlm_with_expert.forward
    call_count = {"fill_kv_cache": 0}

    def _counting_forward(*args, **kwargs):
        if kwargs.get("fill_kv_cache"):
            call_count["fill_kv_cache"] += 1
        return original_forward(*args, **kwargs)

    policy.model.vlm_with_expert.forward = _counting_forward

    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    with torch.no_grad():
        actions, decisions = policy.predict_action_chunk_with_decisions(dict(batch))

    assert call_count["fill_kv_cache"] == 1
    assert torch.isfinite(actions).all()
    assert set(decisions) == {"success_prob", "failure_probs", "progress"}
    assert torch.all((decisions["success_prob"] >= 0) & (decisions["success_prob"] <= 1))


def test_predict_action_chunk_with_decisions_judge_feature_layer_minus_one_reuses_discarded_value(
    build_tiny_policy, build_tiny_batch
):
    """judge_feature_layer=-1(기본)이면 Judge가 prefix 마지막 레이어(sample_actions()가 버리던
    반환값)를 그대로 쓴다 — Bridge Attention 없이도(collect_layer_indices=None) 동작해야 한다."""
    policy = build_tiny_policy(use_judge=True, num_steps=2)
    assert policy.model.judge is not None
    policy.eval()

    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    with torch.no_grad():
        _, decisions = policy.predict_action_chunk_with_decisions(dict(batch))

    assert decisions["success_prob"].shape == (2,)


def test_predict_action_chunk_with_decisions_selector_mask_zeroes_disallowed(build_tiny_policy, build_tiny_batch):
    policy = build_tiny_policy(use_selector=True, selector_num_options=4, num_steps=2)
    policy.eval()

    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    selector_mask = torch.tensor([True, True, False, False])
    with torch.no_grad():
        _, decisions = policy.predict_action_chunk_with_decisions(dict(batch), selector_mask=selector_mask)

    assert torch.allclose(decisions["selector_probs"][:, 2:], torch.zeros(2, 2), atol=1e-7)


def test_cancel_pending_actions_clears_action_queue(build_tiny_policy, build_tiny_batch):
    policy = build_tiny_policy(num_steps=2)
    policy.eval()
    batch = build_tiny_batch(batch_size=1, chunk=8, lang_len=6)

    with torch.no_grad():
        policy.select_action(dict(batch))
    assert len(policy._queues["action"]) > 0

    policy.cancel_pending_actions()
    assert len(policy._queues["action"]) == 0
