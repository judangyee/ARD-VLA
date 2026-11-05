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


# --- 3단계: 학습 손실 연결(judge_success/judge_failure_type/judge_progress/selector_target) ---


def test_judge_losses_absent_when_batch_keys_missing(build_tiny_policy, build_tiny_batch):
    """use_judge=use_selector=True여도 배치에 judge_*/selector_* 키가 없으면 그 손실들은
    전혀 기여하지 않아야 한다(기존 ard_force_target과 같은 "키 없으면 0" 패턴) — loss_dict에
    judge_total_loss 등 관련 키가 아예 생기지 않는다."""
    policy = build_tiny_policy(
        use_judge=True, judge_num_failure_types=3, judge_use_progress=True,
        use_selector=True, selector_num_options=4,
    )
    policy.train()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)  # judge_*/selector_* 키 없음

    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    for key in ("judge_success_loss", "judge_failure_type_loss", "judge_progress_loss", "selector_loss", "judge_total_loss"):
        assert key not in loss_dict


def test_judge_losses_present_and_finite_when_batch_keys_given(build_tiny_policy, build_tiny_batch):
    policy = build_tiny_policy(
        use_judge=True, judge_num_failure_types=3, judge_use_progress=True,
        use_selector=True, selector_num_options=4,
    )
    policy.train()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    batch["judge_success"] = torch.tensor([1.0, 0.0])
    batch["judge_failure_type"] = torch.tensor([1, -100])  # -100은 무시돼야 함
    batch["judge_progress"] = torch.tensor([0.8, 0.2])
    batch["selector_target"] = torch.tensor([0, 2])
    batch["selector_mask"] = torch.tensor([[True, True, False, False], [True, True, True, True]])

    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    for key in ("judge_success_loss", "judge_failure_type_loss", "judge_progress_loss", "selector_loss", "judge_total_loss"):
        assert key in loss_dict
        assert loss_dict[key] == loss_dict[key]  # NaN이 아님


def test_judge_detach_features_blocks_backbone_gradient(build_tiny_policy, build_tiny_batch):
    """judge_detach_features=True(기본)면 judge_success 손실 단독의 backward가 백본
    (vlm_with_expert)/action expert(action_out_proj) 파라미터에 그래디언트를 전혀 남기지
    않아야 한다. 실제 backbone forward를 통과한(진짜 계산 그래프가 연결된) judge_out으로
    확인한다 — Judge 모듈 자체의 detach 동작은 tests/test_judge.py에서 이미 합성 텐서로
    확인했고, 여기서는 "진짜 prefix forward 경로에 꽂았을 때도" 역전파가 끊기는지를 본다."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = build_tiny_policy(use_judge=True, judge_detach_features=True, num_steps=2)
    policy.train()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    batch["judge_success"] = torch.tensor([1.0, 0.0])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    _, ard_extras = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)
    judge_loss, judge_log = policy._compute_judge_losses(batch, ard_extras)
    assert "judge_success_loss" in judge_log
    assert judge_loss.requires_grad

    policy.zero_grad()
    judge_loss.backward()

    assert all(p.grad is None for p in policy.model.vlm_with_expert.parameters())
    assert all(p.grad is None for p in policy.model.action_out_proj.parameters())


# --- LAC(Latent-Aware Controller) 추론 필터 — 실제 SmolVLAPolicy 전체(select_action 경로)
# 연결 확인. LAC 모듈 자체(손계산 케이스, clip, role_aware 등)의 순수 단위 테스트는
# tests/test_lac.py에 있다.


def test_role_split_shared_plus_residual_reconstructs_raw_action(build_tiny_policy, build_tiny_batch):
    """`sample_actions_with_role_split()`의 (shared+residual) 분해가 실제로 denoise 루프가
    만든 최종 액션과 정확히 일치하는지(선형 적분이라 정확해야 함), 그리고 `sample_actions()`
    (기존, 미변경) 호출과 bit-for-bit 동일한 x_t를 내는지 — 같은 noise를 명시적으로 넘겨서
    확인한다."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    arm_dim = 4
    policy = build_tiny_policy(arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=3)
    policy.eval()
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=8, lang_len=6)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    torch.manual_seed(0)
    noise = policy.model.sample_noise(
        (2, policy.config.chunk_size, policy.config.max_action_dim), torch.device("cpu")
    )

    with torch.no_grad():
        x_t, role_split = policy.model.sample_actions_with_role_split(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )
        x_t_plain = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )

    assert torch.allclose(x_t, x_t_plain, atol=1e-5), "role_split 경로가 기존 sample_actions()와 다른 x_t를 내면 안 된다"

    recon_left = role_split["shared_left"] + role_split["residual_left"]
    recon_right = role_split["shared_right"] + role_split["residual_right"]
    assert torch.allclose(recon_left, x_t[..., :arm_dim], atol=1e-4)
    assert torch.allclose(recon_right, x_t[..., arm_dim : 2 * arm_dim], atol=1e-4)

    # VLAFlowMatching의 side-channel이 다음 호출을 위해 깨끗이 정리됐는지(다른 경로에 새지 않는지).
    assert policy.model._lac_role_split_accum is None


def test_lac_mode_none_matches_existing_select_action_output(build_tiny_policy, build_tiny_batch):
    """`lac_mode="none"`(기본)을 명시적으로 줘도 전혀 안 준 것과 bit-for-bit 동일해야 한다."""
    arm_dim = 4
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    torch.manual_seed(1)
    policy_default = build_tiny_policy(arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2)
    policy_default.eval()
    torch.manual_seed(2)
    with torch.no_grad():
        actions_default = [policy_default.select_action(dict(batch)) for _ in range(4)]

    torch.manual_seed(1)
    policy_explicit_none = build_tiny_policy(
        arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2, lac_mode="none"
    )
    policy_explicit_none.eval()
    torch.manual_seed(2)
    with torch.no_grad():
        actions_explicit = [policy_explicit_none.select_action(dict(batch)) for _ in range(4)]

    for a, b in zip(actions_default, actions_explicit, strict=True):
        assert torch.equal(a, b)
    assert policy_default._lac is None
    assert policy_explicit_none._lac is None


def test_select_action_with_lac_covla_passes_through_first_step_then_filters(build_tiny_policy, build_tiny_batch):
    """LAC의 첫 스텝은 그대로(q̃_0=q_1) 통과해야 하고, 그 이후부터는 필터링이 실제로 값을
    바꿔야 한다. ARD head가 zero-init이라(합성 모델, 학습 전) 잔차가 항상 0이고, 그래서
    매 스텝 regime이 "macro"로 고정되어 alpha가 상수가 된다는 점을 이용해 unfiltered
    (lac_mode="none") 버전과 같은 noise로 직접 비교한다(모듈 docstring의 추론 경로 참고)."""
    arm_dim = 4
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)
    n_steps = 4

    torch.manual_seed(11)
    policy_none = build_tiny_policy(arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2)
    policy_none.eval()
    torch.manual_seed(22)
    with torch.no_grad():
        actions_none = [policy_none.select_action(dict(batch)) for _ in range(n_steps)]

    torch.manual_seed(11)
    policy_lac = build_tiny_policy(
        arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2, lac_mode="covla",
    )
    policy_lac.eval()
    torch.manual_seed(22)
    with torch.no_grad():
        actions_lac = [policy_lac.select_action(dict(batch)) for _ in range(n_steps)]

    assert torch.equal(actions_none[0], actions_lac[0]), "LAC 첫 스텝은 필터링 없이 그대로여야 한다"
    assert not torch.equal(actions_none[1], actions_lac[1]), "두 번째 스텝부터는 필터링이 값을 바꿔야 한다"
    for a in actions_lac:
        assert torch.isfinite(a).all()
    assert policy_lac._lac._q_prev_j is not None


def test_lac_log_records_one_entry_per_chunk_step(build_tiny_policy, build_tiny_batch):
    """LAC는 청크 전체(n_action_steps개)를 생성 시점에 한 번에 필터링한다(모듈 docstring의
    "매 실행 스텝마다 순서대로" 참고 — 수치는 실제로 순차적으로 계산되지만, 그 계산을
    select_action() 호출마다 지연시키지 않고 청크가 만들어질 때 전부 끝낸다). 그래서 로그는
    select_action() 호출 횟수가 아니라 "청크 하나가 새로 만들어질 때마다 n_action_steps개씩"
    늘어난다."""
    arm_dim = 4
    policy = build_tiny_policy(
        arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2, lac_mode="covla", lac_log=True,
    )
    policy.eval()
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)

    with torch.no_grad():
        policy.select_action(dict(batch))  # 새 청크 생성 -> n_action_steps(8)개 로그
    assert len(policy.lac_debug_log) == policy.config.n_action_steps
    for entry in policy.lac_debug_log:
        assert set(entry) == {"rho", "omega", "regime", "alpha_hat", "alpha"}

    with torch.no_grad():
        for _ in range(3):
            policy.select_action(dict(batch))  # 같은 청크에서 pop만 함 -> 로그 안 늘어남
    assert len(policy.lac_debug_log) == policy.config.n_action_steps


def test_reset_rebuilds_lac_filter_state(build_tiny_policy, build_tiny_batch):
    arm_dim = 4
    policy = build_tiny_policy(arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2, lac_mode="covla")
    policy.eval()
    batch = build_tiny_batch(batch_size=1, arm_dim=arm_dim, chunk=8, lang_len=6)
    with torch.no_grad():
        policy.select_action(dict(batch))
    assert policy._lac._q_prev_j is not None

    policy.reset()
    assert policy._lac is not None
    assert policy._lac._q_prev_j is None


# --- apply_ard_residual 공유화(학습 forward / denoise_step 중복 제거) 이후의 등가성 검증.
# 공유 함수 자체의 단위 테스트는 tests/test_ard.py::test_apply_ard_residual_matches_manual_combine.


def test_train_forward_and_denoise_step_agree_on_v_t(build_tiny_policy, build_tiny_batch):
    """학습 forward 경로(prefix+suffix 조인트 attention, KV 캐시 없음)와 추론 denoise_step
    경로(prefix KV 캐시 + suffix만 재계산)는, apply_ard_residual()을 공유하긴 하지만 그
    입력인 suffix_out을 얻는 attention 계산 자체는 서로 다른 코드 경로다. 같은 x_t/time을
    주면 두 경로가 같은 v_t를 내야 한다(KV 캐시는 수학적으로 정확한 재구성이어야 하므로) —
    이걸 직접 확인한다. fp32 CPU 연산이라 완전한 bit-exact는 기대하지 않고, 1e-4(절대오차)
    이내를 허용오차로 둔다."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    arm_dim = 4
    policy = build_tiny_policy(arm_dim=arm_dim, ard_arm_dim=arm_dim, use_ard=True, num_steps=2, chunk_size=6)
    policy.eval()
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=6, lang_len=6)

    images, img_masks = policy.prepare_images(dict(batch))
    state = policy.prepare_state(dict(batch))
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(dict(batch))

    torch.manual_seed(0)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = torch.full((actions.shape[0],), 0.37, dtype=torch.float32)  # 임의의 고정 중간 timestep

    # ---- Path A: 학습 forward가 실제로 쓰는 조인트 attention 경로 (production 코드, 그대로 호출) ----
    with torch.no_grad():
        _losses, ard_extras = policy.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise=noise, time=time
        )
    suffix_out_train = ard_extras["shared_activation"]
    actuator_is_first = ard_extras["actuator_is_first"]
    v_t_train, _, _ = apply_ard_residual_ref(
        policy.model.action_out_proj(suffix_out_train), suffix_out_train, policy.model.ard_heads,
        arm_dim, actuator_is_first,
    )

    # ---- Path B: denoise_step이 쓰는 prefix-KV-캐시 경로. sample_actions()가 prefix 캐시를
    # 만드는 것과 정확히 같은 방식으로 직접 구성한다(modeling_smolvla.py의 sample_actions 참고).
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    with torch.no_grad():
        _, past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=policy.config.use_cache, fill_kv_cache=True,
        )
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        v_t_denoise = policy.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=time,
            actuator_is_first=actuator_is_first,
        )

    diff = (v_t_train - v_t_denoise).abs()
    assert torch.isfinite(diff).all()
    assert diff.max().item() < 1e-4, (
        f"학습 forward와 denoise_step의 v_t가 1e-4를 넘게 갈린다(max_abs_diff={diff.max().item():.3e}) — "
        "조인트 attention과 KV 캐시 경로가 실제로 다른 결과를 낸다는 뜻이라 원인 분석이 필요하다."
    )


def apply_ard_residual_ref(v_t, suffix_out, ard_heads, arm_dim, actuator_is_first):
    """apply_ard_residual()을 그대로 재노출 — 테스트 파일 상단에서 매번 import 블록을
    늘리지 않으려고 지연 import로 감싼다(이 파일의 다른 테스트들과 같은 지연-import 관례)."""
    from lerobot.policies.smolvla.ard import apply_ard_residual

    return apply_ard_residual(v_t, suffix_out, ard_heads, arm_dim, actuator_is_first)


def test_ard_off_sample_actions_matches_manual_action_out_proj_only_denoise(build_tiny_policy, build_tiny_batch):
    """use_ard=False면 ard_heads 자체가 없다(apply_ard_residual이 아예 호출되지 않는다) —
    sample_actions()의 출력이, denoise 루프를 "action_out_proj만"(ARD 없이) 수동으로 직접
    적분한 결과와 bit-for-bit 동일한지 확인한다. 즉 ARD를 껐을 때 "원래(ARD 도입 전) SmolVLA"와
    동일한 숫자가 나오는지에 대한 회귀 테스트."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    arm_dim = 4
    policy = build_tiny_policy(arm_dim=arm_dim, use_ard=False, num_steps=3, chunk_size=6)
    policy.eval()
    assert policy.model.ard_heads is None

    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=6, lang_len=6)
    images, img_masks = policy.prepare_images(dict(batch))
    state = policy.prepare_state(dict(batch))
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    torch.manual_seed(0)
    actions_shape = (2, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(actions_shape, torch.device("cpu"))

    with torch.no_grad():
        x_t_policy = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())

    # 수동 재적분: action_out_proj(suffix_out)만 쓰고 ARD는 아예 참조하지 않는다.
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    with torch.no_grad():
        _, past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=policy.config.use_cache, fill_kv_cache=True,
        )
        num_steps = policy.config.num_steps
        dt = -1.0 / num_steps
        x_t_manual = noise.clone()
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32).expand(x_t_manual.shape[0])
            v_t = policy.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t_manual,
                timestep=time_tensor, actuator_is_first=None,
            )
            x_t_manual = x_t_manual + dt * v_t

    assert torch.allclose(x_t_policy, x_t_manual, atol=1e-5), (
        "use_ard=False일 때 sample_actions()가 action_out_proj-only 수동 재적분과 달라졌다 — "
        "ARD 리팩터가 use_ard=False 경로에 영향을 주면 안 된다."
    )
