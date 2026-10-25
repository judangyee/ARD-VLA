"""`lerobot.policies.smolvla.lac`(Co-VLA LAC 필터)에 대한 오프라인 단위 테스트 —
`tests/test_ard.py`와 같은 스타일로 합성 텐서를 직접 넣어 검증한다. 실제 SmolVLAPolicy
전체(prefix forward/denoise 루프와의 연결, lac_mode="none" bit-exact)를 확인하는 통합
테스트는 `tests/test_ard_integration.py`에 있다.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.smolvla.lac import LACConfig, LatentAwareController


def _step(lac, shared, res, q_scale=1.0, stabilizer_is_left=True):
    shared_L = torch.tensor([shared])
    shared_R = torch.tensor([shared])
    res_L = torch.tensor([res[0]])
    res_R = torch.tensor([res[1]])
    q_t = torch.cat([torch.tensor([shared]) * q_scale, torch.tensor([shared]) * q_scale], dim=-1)
    return lac.step(shared_L, shared_R, res_L, res_R, q_t, stabilizer_is_left=stabilizer_is_left)


# --- 손계산 케이스 (2D 관절, 공유 둘 다 [1,0]) ---


def test_case_a_opposite_twist_is_precision_regime():
    lac = LatentAwareController(LACConfig(), joint_dim=2)
    _, debug = _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.0, -0.1]))
    assert debug["regime"] == ["precision"]
    assert debug["omega"].item() == pytest.approx(1.0, abs=1e-3)
    assert debug["alpha_hat"].item() == pytest.approx(0.7, abs=1e-6)


def test_case_b_jitter_is_noise_regime():
    lac = LatentAwareController(LACConfig(), joint_dim=2)
    _, debug = _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.1, 0.0]))
    assert debug["regime"] == ["noise"]
    assert debug["omega"].item() == pytest.approx(0.0, abs=1e-4)
    assert debug["alpha_hat"].item() == pytest.approx(0.3, abs=1e-6)


def test_case_c_coordinated_motion_is_macro_regime():
    lac = LatentAwareController(LACConfig(), joint_dim=2)
    _, debug = _step(lac, [1.0, 0.0], ([0.005, 0.0], [0.005, 0.0]))
    assert debug["regime"] == ["macro"]
    assert debug["rho"].item() == pytest.approx(0.005, abs=1e-6)
    assert debug["alpha_hat"].item() == pytest.approx(0.8, abs=1e-6)


# --- clip / beta 평활화 / reset ---


def test_alpha_hat_is_clipped_to_alpha_min_max():
    cfg = LACConfig(alpha_base=0.9, delta_macro=0.9, alpha_max=0.95)  # 0.9+0.9=1.8 -> clip 0.95
    lac = LatentAwareController(cfg, joint_dim=2)
    _, debug = _step(lac, [1.0, 0.0], ([0.001, 0.0], [0.001, 0.0]))  # macro 유도(rho 작음)
    assert debug["alpha_hat"].item() == pytest.approx(0.95, abs=1e-6)

    cfg2 = LACConfig(alpha_base=0.05, delta_noise=0.5, alpha_min=0.05)  # 0.05-0.5=-0.45 -> clip 0.05
    lac2 = LatentAwareController(cfg2, joint_dim=2)
    _, debug2 = _step(lac2, [1.0, 0.0], ([0.0, 0.1], [0.1, 0.0]))  # noise 유도
    assert debug2["alpha_hat"].item() == pytest.approx(0.05, abs=1e-6)


def test_beta_smoothing_blends_previous_alpha():
    """두 번째 스텝부터는 alpha_t = (1-beta)*alpha_{t-1} + beta*alpha_hat이어야 한다."""
    cfg = LACConfig(beta=0.2)
    lac = LatentAwareController(cfg, joint_dim=2)
    # 1스텝: macro(alpha_hat=0.8) -> alpha_0 = alpha_hat(히스토리 없음, 그대로)
    _, debug0 = _step(lac, [1.0, 0.0], ([0.005, 0.0], [0.005, 0.0]))
    assert debug0["alpha"].item() == pytest.approx(0.8, abs=1e-6)

    # 2스텝: noise(alpha_hat=0.3) -> alpha_1 = 0.8*0.8 + 0.2*0.3 = 0.70
    _, debug1 = _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.1, 0.0]))
    expected = (1 - cfg.beta) * 0.8 + cfg.beta * 0.3
    assert debug1["alpha"].item() == pytest.approx(expected, abs=1e-6)


def test_first_step_output_equals_raw_input():
    lac = LatentAwareController(LACConfig(), joint_dim=2)
    q_t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    q_tilde, _ = lac.step(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 0.1]]), torch.tensor([[0.0, -0.1]]), q_t,
    )
    assert torch.equal(q_tilde, q_t)


def test_reset_clears_state_so_next_step_is_first_again():
    lac = LatentAwareController(LACConfig(), joint_dim=2)
    _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.0, -0.1]), q_scale=5.0)
    assert lac._q_prev_j is not None

    lac.reset()
    assert lac._q_prev_j is None
    assert lac._alpha_prev is None

    q_t = torch.tensor([[9.0, 9.0, 9.0, 9.0]])
    q_tilde, _ = lac.step(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 0.1]]), torch.tensor([[0.0, -0.1]]), q_t,
    )
    assert torch.equal(q_tilde, q_t)  # reset 후 다시 "첫 스텝"으로 취급됨


# --- 그리퍼 pass-through ---


def test_gripper_dimension_passes_through_unfiltered():
    """그리퍼 채널은 ρ/ω 계산에서 제외되고, 필터링도 전혀 거치지 않아야 한다(입력 그대로)."""
    lac = LatentAwareController(LACConfig(), joint_dim=2, gripper_dim=1)
    # 1스텝으로 상태를 만든 뒤, 2스텝에서 그리퍼 값이 필터링 없이 그대로 나오는지 확인.
    arm = lambda joint, gripper: torch.tensor([[*joint, gripper]])  # noqa: E731
    shared = arm([1.0, 0.0], 0.0)
    res1 = arm([0.0, 0.1], 0.0)
    q1 = torch.cat([arm([1.0, 0.0], 0.1), arm([1.0, 0.0], -0.1)], dim=-1)
    lac.step(shared, shared, res1, -res1, q1)

    res2 = arm([0.1, 0.0], 0.0)  # noise 유도
    q2 = torch.cat([arm([2.0, 2.0], 0.7), arm([2.0, 2.0], -0.7)], dim=-1)
    q_tilde2, _ = lac.step(shared, shared, res2, arm([0.0, 0.1], 0.0), q2)

    assert q_tilde2[0, 2].item() == pytest.approx(0.7, abs=1e-6)  # left gripper
    assert q_tilde2[0, 5].item() == pytest.approx(-0.7, abs=1e-6)  # right gripper


# --- role_aware ---


def test_role_aware_stabilizer_alpha_never_exceeds_cap():
    cfg = LACConfig(role_aware=True, alpha_stab_max=0.5, alpha_base=0.4, delta_macro=0.4)
    lac = LatentAwareController(cfg, joint_dim=2)
    # macro 유도(alpha_hat_shared = 0.8) -> stabilizer는 0.5로 clip돼야 함.
    _, debug = _step(lac, [1.0, 0.0], ([0.005, 0.0], [0.005, 0.0]), stabilizer_is_left=True)
    assert debug["alpha_hat_stab"].item() <= cfg.alpha_stab_max + 1e-9
    assert debug["alpha_hat_stab"].item() == pytest.approx(0.5, abs=1e-6)


def test_role_aware_actuator_alpha_high_when_cooperative():
    cfg = LACConfig(role_aware=True, alpha_act_cooperative=0.9, tau_omega=0.3)
    lac = LatentAwareController(cfg, joint_dim=2)
    # 왼팔=Stabilizer, 오른팔=Actuator. 반대 비틀기(협동) -> Actuator alpha_hat = 0.9.
    _, debug = _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.0, -0.1]), stabilizer_is_left=True)
    assert debug["alpha_hat_act"].item() == pytest.approx(0.9, abs=1e-6)


def test_role_aware_actuator_alpha_lowered_only_on_noise():
    cfg = LACConfig(role_aware=True, alpha_act_cooperative=0.9, delta_noise=0.1, alpha_base=0.4)
    lac = LatentAwareController(cfg, joint_dim=2)
    # 떨림(noise) 유도 -> Actuator도 covla와 동일하게 alpha_base - delta_noise = 0.3으로 낮아짐.
    _, debug = _step(lac, [1.0, 0.0], ([0.0, 0.1], [0.1, 0.0]), stabilizer_is_left=True)
    assert debug["alpha_hat_act"].item() == pytest.approx(0.3, abs=1e-6)


def test_role_aware_respects_stabilizer_is_left_flag():
    """stabilizer_is_left=False면 오른팔이 Stabilizer가 되어 그 쪽 alpha가 상한에 걸려야 한다."""
    cfg = LACConfig(role_aware=True, alpha_stab_max=0.5)
    lac = LatentAwareController(cfg, joint_dim=2)
    q_t = torch.cat([torch.tensor([[10.0, 0.0]]), torch.tensor([[-10.0, 0.0]])], dim=-1)
    shared = torch.tensor([[1.0, 0.0]])
    q_tilde, debug = lac.step(
        shared, shared, torch.tensor([[0.005, 0.0]]), torch.tensor([[0.005, 0.0]]), q_t,
        stabilizer_is_left=False,
    )
    # 첫 스텝이라 필터링 없이 그대로 나오는 것만 확인(상태 세팅), alpha_hat_stab만 체크.
    assert debug["alpha_hat_stab"].item() <= cfg.alpha_stab_max + 1e-9


# --- mirror_sign ---


def test_mirror_sign_flips_omega_sign():
    """부호가 반대인 관절(미러링)을 보정하지 않으면 ω 계산이 틀린다 — mirror_sign=[-1]로
    보정한 경우와 보정하지 않은([1]) 경우의 ω 부호가 반대가 돼야 한다."""
    shared = torch.tensor([[1.0]])
    res_L = torch.tensor([[0.1]])
    res_R = torch.tensor([[0.1]])  # 미러링된 관절이라 왼팔과 "같은 부호"로 기록됨(실제로는 반대 동작)
    q_t = torch.cat([shared, shared], dim=-1)

    lac_no_mirror = LatentAwareController(LACConfig(mirror_sign=[1.0]), joint_dim=1)
    _, debug_no_mirror = lac_no_mirror.step(shared, shared, res_L, res_R, q_t)

    lac_mirrored = LatentAwareController(LACConfig(mirror_sign=[-1.0]), joint_dim=1)
    _, debug_mirrored = lac_mirrored.step(shared, shared, res_L, res_R, q_t)

    assert debug_no_mirror["omega"].item() == pytest.approx(-debug_mirrored["omega"].item(), abs=1e-6)


def test_mirror_sign_length_mismatch_raises():
    with pytest.raises(ValueError, match="mirror_sign"):
        LatentAwareController(LACConfig(mirror_sign=[1.0, 1.0]), joint_dim=3)


# --- SmolVLAConfig의 lac_mode 검증 (실제 SmolVLAPolicy 생성 없이 config만) ---


def _base_kwargs(**overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    kwargs = dict(
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )
    kwargs.update(overrides)
    return kwargs


def test_config_lac_mode_invalid_raises():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    with pytest.raises(ValueError, match="lac_mode"):
        SmolVLAConfig(**_base_kwargs(lac_mode="bogus"))


def test_config_lac_requires_use_ard():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    with pytest.raises(ValueError, match="use_ard"):
        SmolVLAConfig(**_base_kwargs(use_ard=False, lac_mode="covla"))


def test_config_lac_rejects_rtc_combo():
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    with pytest.raises(ValueError, match="RTC"):
        SmolVLAConfig(**_base_kwargs(use_ard=True, lac_mode="covla", rtc_config=RTCConfig(enabled=True)))


def test_config_lac_mirror_sign_length_validation():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    with pytest.raises(ValueError, match="lac_mirror_sign"):
        SmolVLAConfig(
            **_base_kwargs(use_ard=True, lac_mode="covla", ard_arm_dim=8, ard_gripper_dim=1, lac_mirror_sign=[1.0, 1.0])
        )

    # 길이가 맞으면(7관절) 에러 없음.
    SmolVLAConfig(
        **_base_kwargs(
            use_ard=True, lac_mode="covla", ard_arm_dim=8, ard_gripper_dim=1,
            lac_mirror_sign=[-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
        )
    )
