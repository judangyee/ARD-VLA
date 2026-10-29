"""'설정 가능한 ARD'(configurable ARD) — role_map/action_space/pos_loss_type/per_timestep_norm
+ RoCo Challenge 프리셋 테스트.

대부분 `conftest.py`의 `build_tiny_policy`/`build_tiny_batch`로 작은 합성 SmolVLM 백본 위에서
실제 SmolVLAPolicy 전체(forward/backward/추론)를 돌린다 — Hugging Face Hub 접근 없음. 순수
설정 검증(role_map 파생, mirror_sign 경고)과 수학적 성질(per_timestep_norm 왕복, pos_loss_type
기울기)은 정책 생성 없이 직접 단위 테스트한다.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

import pytest

ROCO_PRESET_PATH = Path(__file__).resolve().parents[1] / "configs" / "presets" / "roco_board.yaml"


def _minimal_config(**overrides):
    """정책을 만들지 않고 SmolVLAConfig 검증 로직만 확인할 때 쓰는 최소 생성자 — Hub 접근이
    전혀 필요 없다(SmolVLAConfig 생성 자체는 VLM을 로드하지 않음)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    kwargs = dict(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        max_state_dim=16,
        max_action_dim=16,
        device="cpu",
    )
    kwargs.update(overrides)
    return SmolVLAConfig(**kwargs)


# --- 1. 기본 config: 새 필드들이 기본값일 때 기존 동작과 bit-exact ---


def test_new_config_fields_default_to_noop(build_tiny_policy, build_tiny_batch):
    """role_map/action_space/pos_loss_type/per_timestep_norm을 전혀 안 주는 것과, 기본값을
    그대로 명시해서 주는 것이 bit-for-bit 동일해야 한다 — 회귀 없음.

    batch는 한 번만 만들어서 재사용하고, forward() 직전에 매번 같은 seed로 리셋한다 —
    forward()가 내부에서 noise/time을 샘플링하므로(sample_noise/sample_time), 두 호출이
    "정확히 같은 지점"에서 RNG를 소비하게 만들어야 한다."""
    torch.manual_seed(5)
    policy_implicit = build_tiny_policy(use_ard=True, num_steps=2)
    policy_implicit.eval()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    torch.manual_seed(6)
    with torch.no_grad():
        loss_implicit, dict_implicit = policy_implicit.forward(dict(batch))

    torch.manual_seed(5)
    policy_explicit = build_tiny_policy(
        use_ard=True, num_steps=2,
        role_map=None, action_space="joint", pos_loss_type="l2", per_timestep_norm=False,
    )
    policy_explicit.eval()
    torch.manual_seed(6)
    with torch.no_grad():
        loss_explicit, dict_explicit = policy_explicit.forward(dict(batch))

    assert torch.equal(loss_implicit, loss_explicit)
    for key in dict_implicit:
        a, b = dict_implicit[key], dict_explicit[key]
        a = a.item() if hasattr(a, "item") else a
        b = b.item() if hasattr(b, "item") else b
        assert a == b, f"{key}: {a!r} != {b!r}"


# --- 2. 레이아웃 변경(7+7=14차원)에서도 forward/backward 정상 ---


def test_asymmetric_layout_7_plus_7_forward_backward(build_tiny_policy, build_tiny_batch):
    """ard_arm_dim=7(RoCo류 14차원 레이아웃)로 바꿔도 AsymmetricResidualHeads 출력 차원이
    자동으로 맞춰져서 forward+backward가 정상 동작해야 한다."""
    arm_dim = 7
    policy = build_tiny_policy(arm_dim=arm_dim, use_ard=True, ard_arm_dim=arm_dim, ard_gripper_dim=1, num_steps=2)
    policy.train()
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=8, lang_len=6)

    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.model.ard_heads.stabilizer_head[-1].out_features == arm_dim
    assert policy.model.ard_heads.actuator_head[-1].out_features == arm_dim


# --- 3. role_map 반전 시 actuator/stabilizer가 반대 팔로 라우팅되는지 ---


def test_role_map_reverses_actuator_is_first():
    from lerobot.policies.smolvla.ard import resolve_actuator_is_first

    cfg_default = _minimal_config(use_ard=True, role_map={"stabilizer": "left", "actuator": "right"})
    cfg_flipped = _minimal_config(use_ard=True, role_map={"stabilizer": "right", "actuator": "left"})

    assert cfg_default.ard_default_actuator_arm == "right"
    assert cfg_flipped.ard_default_actuator_arm == "left"

    is_first_default = resolve_actuator_is_first(cfg_default.ard_default_actuator_arm, batch_size=3, device=torch.device("cpu"))
    is_first_flipped = resolve_actuator_is_first(cfg_flipped.ard_default_actuator_arm, batch_size=3, device=torch.device("cpu"))
    assert bool(is_first_default[0]) != bool(is_first_flipped[0])


def test_role_map_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="role_map"):
        _minimal_config(use_ard=True, role_map={"stabilizer": "left"})  # actuator 키 없음
    with pytest.raises(ValueError, match="role_map"):
        _minimal_config(use_ard=True, role_map={"stabilizer": "left", "actuator": "left"})  # 같은 팔


# --- 4. pos_loss_type: 수학적 성질(L2 기울기 소실) + 실제 wiring 둘 다 확인 ---


def test_l1_gradient_does_not_vanish_near_target_unlike_l2():
    """목표에 아주 가까운 지점에서 L2의 기울기는 거의 0으로 사라지지만, L1의 기울기는
    크기가 1로 일정하다 — pos_loss_type="l1"을 추가한 이유(arXiv 2603.15469)의 수학적 근거."""
    pred = torch.tensor([1.0001], requires_grad=True)
    target = torch.tensor([1.0])

    l2 = F.mse_loss(pred, target, reduction="none")
    l2.backward()
    l2_grad = pred.grad.clone().item()
    pred.grad = None

    l1 = F.l1_loss(pred, target, reduction="none")
    l1.backward()
    l1_grad = pred.grad.clone().item()

    assert abs(l2_grad) < 1e-3, f"L2 기울기가 거의 0이어야 한다: {l2_grad}"
    assert abs(abs(l1_grad) - 1.0) < 1e-6, f"L1 기울기는 크기 1이어야 한다: {l1_grad}"


def test_pos_loss_type_dispatches_to_matching_torch_loss_fn(build_tiny_policy, build_tiny_batch, monkeypatch):
    """configuration_smolvla.py의 pos_loss_type이 modeling_smolvla.py에서 실제로 올바른
    torch.nn.functional.*_loss로 분기되는지, 함수 호출을 가로채서 직접 확인한다."""
    import lerobot.policies.smolvla.modeling_smolvla as mod

    calls = []
    for name, fn in (("l2", mod.F.mse_loss), ("l1", mod.F.l1_loss), ("smooth_l1", mod.F.smooth_l1_loss)):
        def _wrap(name=name, fn=fn):
            def _inner(*a, **k):
                calls.append(name)
                return fn(*a, **k)
            return _inner
        monkeypatch.setattr(mod.F, {"l2": "mse_loss", "l1": "l1_loss", "smooth_l1": "smooth_l1_loss"}[name], _wrap())

    for pos_loss_type, expected in (("l2", "l2"), ("l1", "l1"), ("smooth_l1", "smooth_l1")):
        calls.clear()
        policy = build_tiny_policy(use_ard=True, num_steps=2, pos_loss_type=pos_loss_type)
        policy.train()
        batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
        policy.forward(batch)
        assert expected in calls, f"pos_loss_type={pos_loss_type!r}인데 {expected!r}가 호출되지 않음(호출된 것: {calls})"


def test_pos_loss_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="pos_loss_type"):
        _minimal_config(pos_loss_type="huber")


# --- 5. per_timestep_norm: 수학적 왕복 오차 + on/off 동작 ---


def test_timestep_norm_round_trip_scalar():
    from lerobot.policies.smolvla.ard import apply_timestep_norm, compute_timestep_norm_sigma, invert_timestep_norm

    actions = torch.randn(3, 10, 5)
    sigma = compute_timestep_norm_sigma(a=2.0, s=0.5, chunk_size=10, device=actions.device, dtype=actions.dtype)
    scaled = apply_timestep_norm(actions, sigma)
    restored = invert_timestep_norm(scaled, sigma)
    assert torch.allclose(actions, restored, atol=1e-6)
    assert not torch.allclose(actions, scaled)  # 실제로 스케일이 걸렸는지


def test_timestep_norm_round_trip_per_dim_with_padding():
    from lerobot.policies.smolvla.ard import apply_timestep_norm, compute_timestep_norm_sigma, invert_timestep_norm

    actions = torch.randn(2, 6, 4)  # action_dim=4인데 실제 피팅은 2채널만 됐다고 가정(패딩 2채널)
    sigma = compute_timestep_norm_sigma(
        a=[1.5, 2.0], s=[0.1, 0.4], chunk_size=6, device=actions.device, dtype=actions.dtype, pad_to=4
    )
    assert sigma.shape == (6, 4)
    assert torch.equal(sigma[:, 2:], torch.ones(6, 2))  # 패딩 채널은 identity(a=1,s=0)

    scaled = apply_timestep_norm(actions, sigma)
    restored = invert_timestep_norm(scaled, sigma)
    assert torch.allclose(actions, restored, atol=1e-6)
    # 패딩 채널(identity)은 스케일링으로 전혀 안 바뀌어야 한다.
    assert torch.equal(actions[..., 2:], scaled[..., 2:])


def test_per_timestep_norm_off_matches_baseline_on_changes_but_stays_finite(build_tiny_policy, build_tiny_batch):
    torch.manual_seed(7)
    policy_off = build_tiny_policy(use_ard=True, num_steps=2, per_timestep_norm=False)
    policy_off.eval()
    batch = build_tiny_batch(batch_size=2, chunk=8, lang_len=6)
    torch.manual_seed(8)
    with torch.no_grad():
        loss_off, _ = policy_off.forward(dict(batch))

    torch.manual_seed(7)
    policy_on = build_tiny_policy(
        use_ard=True, num_steps=2, per_timestep_norm=True, per_timestep_norm_a=1.0, per_timestep_norm_s=0.3,
    )
    policy_on.eval()
    torch.manual_seed(8)
    with torch.no_grad():
        loss_on, _ = policy_on.forward(dict(batch))

    assert torch.isfinite(loss_off) and torch.isfinite(loss_on)
    assert not torch.equal(loss_off, loss_on), "per_timestep_norm이 실제로 손실을 바꿔야 한다"


def test_per_timestep_norm_inference_path_finite(build_tiny_policy, build_tiny_batch):
    policy = build_tiny_policy(use_ard=True, num_steps=2, per_timestep_norm=True, per_timestep_norm_s=0.2)
    policy.eval()
    batch = build_tiny_batch(batch_size=1, chunk=8, lang_len=6)
    with torch.no_grad():
        actions = policy.predict_action_chunk(dict(batch))
    assert torch.isfinite(actions).all()
    assert actions.shape == (1, policy.config.chunk_size, policy.config.action_feature.shape[0])


def test_per_timestep_norm_a_must_be_positive():
    with pytest.raises(ValueError, match="per_timestep_norm_a"):
        _minimal_config(per_timestep_norm=True, per_timestep_norm_a=0.0)


# --- 6. action_space="ee" -> lac_mirror_sign 강제 비활성 + 경고 ---


def test_action_space_ee_forces_mirror_sign_none_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        config = _minimal_config(
            use_ard=True, ard_arm_dim=7, ard_gripper_dim=0, lac_mode="covla",
            action_space="ee", lac_mirror_sign=[1.0] * 7,
        )
    assert config.lac_mirror_sign is None
    assert any("action_space" in r.message for r in caplog.records)


def test_action_space_joint_keeps_mirror_sign():
    config = _minimal_config(
        use_ard=True, ard_arm_dim=7, ard_gripper_dim=0, lac_mode="covla",
        action_space="joint", lac_mirror_sign=[1.0] * 7,
    )
    assert config.lac_mirror_sign == [1.0] * 7


def test_action_space_rejects_unknown_value():
    with pytest.raises(ValueError, match="action_space"):
        _minimal_config(action_space="cartesian")


# --- 8. ard_angle_dims: RoCo rx/ry/rz 각도 래핑 ---


def test_wrap_to_pi_removes_2pi_boundary_discontinuity():
    from lerobot.policies.smolvla.ard import wrap_to_pi

    # 3.14 vs -3.14: 물리적으로는 거의 같은 각도인데 raw 차이는 ~2*pi에 가깝다.
    diff = torch.tensor([3.14 - (-3.14), 0.1, -0.1, 2 * math.pi, -2 * math.pi])
    wrapped = wrap_to_pi(diff)
    assert wrapped[0].abs() < 0.01, f"경계 불연속이 감싸지지 않음: {wrapped[0]}"
    assert torch.allclose(wrapped[1:3], diff[1:3])  # 경계 멀리 있는 값은 그대로
    assert torch.allclose(wrapped[3:], torch.zeros(2), atol=1e-5)  # 정확히 2*pi 배수는 0


def test_build_angle_channel_mask_applies_to_both_arm_blocks_and_skips_padding():
    from lerobot.policies.smolvla.ard import build_angle_channel_mask

    mask = build_angle_channel_mask(angle_dims=[3, 4, 5], arm_dim=7, total_dim=16, device=torch.device("cpu"))
    expected = torch.zeros(16, dtype=torch.bool)
    expected[[3, 4, 5, 10, 11, 12]] = True  # 왼팔 블록[0:7] + 오른팔 블록[7:14], 패딩[14:16]은 제외
    assert torch.equal(mask, expected)


def test_apply_angle_aware_target_wraps_only_masked_channels():
    from lerobot.policies.smolvla.ard import apply_angle_aware_target

    u_t = torch.tensor([[3.14, 5.0]])
    v_t = torch.tensor([[-3.14, 0.0]])  # 채널0: 경계 근처(거의 같은 각도), 채널1: 진짜 큰 차이
    mask = torch.tensor([True, False])

    v_t_for_loss = apply_angle_aware_target(u_t, v_t, mask)
    diff0 = (u_t[0, 0] - v_t_for_loss[0, 0]).item()
    assert abs(diff0) < 0.01, f"마스크된 채널은 wrap돼야 한다: {diff0}"
    assert torch.equal(v_t_for_loss[:, 1], v_t[:, 1])  # 마스크 안 된 채널은 그대로


def test_apply_angle_aware_target_is_identity_when_mask_all_false():
    from lerobot.policies.smolvla.ard import apply_angle_aware_target

    u_t = torch.randn(2, 4)
    v_t = torch.randn(2, 4)
    mask = torch.zeros(4, dtype=torch.bool)
    # u_t - (u_t - v_t)가 부동소수점상 v_t와 완전히 bit-exact하지 않을 수 있어(비결합성),
    # 실제 modeling_smolvla.py는 ard_angle_dims가 없으면 이 함수를 호출조차 안 한다(아래
    # test_pos_loss_type_dispatches_to_matching_torch_loss_fn과 같은 패턴) — 여기서는 수학적으로
    # "거의 동일"함만 확인한다.
    assert torch.allclose(apply_angle_aware_target(u_t, v_t, mask), v_t, atol=1e-6)


def test_ard_angle_dims_rejects_invalid_values():
    with pytest.raises(ValueError, match="ard_angle_dims"):
        _minimal_config(use_ard=True, ard_angle_dims=[])
    with pytest.raises(ValueError, match="ard_angle_dims"):
        _minimal_config(use_ard=True, ard_angle_dims=[3, 3])
    with pytest.raises(ValueError, match="ard_angle_dims"):
        _minimal_config(use_ard=True, ard_arm_dim=7, ard_angle_dims=[7])  # arm_dim 범위 밖
    with pytest.raises(ValueError, match="ard_angle_dims"):
        _minimal_config(use_ard=False, ard_angle_dims=[3])  # use_ard=False인데 지정


def test_ard_angle_dims_default_skips_wrapping_call(build_tiny_policy, build_tiny_batch, monkeypatch):
    """ard_angle_dims=None(기본)이면 modeling_smolvla.py가 apply_angle_aware_target을 전혀
    호출하지 않아야 한다(호출 자체가 없어야 bit-exact가 보장됨)."""
    import lerobot.policies.smolvla.modeling_smolvla as mod

    calls = []
    original = mod.apply_angle_aware_target

    def _counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(mod, "apply_angle_aware_target", _counting)

    policy = build_tiny_policy(arm_dim=7, use_ard=True, ard_arm_dim=7, ard_gripper_dim=1, num_steps=2)
    policy.train()
    batch = build_tiny_batch(batch_size=2, arm_dim=7, chunk=8, lang_len=6)
    policy.forward(batch)
    assert calls == [], "ard_angle_dims=None인데 apply_angle_aware_target이 호출됐다"


def test_ard_angle_dims_set_calls_wrapping_and_stays_finite(build_tiny_policy, build_tiny_batch, monkeypatch):
    import lerobot.policies.smolvla.modeling_smolvla as mod

    calls = []
    original = mod.apply_angle_aware_target

    def _counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(mod, "apply_angle_aware_target", _counting)

    arm_dim = 7
    policy = build_tiny_policy(
        arm_dim=arm_dim, use_ard=True, ard_arm_dim=arm_dim, ard_gripper_dim=1,
        ard_angle_dims=[3, 4, 5], num_steps=2,
    )
    policy.train()
    batch = build_tiny_batch(batch_size=2, arm_dim=arm_dim, chunk=8, lang_len=6)
    loss, _ = policy.forward(batch)

    assert calls == [1]
    assert torch.isfinite(loss)
    loss.backward()


# --- 7. RoCo Challenge 프리셋 ---


def test_roco_preset_loads_into_valid_config():
    overrides = yaml.safe_load(ROCO_PRESET_PATH.read_text())
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(44,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        max_state_dim=44,
        max_action_dim=14,
        device="cpu",
        **overrides,
    )
    config.validate_features()

    assert config.ard_default_actuator_arm == "left"  # role_map={stabilizer: right, actuator: left}
    assert config.action_space == "ee"
    assert config.pos_loss_type == "l1"
    assert config.chunk_size == 20
    assert config.n_action_steps == 5
    assert config.per_timestep_norm is True
    assert config.freeze_vision_encoder is True
    assert config.lac_mode == "none"
    # 확인된 RoCo 스키마: 팔당 [ee pose 6 + 그리퍼 1] = 7채널, rx/ry/rz는 상대 인덱스 3,4,5.
    assert config.ard_arm_dim == 7
    assert config.ard_gripper_dim == 1
    assert config.ard_angle_dims == [3, 4, 5]
    assert 2 * config.ard_arm_dim == config.action_feature.shape[0] == 14


def test_roco_preset_tiny_model_trains_and_infers(build_tiny_policy, build_tiny_batch):
    """RoCo 프리셋(14차원=7+7)을 작은 합성 모델에 그대로 적용해 몇 스텝 학습/추론이 정상
    동작하는지 확인한다 — 실제 RoCo 데이터셋으로는 이 샌드박스에서 돌릴 수 없다(HF Hub
    차단, docs/configurable_ard.md의 "미확인" 참고)."""
    overrides = yaml.safe_load(ROCO_PRESET_PATH.read_text())
    policy = build_tiny_policy(arm_dim=7, num_steps=2, **overrides)
    policy.train()
    batch = build_tiny_batch(batch_size=2, arm_dim=7, chunk=overrides["chunk_size"], lang_len=6)

    loss, loss_dict = policy.forward(dict(batch))
    assert torch.isfinite(loss)
    loss.backward()
    assert loss_dict["ard_mode"] == "asymmetric"

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(dict(batch))
    assert torch.isfinite(actions).all()
    assert actions.shape == (2, policy.config.chunk_size, policy.config.action_feature.shape[0])
