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


# --- 8. angle_unwrap_dims: 데이터 전처리 unwrap/rewrap (ARD/손실과 무관) ---
#
# 이전에는 pos_loss(u_t/v_t = flow-matching 속도장)에 직접 각도 래핑을 걸었는데, 틀린
# 설계였다 — u_t/v_t는 액션이 아니라 노이즈-액션의 속도장이라 각도 주기성이 적용되지
# 않고, 정상적인 큰 오차까지 왜곡시킨다(PROGRESS.md 참고). 대신 데이터 전처리 단계
# (processor_smolvla.py의 AngleUnwrapProcessorStep/AngleRewrapProcessorStep)에서 다룬다.


def test_angle_unwrap_dims_rejects_invalid_values():
    with pytest.raises(ValueError, match="angle_unwrap_dims"):
        _minimal_config(angle_unwrap_dims=[])
    with pytest.raises(ValueError, match="angle_unwrap_dims"):
        _minimal_config(angle_unwrap_dims=[3, 3])
    with pytest.raises(ValueError, match="angle_unwrap_dims"):
        _minimal_config(max_action_dim=16, angle_unwrap_dims=[16])  # max_action_dim 범위 밖


def _make_transition(action):
    from lerobot.processor import TransitionKey

    return {TransitionKey.ACTION: action, TransitionKey.OBSERVATION: None, TransitionKey.COMPLEMENTARY_DATA: {}}


def test_angle_unwrap_step_makes_chunk_continuous_across_pi_boundary():
    from lerobot.policies.smolvla.processor_smolvla import AngleUnwrapProcessorStep
    from lerobot.processor import TransitionKey

    step = AngleUnwrapProcessorStep(angle_dims=(0,))
    t = torch.linspace(3.0, 3.6, 5)
    wrapped = ((t + math.pi) % (2 * math.pi)) - math.pi  # ±pi 경계를 넘는 raw 궤적
    raw = torch.zeros(1, 5, 2)
    raw[0, :, 0] = wrapped
    raw[0, :, 1] = torch.arange(5.0)  # angle_dims에 없는 채널은 건드리면 안 됨

    out = step(_make_transition(raw.clone()))[TransitionKey.ACTION]

    assert torch.allclose(out[0, :, 0], t, atol=1e-4), f"연속적으로 펴지지 않음: {out[0, :, 0]}"
    assert torch.equal(out[..., 1], raw[..., 1])  # 지정 안 한 채널은 그대로


def test_angle_rewrap_step_inverts_unwrap():
    from lerobot.policies.smolvla.processor_smolvla import AngleRewrapProcessorStep
    from lerobot.processor import TransitionKey

    step = AngleRewrapProcessorStep(angle_dims=(0, 1))
    continuous = torch.tensor([3.6, -4.0, 0.2])  # 채널 0,1은 pi 밖(unwrap된 공간), 채널 2는 대상 아님
    out = step(_make_transition(continuous.clone()))[TransitionKey.ACTION]

    assert torch.all(out[:2] >= -math.pi) and torch.all(out[:2] < math.pi)
    assert out[2] == continuous[2]  # angle_dims에 없는 채널은 그대로
    # wrap_to_pi(continuous)와 동일해야 함(수동 계산과 비교)
    expected = continuous[:2] - 2 * math.pi * torch.round(continuous[:2] / (2 * math.pi))
    assert torch.allclose(out[:2], expected, atol=1e-6)


def test_angle_unwrap_rewrap_round_trip_within_one_chunk():
    from lerobot.policies.smolvla.processor_smolvla import AngleRewrapProcessorStep, AngleUnwrapProcessorStep
    from lerobot.processor import TransitionKey

    unwrap_step = AngleUnwrapProcessorStep(angle_dims=(0, 1))
    rewrap_step = AngleRewrapProcessorStep(angle_dims=(0, 1))

    torch.manual_seed(0)
    raw = (torch.rand(1, 6, 3) * 2 - 1) * math.pi  # [-pi, pi) 범위의 raw 액션 청크

    unwrapped = unwrap_step(_make_transition(raw.clone()))[TransitionKey.ACTION]
    rewrapped = rewrap_step(_make_transition(unwrapped[0, -1].clone()))[TransitionKey.ACTION]  # 마지막 스텝만 재감싸기

    assert torch.allclose(rewrapped[:2], raw[0, -1, :2], atol=1e-4)
    assert torch.equal(rewrapped[2], raw[0, -1, 2])  # unwrap 대상이 아닌 채널


def test_angle_steps_pass_through_when_action_missing_or_disabled():
    """추론 전처리는 액션이 없는 입력(실제 로봇 관측)도 받는다 — 에러 없이 통과해야 한다.
    angle_dims가 비어 있으면(기본) 둘 다 완전한 identity다."""
    from lerobot.policies.smolvla.processor_smolvla import AngleRewrapProcessorStep, AngleUnwrapProcessorStep
    from lerobot.processor import TransitionKey

    no_action = _make_transition(None)
    assert AngleUnwrapProcessorStep(angle_dims=(0,))(no_action)[TransitionKey.ACTION] is None
    assert AngleRewrapProcessorStep(angle_dims=(0,))(no_action)[TransitionKey.ACTION] is None

    raw = torch.randn(1, 4, 3)
    assert torch.equal(AngleUnwrapProcessorStep()(_make_transition(raw.clone()))[TransitionKey.ACTION], raw)
    single = torch.randn(3)
    assert torch.equal(AngleRewrapProcessorStep()(_make_transition(single.clone()))[TransitionKey.ACTION], single)


def test_angle_unwrap_dims_inserted_into_pipeline_only_when_set(monkeypatch):
    """make_smolvla_pre_post_processors()가 실제로 만드는 파이프라인을 직접 검사한다 —
    angle_unwrap_dims=None(기본)이면 AngleUnwrapProcessorStep/AngleRewrapProcessorStep이
    전혀 없고, 설정하면 각각 정확히 하나씩 들어간다. 토크나이저 로드(HF Hub 접근)만
    몽키패치로 피한다(conftest.py의 AutoConfig/AutoProcessor 패턴과 동일)."""
    import transformers

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.processor_smolvla import (
        AngleRewrapProcessorStep,
        AngleUnwrapProcessorStep,
        make_smolvla_pre_post_processors,
    )

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object()))

    def _build(**overrides):
        config = SmolVLAConfig(
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
            max_state_dim=16, max_action_dim=16, device="cpu", **overrides,
        )
        return make_smolvla_pre_post_processors(config)

    pre_off, post_off = _build()
    assert not any(isinstance(s, AngleUnwrapProcessorStep) for s in pre_off.steps)
    assert not any(isinstance(s, AngleRewrapProcessorStep) for s in post_off.steps)

    pre_on, post_on = _build(angle_unwrap_dims=[3, 4, 5])
    assert sum(isinstance(s, AngleUnwrapProcessorStep) for s in pre_on.steps) == 1
    assert sum(isinstance(s, AngleRewrapProcessorStep) for s in post_on.steps) == 1


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
    # 확인된 RoCo 스키마: 팔당 [ee pose 6 + 그리퍼 1] = 7채널.
    assert config.ard_arm_dim == 7
    assert config.ard_gripper_dim == 1
    assert config.angle_unwrap_dims is None  # 전처리 unwrap은 일단 꺼둠(docs 참고)
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
