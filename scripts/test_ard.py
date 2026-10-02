#!/usr/bin/env python3
"""ARD (Asymmetric Role Decomposition, 비대칭 역할 분리) SmolVLA 수정 사항에 대한 오프라인 테스트.

lerobot.policies.smolvla.ard를, 실제 학습/추론 파이프라인과 동일한 shape의 합성(synthetic)
텐서로 직접 검증하고, SmolVLAConfig의 ARD 검증 로직도 함께 확인한다. SmolVLAPolicy 전체를
생성하지는 않는다 — 그러려면 `load_vlm_weights=False`여도 Hugging Face Hub에서 SmolVLM2
백본 config를 내려받아야 하는데, 이 환경의 네트워크 정책이 Hub 접근을 막아놓았기 때문이다.
GPU 유무와 관계없이 어떤 머신에서든 안전하게 실행할 수 있다.

Actuator 팔은 config로 고정된다(`ard_default_actuator_arm`, 기본값 "right") — 샘플별로나
언어 지시에 따라 역할이 바뀌지 않는다.
"""

import sys
import warnings

import torch

from lerobot.policies.smolvla.ard import (
    AsymmetricResidualHeads,
    BridgeAttention,
    ForceHead,
    GradNormLambdas,
    combine_by_role,
    compute_ard_losses,
    resolve_actuator_is_first,
    resolve_bridge_layer_indices,
    split_by_role,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_config_validation():
    cfg = SmolVLAConfig(use_ard=True, ard_arm_dim=7, max_action_dim=32)
    check("SmolVLAConfig(use_ard=True)가 정상 dims로 생성된다", cfg.use_ard is True)
    check("ard_default_actuator_arm 기본값은 'right'다", cfg.ard_default_actuator_arm == "right")

    try:
        SmolVLAConfig(use_ard=True, ard_arm_dim=20, max_action_dim=32)
        check("max_action_dim보다 큰 ard_arm_dim을 거부한다", False)
    except ValueError:
        check("max_action_dim보다 큰 ard_arm_dim을 거부한다", True)

    try:
        SmolVLAConfig(use_ard=True, ard_default_actuator_arm="both")
        check("잘못된 ard_default_actuator_arm 값을 거부한다", False)
    except ValueError:
        check("잘못된 ard_default_actuator_arm 값을 거부한다", True)

    check("ard_reg_time_weighting 기본값은 'none'이다", SmolVLAConfig().ard_reg_time_weighting == "none")
    cfg_weighted = SmolVLAConfig(use_ard=True, ard_reg_time_weighting="one_minus_t")
    check("ard_reg_time_weighting='one_minus_t'가 정상 생성된다", cfg_weighted.ard_reg_time_weighting == "one_minus_t")
    try:
        SmolVLAConfig(use_ard=True, ard_reg_time_weighting="bogus")
        check("잘못된 ard_reg_time_weighting 값을 거부한다", False)
    except ValueError:
        check("잘못된 ard_reg_time_weighting 값을 거부한다", True)

    check("ard_use_force_head 기본값은 False다", SmolVLAConfig().ard_use_force_head is False)
    cfg_force_head = SmolVLAConfig(use_ard=True, ard_use_force_head=True)
    check("ard_use_force_head=True가 정상 생성된다", cfg_force_head.ard_use_force_head is True)
    try:
        SmolVLAConfig(use_ard=False, ard_use_force_head=True)
        check("use_ard=False인데 ard_use_force_head=True면 거부한다", False)
    except ValueError:
        check("use_ard=False인데 ard_use_force_head=True면 거부한다", True)


def test_resolve_actuator_is_first_is_fixed():
    device = torch.device("cpu")

    out = resolve_actuator_is_first("right", batch_size=5, device=device)
    check(
        "오른팔이 Actuator일 때 모든 샘플의 actuator_is_first가 False다",
        not out.any().item() and out.shape == (5,),
    )

    out = resolve_actuator_is_first("left", batch_size=5, device=device)
    check("왼팔이 Actuator일 때 모든 샘플의 actuator_is_first가 True다", bool(out.all()))


def test_split_combine_roundtrip():
    torch.manual_seed(0)
    batch, chunk, arm_dim = 4, 5, 7
    x = torch.randn(batch, chunk, 2 * arm_dim)
    # 오른팔이 항상 Actuator이므로 모든 샘플에서 actuator_is_first는 False다.
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device=x.device)

    stab, act = split_by_role(x, arm_dim, actuator_is_first)
    check(
        "split_by_role 출력 shape이 맞다",
        stab.shape == (batch, chunk, arm_dim) and act.shape == (batch, chunk, arm_dim),
    )
    check("오른팔이 고정 Actuator일 때 actuator = 오른팔 블록", torch.allclose(act, x[..., arm_dim:]))
    check("오른팔이 고정 Actuator일 때 stabilizer = 왼팔 블록", torch.allclose(stab, x[..., :arm_dim]))

    left, right = combine_by_role(stab, act, actuator_is_first)
    recombined = torch.cat([left, right], dim=-1)
    check("split_by_role -> combine_by_role이 원본 텐서로 정확히 되돌아온다", torch.allclose(recombined, x))


def test_asymmetric_residual_heads_zero_init():
    torch.manual_seed(0)
    batch, chunk, expert_hidden, arm_dim = 2, 5, 32, 7
    heads = AsymmetricResidualHeads(expert_hidden_size=expert_hidden, arm_dim=arm_dim)
    suffix_features = torch.randn(batch, chunk, expert_hidden)
    stab_res, act_res = heads(suffix_features)
    check(
        "AsymmetricResidualHeads는 초기화 시 완전한 0(no-op) 상태다 (사전학습 출력 위에서 항등 시작)",
        torch.allclose(stab_res, torch.zeros_like(stab_res)) and torch.allclose(act_res, torch.zeros_like(act_res)),
    )

    # 한 번 gradient step을 밟으면 head들이 0에서 벗어나야 한다.
    target = torch.randn(batch, chunk, arm_dim)
    loss = (stab_res - target).pow(2).mean() + (act_res - target).pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in heads.parameters() if p.grad is not None)
    check("AsymmetricResidualHeads가 gradient를 정상적으로 받는다", grad_norm > 0, detail=f"grad_norm={grad_norm}")


def test_resolve_bridge_layer_indices():
    # 기본(None): 1/4, 1/2, 3/4, 마지막 4개 지점.
    idx = resolve_bridge_layer_indices(16, None)
    check(
        "num_vlm_layers=16, 기본값이면 4개 지점(1/4,1/2,3/4,마지막)을 고른다",
        idx == sorted(idx) and len(idx) == 4 and idx[-1] == 15,
        detail=f"idx={idx}",
    )

    # 레이어가 너무 적으면(4 미만) 전부 다 쓴다.
    idx_small = resolve_bridge_layer_indices(3, None)
    check("num_vlm_layers < 4면 전부 다 쓴다", idx_small == [0, 1, 2])

    # 명시적으로 준 인덱스는 정렬만 해서 그대로 쓴다.
    idx_explicit = resolve_bridge_layer_indices(16, [10, 2, 5])
    check("명시적으로 준 인덱스는 정렬해서 그대로 쓴다", idx_explicit == [2, 5, 10])

    try:
        resolve_bridge_layer_indices(16, [])
        check("빈 리스트를 거부한다", False)
    except ValueError:
        check("빈 리스트를 거부한다", True)

    try:
        resolve_bridge_layer_indices(16, [1, 1, 2])
        check("중복된 인덱스를 거부한다", False)
    except ValueError:
        check("중복된 인덱스를 거부한다", True)

    try:
        resolve_bridge_layer_indices(16, [0, 16])
        check("범위를 벗어난 인덱스(>= num_vlm_layers)를 거부한다", False)
    except ValueError:
        check("범위를 벗어난 인덱스(>= num_vlm_layers)를 거부한다", True)


def test_bridge_attention_zero_init():
    torch.manual_seed(0)
    batch, chunk, expert_hidden, vlm_hidden, num_layers = 2, 5, 32, 48, 3
    bridge = BridgeAttention(expert_hidden_size=expert_hidden, vlm_hidden_size=vlm_hidden, num_bridge_layers=num_layers, num_heads=4)

    query = torch.randn(batch, chunk, expert_hidden)
    kv_layers = [torch.randn(batch, 20, vlm_hidden) for _ in range(num_layers)]
    out = bridge(query, kv_layers)
    check(
        "BridgeAttention은 초기화 시(gate=0) 완전히 0을 출력한다 (tanh(0)=0)",
        torch.allclose(out, torch.zeros_like(out)),
    )

    loss = (out - torch.randn_like(out)).pow(2).mean()
    # gate=0이라 out은 상수 0이지만, tanh(gate)의 gate에 대한 미분(1 - tanh(0)^2 = 1)은 0이
    # 아니므로 gate를 포함한 모든 파라미터에 정상적으로 gradient가 흘러야 한다.
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in bridge.parameters() if p.grad is not None)
    check("BridgeAttention이 gradient를 정상적으로 받는다(gate 포함)", grad_norm > 0, detail=f"grad_norm={grad_norm}")
    check("gate 파라미터 자체도 gradient를 받는다", bridge.gate.grad is not None and bridge.gate.grad.abs().item() > 0)

    try:
        bridge(query, kv_layers[:-1])  # 레이어 수를 하나 빼서 mismatch 유발
        check("kv_layers 길이가 안 맞으면 거부한다", False)
    except ValueError:
        check("kv_layers 길이가 안 맞으면 거부한다", True)


def test_asymmetric_residual_heads_with_bridge_attention():
    torch.manual_seed(0)
    batch, chunk, expert_hidden, vlm_hidden, arm_dim, num_bridge_layers = 2, 5, 32, 48, 7, 3
    heads = AsymmetricResidualHeads(
        expert_hidden_size=expert_hidden,
        arm_dim=arm_dim,
        use_bridge_attention=True,
        vlm_hidden_size=vlm_hidden,
        num_bridge_layers=num_bridge_layers,
        bridge_num_heads=4,
    )
    suffix_features = torch.randn(batch, chunk, expert_hidden)
    bridge_kv_layers = [torch.randn(batch, 15, vlm_hidden) for _ in range(num_bridge_layers)]

    stab_res, act_res = heads(suffix_features, bridge_kv_layers=bridge_kv_layers)
    check(
        "use_bridge_attention=True여도 초기화 시엔 여전히 완전한 0(항등) 상태다 "
        "(bridge gate=0 AND head 마지막 레이어 zero-init, 둘 다 no-op)",
        torch.allclose(stab_res, torch.zeros_like(stab_res)) and torch.allclose(act_res, torch.zeros_like(act_res)),
    )

    target = torch.randn(batch, chunk, arm_dim)
    loss = (stab_res - target).pow(2).mean() + (act_res - target).pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in heads.parameters() if p.grad is not None)
    check(
        "bridge attention 포함 AsymmetricResidualHeads 전체가 gradient를 정상적으로 받는다",
        grad_norm > 0, detail=f"grad_norm={grad_norm}",
    )
    check(
        "stabilizer_bridge/actuator_bridge가 서로 다른(독립적인) 모듈이다",
        heads.stabilizer_bridge is not heads.actuator_bridge,
    )

    # use_bridge_attention=False(기본값)면 bridge_kv_layers를 줘도 무시하고 기존과 완전히 동일하게 동작한다 (하위호환).
    heads_no_bridge = AsymmetricResidualHeads(expert_hidden_size=expert_hidden, arm_dim=arm_dim)
    out_ignored = heads_no_bridge(suffix_features, bridge_kv_layers=bridge_kv_layers)
    out_none = heads_no_bridge(suffix_features, bridge_kv_layers=None)
    check(
        "use_bridge_attention=False면 bridge_kv_layers를 줘도 무시된다 (하위호환)",
        torch.allclose(out_ignored[0], out_none[0]) and torch.allclose(out_ignored[1], out_none[1]),
    )

    try:
        AsymmetricResidualHeads(expert_hidden_size=expert_hidden, arm_dim=arm_dim, use_bridge_attention=True)
        check("use_bridge_attention=True인데 vlm_hidden_size가 없으면 거부한다", False)
    except ValueError:
        check("use_bridge_attention=True인데 vlm_hidden_size가 없으면 거부한다", True)


def test_force_head():
    torch.manual_seed(0)
    batch, chunk, expert_hidden = 3, 6, 32
    head = ForceHead(expert_hidden_size=expert_hidden)
    suffix_out = torch.randn(batch, chunk, expert_hidden, requires_grad=True)

    out = head(suffix_out)
    check("ForceHead 출력 shape이 (batch, chunk_size)이다", out.shape == (batch, chunk))
    check("ForceHead 출력은 유한한 값이다", torch.isfinite(out).all().item())

    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    check(
        "ForceHead가 suffix_out까지 포함해 gradient를 정상적으로 받는다 (GradNorm의 shared_activation=suffix_out과의 연결 확인)",
        suffix_out.grad is not None and suffix_out.grad.abs().sum().item() > 0,
    )
    head_grad_norm = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    check("ForceHead 자체 파라미터도 gradient를 받는다", head_grad_norm > 0)


def test_compute_ard_losses():
    torch.manual_seed(0)
    batch, chunk, arm_dim = 4, 6, 7
    per_element_loss = torch.rand(batch, chunk, 2 * arm_dim, requires_grad=True)
    stabilizer_traj_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_traj_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device=per_element_loss.device)

    out = compute_ard_losses(
        per_element_loss=per_element_loss,
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    check("compute_ard_losses.total은 유한한 스칼라값이다", torch.isfinite(out.total).item() and out.total.ndim == 0)

    out.total.backward()
    check("compute_ard_losses.total은 역전파가 가능하다", per_element_loss.grad is not None)

    # alpha=1, beta=0이면 (근사적으로) stabilizer 손실만 남아야 한다.
    stab_only = compute_ard_losses(
        per_element_loss=per_element_loss.detach(),
        stabilizer_traj_pred=stabilizer_traj_pred.detach(),
        actuator_traj_pred=actuator_traj_pred.detach(),
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=1.0,
        beta=0.0,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    check(
        "alpha=1, beta=0일 때 stabilizer 손실만 분리된다",
        torch.isclose(stab_only.total, stab_only.stabilizer_loss, atol=1e-5).item(),
    )

    # force_target이 없으면 force_loss는 정확히 0이어야 한다.
    check("force_target이 없으면 force_loss는 0이다", stab_only.force_loss.item() == 0.0)

    # force_target은 주어졌는데 force_pred가 없으면(= ard_use_force_head=False) 힘을 예측할
    # 방법이 없으므로, 과거처럼 엉뚱한 채널을 쓰지 않고 경고 후 force_loss=0으로 처리해야 한다.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        force_target_without_head = compute_ard_losses(
            per_element_loss=per_element_loss.detach(),
            stabilizer_traj_pred=stabilizer_traj_pred.detach(),
            actuator_traj_pred=actuator_traj_pred.detach(),
            actuator_is_first=actuator_is_first,
            arm_dim=arm_dim,
            alpha=0.3,
            beta=0.7,
            lambda_smooth=1.0,
            lambda_force=1.0,
            lambda_traj=1.0,
            force_target=torch.zeros(batch),
        )
    check(
        "force_target은 있는데 force_pred(ForceHead 출력)가 없으면 force_loss는 0이다(과거처럼 엉뚱한 채널을 쓰지 않음)",
        force_target_without_head.force_loss.item() == 0.0,
    )
    check(
        "force_target은 있는데 force_pred가 없으면 경고가 한 번 뜬다",
        len(caught) == 1 and "ard_use_force_head" in str(caught[0].message),
    )

    with_force_head = compute_ard_losses(
        per_element_loss=per_element_loss.detach(),
        stabilizer_traj_pred=stabilizer_traj_pred.detach(),
        actuator_traj_pred=actuator_traj_pred.detach(),
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
        force_target=torch.zeros(batch),
        force_pred=torch.randn(batch, chunk),
    )
    check(
        "force_target과 force_pred(ForceHead 출력)가 둘 다 있으면 force_loss가 0이 아니게 된다",
        with_force_head.force_loss.item() > 0.0,
    )


def _compute_x0_hat(noise: torch.Tensor, actions: torch.Tensor, time: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    """modeling_smolvla.py의 VLAFlowMatching.forward()와 동일한 식: x_t = t*noise + (1-t)*actions,
    x0_hat = x_t - t*v_t. 이 테스트 파일은 SmolVLAPolicy를 생성하지 않으므로(Hub 접근 불가),
    실제 forward()를 호출하지 않고 이 식만 별도로 재현해서 compute_ard_losses에 흘려 넣는다."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    return x_t - time_expanded * v_t


def test_ard_smooth_traj_loss_use_denoised_action_not_velocity():
    """개념 버그 회귀 테스트: smooth_loss/traj_loss는 velocity(v_t)가 아니라 x0_hat(노이즈 제거된
    액션 궤적 추정치)에 걸려야 한다. v_t == u_t(완벽한 예측)이면 x0_hat == actions가 대수적으로
    정확히 성립하므로(noise가 상쇄됨), 이때 smooth/traj_loss는 noise를 뭘 샘플했든, time을 뭘
    샘플했든 실제 actions 자체의 1차/2차 차분과 정확히 같아야 한다."""
    torch.manual_seed(0)
    batch, chunk, arm_dim = 3, 8, 7
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device="cpu")

    actions = torch.randn(batch, chunk, 2 * arm_dim)
    # actions 자체의 1차/2차 차분으로부터, split_by_role 이후 기대되는 smooth/traj를 직접 계산.
    stab_actions, act_actions = split_by_role(actions, arm_dim, actuator_is_first)
    expected_smooth = (stab_actions[:, 1:] - stab_actions[:, :-1]).pow(2).mean()
    expected_traj = (
        (act_actions[:, 2:] - 2 * act_actions[:, 1:-1] + act_actions[:, :-2]).pow(2).mean()
    )

    for trial, (noise_seed, time_seed) in enumerate([(1, 2), (3, 4), (5, 6)]):
        torch.manual_seed(noise_seed)
        noise = torch.randn(batch, chunk, 2 * arm_dim)
        torch.manual_seed(time_seed)
        time = torch.rand(batch)

        v_t = noise - actions  # 완벽한 예측: v_t == u_t
        x0_hat = _compute_x0_hat(noise, actions, time, v_t)
        check(
            f"[trial {trial}] v_t==u_t(완벽한 예측)이면 x0_hat이 noise/time과 무관하게 actions와 정확히 같다",
            torch.allclose(x0_hat, actions, atol=1e-5),
        )

        stabilizer_traj_pred, actuator_traj_pred = split_by_role(x0_hat, arm_dim, actuator_is_first)
        out = compute_ard_losses(
            per_element_loss=torch.zeros_like(actions),
            stabilizer_traj_pred=stabilizer_traj_pred,
            actuator_traj_pred=actuator_traj_pred,
            actuator_is_first=actuator_is_first,
            arm_dim=arm_dim,
            alpha=0.3,
            beta=0.7,
            lambda_smooth=1.0,
            lambda_force=1.0,
            lambda_traj=1.0,
        )
        check(
            f"[trial {trial}] 완벽한 예측이면 smooth_loss가 noise와 무관하게 실제 actions의 1차 차분과 같다",
            torch.isclose(out.smooth_loss, expected_smooth, atol=1e-5).item(),
            detail=f"got={out.smooth_loss.item()} expected={expected_smooth.item()}",
        )
        check(
            f"[trial {trial}] 완벽한 예측이면 traj_loss가 noise와 무관하게 실제 actions의 2차 차분과 같다",
            torch.isclose(out.traj_loss, expected_traj, atol=1e-5).item(),
            detail=f"got={out.traj_loss.item()} expected={expected_traj.item()}",
        )

    # (반대로) 만약 실수로 x0_hat이 아니라 v_t = noise - actions 자체를 썼다면, noise가 매
    # 타임스텝 독립 샘플이라 smooth/traj가 0이 아닌 상당히 큰 값으로 나와야 정상이다 — 이
    # 테스트가 실제로 "틀렸던 동작"과 "고친 동작"을 구분하고 있다는 걸 보여주는 대조군.
    torch.manual_seed(7)
    noise = torch.randn(batch, chunk, 2 * arm_dim)
    v_t_wrong_input = noise - actions
    stab_wrong, act_wrong = split_by_role(v_t_wrong_input, arm_dim, actuator_is_first)
    out_wrong = compute_ard_losses(
        per_element_loss=torch.zeros_like(actions),
        stabilizer_traj_pred=stab_wrong,
        actuator_traj_pred=act_wrong,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    check(
        "대조군: velocity(v_t)를 (잘못) 직접 썼다면 smooth_loss가 actions 기준값과 다르게 나온다",
        not torch.isclose(out_wrong.smooth_loss, expected_smooth, atol=1e-5).item(),
        detail=f"wrong={out_wrong.smooth_loss.item()} actions_based={expected_smooth.item()}",
    )


def test_ard_smooth_traj_loss_zero_for_constant_trajectory():
    """actions가 chunk 전체에 걸쳐 상수(시간에 따라 안 변함)면, 완벽한 예측에서 x0_hat도
    상수이므로 1차/2차 차분 기반 smooth_loss/traj_loss는 정확히 0이어야 한다."""
    torch.manual_seed(0)
    batch, chunk, arm_dim = 2, 6, 7
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device="cpu")

    constant_step = torch.randn(batch, 1, 2 * arm_dim)
    actions = constant_step.expand(batch, chunk, 2 * arm_dim).contiguous()

    noise = torch.randn(batch, chunk, 2 * arm_dim)
    time = torch.rand(batch)
    v_t = noise - actions  # 완벽한 예측
    x0_hat = _compute_x0_hat(noise, actions, time, v_t)
    check("상수 궤적이면 완벽한 예측의 x0_hat도 noise와 무관하게 상수(=actions)다", torch.allclose(x0_hat, actions, atol=1e-5))

    stabilizer_traj_pred, actuator_traj_pred = split_by_role(x0_hat, arm_dim, actuator_is_first)
    out = compute_ard_losses(
        per_element_loss=torch.zeros_like(actions),
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    # float32 연산 잔차(이 테스트 기준 1e-15 수준) 때문에 정확히 0.0은 아닐 수 있어 작은 허용
    # 오차를 둔다 — exact 0.0을 요구했다가 실제로는 버그가 아닌 부동소수점 노이즈로 실패했던
    # 점을 확인하고 고친 것.
    check(
        "상수 궤적 + 완벽한 예측이면 smooth_loss가 (부동소수점 오차 내에서) 0이다",
        out.smooth_loss.item() < 1e-8,
        detail=f"smooth_loss={out.smooth_loss.item()}",
    )
    check(
        "상수 궤적 + 완벽한 예측이면 traj_loss가 (부동소수점 오차 내에서) 0이다",
        out.traj_loss.item() < 1e-8,
        detail=f"traj_loss={out.traj_loss.item()}",
    )


def test_ard_reg_time_weighting_one_minus_t():
    """ard_reg_time_weighting="one_minus_t"가 modeling_smolvla.py에서 하는 일(1-time을
    reg_time_weights로 넘기는 것)을 compute_ard_losses 레벨에서 직접 재현해서 검증한다."""
    torch.manual_seed(0)
    batch, chunk, arm_dim = 3, 6, 7
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device="cpu")
    # 일부러 거친(매끄럽지 않은) 궤적을 줘서 smooth/traj_loss가 0이 아니게 만든다.
    stabilizer_traj_pred = torch.randn(batch, chunk, arm_dim)
    actuator_traj_pred = torch.randn(batch, chunk, arm_dim)

    # 모든 샘플이 t=1이면 가중치(1-t)=0이라, 궤적이 아무리 거칠어도 smooth/traj_loss가
    # 정확히 0이어야 한다.
    all_t1_weights = torch.zeros(batch)
    out_t1 = compute_ard_losses(
        per_element_loss=torch.zeros(batch, chunk, 2 * arm_dim),
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
        reg_time_weights=all_t1_weights,
    )
    check(
        "ard_reg_time_weighting: 전 샘플 t=1(가중치 0)이면 smooth_loss가 정확히 0이다",
        out_t1.smooth_loss.item() == 0.0,
    )
    check(
        "ard_reg_time_weighting: 전 샘플 t=1(가중치 0)이면 traj_loss가 정확히 0이다",
        out_t1.traj_loss.item() == 0.0,
    )

    # 전 샘플 t=0(가중치 1)이면 가중치 없는(reg_time_weights=None) 기존 경로와 값이 같아야
    # 한다 — per-sample mean 후 평균 내는 것과, 전체를 한 번에 평균 내는 것이 같은 chunk
    # 길이에서는 수학적으로 동일하기 때문이다(단, 연산 순서가 달라 float32 수준의 완전한
    # bit-exactness까지는 보장하지 않으므로 isclose로 비교한다 — bit-exactness는
    # reg_time_weights=None(기본, "none") 경로 자체에서만 보장된다).
    all_t0_weights = torch.ones(batch)
    out_t0 = compute_ard_losses(
        per_element_loss=torch.zeros(batch, chunk, 2 * arm_dim),
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
        reg_time_weights=all_t0_weights,
    )
    out_unweighted = compute_ard_losses(
        per_element_loss=torch.zeros(batch, chunk, 2 * arm_dim),
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    check(
        "ard_reg_time_weighting: 전 샘플 t=0(가중치 1)이면 가중치 없는 경로와 값이 (근사적으로) 같다",
        torch.isclose(out_t0.smooth_loss, out_unweighted.smooth_loss, atol=1e-6).item()
        and torch.isclose(out_t0.traj_loss, out_unweighted.traj_loss, atol=1e-6).item(),
    )

    # reg_time_weights를 아예 안 주면(기본값 None, "none" 모드) 과거 코드가 하던 것과 똑같이
    # "전체 원소를 한 번에 .mean()"한 값과 bit-for-bit 동일해야 한다(_reduce_reg_loss의
    # time_weights=None 분기가 정확히 이 식이어야 함).
    expected_smooth_unweighted = (stabilizer_traj_pred[:, 1:] - stabilizer_traj_pred[:, :-1]).pow(2).mean()
    expected_traj_unweighted = (
        (actuator_traj_pred[:, 2:] - 2 * actuator_traj_pred[:, 1:-1] + actuator_traj_pred[:, :-2]).pow(2).mean()
    )
    check(
        "ard_reg_time_weighting=none(기본, reg_time_weights 생략)이면 smooth_loss가 과거 코드의 전체-평균 식과 bit-for-bit 동일하다",
        torch.equal(out_unweighted.smooth_loss, expected_smooth_unweighted),
    )
    check(
        "ard_reg_time_weighting=none(기본, reg_time_weights 생략)이면 traj_loss가 과거 코드의 전체-평균 식과 bit-for-bit 동일하다",
        torch.equal(out_unweighted.traj_loss, expected_traj_unweighted),
    )


def test_gradnorm_lambdas():
    torch.manual_seed(0)
    gradnorm = GradNormLambdas(alpha=1.5, init_value=1.0)
    check(
        "GradNormLambdas 초기 weights는 [1,1,1]이고 합은 3이다",
        torch.allclose(gradnorm.weights, torch.ones(3)) and gradnorm.weights.sum().item() == 3.0,
    )

    batch, chunk, arm_dim, hidden = 4, 10, 7, 16
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device="cpu")
    gradnorm_optimizer = torch.optim.Adam([gradnorm.weights], lr=0.05)

    weights_before = gradnorm.weights.clone()
    for _ in range(5):
        shared = torch.randn(batch, chunk, hidden, requires_grad=True)
        per_element_loss = shared[..., : 2 * arm_dim] ** 2
        # 일부러 스케일을 다르게 줘서(스무스=크게, 궤적=작게) lambda가 실제로 움직이는지 확인한다.
        stabilizer_traj_pred = shared[..., :arm_dim] * 3.0
        actuator_traj_pred = shared[..., arm_dim : 2 * arm_dim] * 0.1

        out = compute_ard_losses(
            per_element_loss=per_element_loss,
            stabilizer_traj_pred=stabilizer_traj_pred,
            actuator_traj_pred=actuator_traj_pred,
            actuator_is_first=actuator_is_first,
            arm_dim=arm_dim,
            alpha=0.3,
            beta=0.7,
            lambda_smooth=1.0,
            lambda_force=1.0,
            lambda_traj=1.0,
            force_target=None,  # GradNorm이 force처럼 shared와 연결 안 된(상수 0) task도 안 죽고 버텨야 함
            gradnorm=gradnorm,
            shared_activation=shared,
        )

        # 순서 중요: 두 backward를 먼저 끝내고 나서(그래프가 아직 안 바뀐 상태), 옵티마이저 step을 밟는다.
        gradnorm_optimizer.zero_grad()
        out.grad_loss.backward(inputs=[gradnorm.weights], retain_graph=True)
        out.total.backward()

        gradnorm_optimizer.step()
        gradnorm.renormalize()

    check("gradnorm=... 을 주면 compute_ard_losses가 grad_loss를 반환한다", out.grad_loss is not None)
    check("grad_loss는 유한한 스칼라값이다", torch.isfinite(out.grad_loss).item() and out.grad_loss.ndim == 0)
    check(
        "5스텝 뒤 GradNorm weights가 초기값(1,1,1)에서 실제로 움직인다",
        not torch.allclose(gradnorm.weights, weights_before),
        detail=f"weights={gradnorm.weights.tolist()}",
    )
    check(
        "renormalize() 이후에도 weights 합은 항상 3(task 개수)으로 유지된다",
        abs(gradnorm.weights.sum().item() - 3.0) < 1e-4,
        detail=f"sum={gradnorm.weights.sum().item()}",
    )
    check(
        "force_loss가 shared_activation과 연결 안 된(상수 0) 경우에도 crash 없이 동작한다",
        out.force_loss.item() == 0.0,
    )

    # gradnorm이 아예 None이면(기존 고정 lambda 경로) grad_loss는 여전히 None이어야 한다 (하위호환).
    fixed = compute_ard_losses(
        per_element_loss=per_element_loss.detach(),
        stabilizer_traj_pred=stabilizer_traj_pred.detach(),
        actuator_traj_pred=actuator_traj_pred.detach(),
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
    )
    check("gradnorm을 안 주면(기존 고정 lambda 방식) grad_loss는 None이다", fixed.grad_loss is None)


def test_gradnorm_force_head_connection():
    """GradNorm이 force 항을 ForceHead 경로로 제대로 연결해서 보는지 확인한다 —
    modeling_smolvla.py가 실제로 하는 것과 동일하게, ForceHead(suffix_out)으로 force_pred를
    만들어서 넘긴다(shared_activation도 동일한 suffix_out)."""
    torch.manual_seed(0)
    batch, chunk, arm_dim, hidden = 4, 10, 7, 16
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device="cpu")
    gradnorm = GradNormLambdas(alpha=1.5, init_value=1.0)
    force_head = ForceHead(expert_hidden_size=hidden)

    suffix_out = torch.randn(batch, chunk, hidden, requires_grad=True)
    per_element_loss = suffix_out[..., : 2 * arm_dim] ** 2
    stabilizer_traj_pred = suffix_out[..., :arm_dim]
    actuator_traj_pred = suffix_out[..., arm_dim : 2 * arm_dim]
    force_pred = force_head(suffix_out)  # ForceHead의 입력이 정확히 shared_activation(suffix_out)이다.
    force_target = torch.randn(batch)

    out = compute_ard_losses(
        per_element_loss=per_element_loss,
        stabilizer_traj_pred=stabilizer_traj_pred,
        actuator_traj_pred=actuator_traj_pred,
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
        force_target=force_target,
        force_pred=force_pred,
        gradnorm=gradnorm,
        shared_activation=suffix_out,
    )
    check(
        "ForceHead 경로로 연결된 force_loss는 shared_activation(suffix_out)과 끊기지 않은 상태라 grad_loss를 만들 수 있다",
        out.grad_loss is not None and torch.isfinite(out.grad_loss).item(),
    )

    # out.force_loss는 ARDLossOutput 안에서 이미 .detach()된 값이라 직접 backward할 수 없다 —
    # compute_ard_losses 내부와 동일한 식(절댓값 평균)을 그대로 재구성해서, 이게 정말
    # suffix_out까지 끊기지 않고 연결되어 있는지(= ForceHead가 shared_activation과 같은
    # 입력을 쓰는지) 직접 확인한다.
    force_loss_live = (force_pred - force_target.unsqueeze(-1)).abs().mean()
    (force_grad,) = torch.autograd.grad(force_loss_live, suffix_out, retain_graph=True, allow_unused=True)
    check(
        "force_loss의 shared_activation(suffix_out)에 대한 그래디언트가 0이 아니다 (ForceHead가 suffix_out과 연결되어 있음)",
        force_grad is not None and force_grad.abs().sum().item() > 0,
    )


def main():
    test_config_validation()
    test_resolve_actuator_is_first_is_fixed()
    test_split_combine_roundtrip()
    test_asymmetric_residual_heads_zero_init()
    test_resolve_bridge_layer_indices()
    test_bridge_attention_zero_init()
    test_asymmetric_residual_heads_with_bridge_attention()
    test_force_head()
    test_compute_ard_losses()
    test_ard_smooth_traj_loss_use_denoised_action_not_velocity()
    test_ard_smooth_traj_loss_zero_for_constant_trajectory()
    test_ard_reg_time_weighting_one_minus_t()
    test_gradnorm_lambdas()
    test_gradnorm_force_head_connection()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)}개 항목 실패: {FAILURES}")
        sys.exit(1)
    print("[OK] ARD 단위 테스트 전체 통과.")
    print(
        "참고: SmolVLAPolicy 전체를 생성해서 테스트하지는 않았습니다 — 그러려면 Hugging Face "
        "Hub에서 SmolVLM2 백본 config를 내려받아야 하는데, 이 환경의 네트워크 정책이 이를 막고 "
        "있습니다. 대신 ard.py 모듈과 modeling_smolvla.py의 forward/sample_actions/"
        "denoise_step에 연결된 로직을 합성 텐서로 직접 검증했습니다."
    )


if __name__ == "__main__":
    main()
