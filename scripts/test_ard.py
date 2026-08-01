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

import torch

from lerobot.policies.smolvla.ard import (
    AsymmetricResidualHeads,
    combine_by_role,
    compute_ard_losses,
    resolve_actuator_is_first,
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


def test_compute_ard_losses():
    torch.manual_seed(0)
    batch, chunk, arm_dim = 4, 6, 7
    per_element_loss = torch.rand(batch, chunk, 2 * arm_dim, requires_grad=True)
    stabilizer_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_is_first = resolve_actuator_is_first("right", batch_size=batch, device=per_element_loss.device)

    out = compute_ard_losses(
        per_element_loss=per_element_loss,
        stabilizer_pred=stabilizer_pred,
        actuator_pred=actuator_pred,
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
        stabilizer_pred=stabilizer_pred.detach(),
        actuator_pred=actuator_pred.detach(),
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

    with_force = compute_ard_losses(
        per_element_loss=per_element_loss.detach(),
        stabilizer_pred=stabilizer_pred.detach(),
        actuator_pred=actuator_pred.detach(),
        actuator_is_first=actuator_is_first,
        arm_dim=arm_dim,
        alpha=0.3,
        beta=0.7,
        lambda_smooth=1.0,
        lambda_force=1.0,
        lambda_traj=1.0,
        force_target=torch.zeros(batch),
    )
    check("force_target을 주면 force_loss가 0이 아니게 된다", with_force.force_loss.item() > 0.0)


def main():
    test_config_validation()
    test_resolve_actuator_is_first_is_fixed()
    test_split_combine_roundtrip()
    test_asymmetric_residual_heads_zero_init()
    test_compute_ard_losses()

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
