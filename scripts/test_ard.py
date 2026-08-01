#!/usr/bin/env python3
"""Offline tests for the ARD (Asymmetric Role Decomposition) SmolVLA modification.

Exercises lerobot.policies.smolvla.ard directly with synthetic tensors shaped like the real
training/inference pipeline, plus SmolVLAConfig's ARD validation. Does not construct the full
SmolVLAPolicy: that requires downloading the SmolVLM2 backbone config from the Hugging Face Hub,
which this environment's network policy blocks. Safe to run on any machine, GPU or not.
"""

import sys

import torch

from lerobot.policies.smolvla.ard import (
    AsymmetricResidualHeads,
    RoleClassifier,
    combine_by_role,
    compute_ard_losses,
    pool_language_embedding,
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
    check("SmolVLAConfig(use_ard=True) constructs with valid dims", cfg.use_ard is True)

    try:
        SmolVLAConfig(use_ard=True, ard_arm_dim=20, max_action_dim=32)
        check("rejects ard_arm_dim too large for max_action_dim", False)
    except ValueError:
        check("rejects ard_arm_dim too large for max_action_dim", True)

    try:
        SmolVLAConfig(use_ard=True, ard_default_actuator_arm="both")
        check("rejects invalid ard_default_actuator_arm", False)
    except ValueError:
        check("rejects invalid ard_default_actuator_arm", True)


def test_split_combine_roundtrip():
    torch.manual_seed(0)
    batch, chunk, arm_dim = 4, 5, 7
    x = torch.randn(batch, chunk, 2 * arm_dim)
    actuator_is_first = torch.tensor([True, False, True, False])

    stab, act = split_by_role(x, arm_dim, actuator_is_first)
    check("split_by_role output shapes", stab.shape == (batch, chunk, arm_dim) and act.shape == (batch, chunk, arm_dim))

    left, right = combine_by_role(stab, act, actuator_is_first)
    recombined = torch.cat([left, right], dim=-1)
    check("split_by_role -> combine_by_role round-trips to the original tensor", torch.allclose(recombined, x))

    # Sample 0: actuator_is_first=True, so the actuator half must equal the original LEFT half.
    check("actuator_is_first routes the left block to the actuator", torch.allclose(act[0], x[0, :, :arm_dim]))
    # Sample 1: actuator_is_first=False, so the actuator half must equal the original RIGHT half.
    check("actuator_is_first=False routes the right block to the actuator", torch.allclose(act[1], x[1, :, arm_dim:]))


def test_resolve_actuator_is_first_priority():
    device = torch.device("cpu")
    label = torch.tensor([1.0, 0.0])
    logits = torch.tensor([-5.0, 5.0])

    out = resolve_actuator_is_first("right", 2, device, role_label=label, role_logits=logits)
    check("role_label takes priority over role_logits", torch.equal(out, torch.tensor([True, False])))

    out = resolve_actuator_is_first("right", 2, device, role_label=None, role_logits=logits)
    check("role_logits used when no role_label", torch.equal(out, torch.tensor([False, True])))

    out = resolve_actuator_is_first("left", 3, device, role_label=None, role_logits=None)
    check("static default used when neither label nor logits given", bool(out.all()))


def test_role_classifier_and_pooling():
    torch.manual_seed(0)
    batch, seq_len, hidden = 3, 6, 16
    lang_emb = torch.randn(batch, seq_len, hidden)
    lang_masks = torch.zeros(batch, seq_len, dtype=torch.bool)
    lang_masks[:, :3] = True  # first 3 tokens valid, rest padding

    pooled = pool_language_embedding(lang_emb, lang_masks)
    check("pool_language_embedding shape", pooled.shape == (batch, hidden))
    manual = lang_emb[:, :3].mean(dim=1)
    check("pool_language_embedding ignores padding", torch.allclose(pooled, manual, atol=1e-5))

    clf = RoleClassifier(hidden_size=hidden, classifier_hidden_dim=8)
    logits = clf(pooled)
    check("RoleClassifier output shape", logits.shape == (batch,))


def test_asymmetric_residual_heads_zero_init():
    torch.manual_seed(0)
    batch, chunk, expert_hidden, arm_dim = 2, 5, 32, 7
    heads = AsymmetricResidualHeads(expert_hidden_size=expert_hidden, arm_dim=arm_dim)
    suffix_features = torch.randn(batch, chunk, expert_hidden)
    stab_res, act_res = heads(suffix_features)
    check(
        "AsymmetricResidualHeads is a zero no-op at init (identity on top of pretrained output)",
        torch.allclose(stab_res, torch.zeros_like(stab_res)) and torch.allclose(act_res, torch.zeros_like(act_res)),
    )

    # After a gradient step the heads should move away from zero.
    target = torch.randn(batch, chunk, arm_dim)
    loss = (stab_res - target).pow(2).mean() + (act_res - target).pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in heads.parameters() if p.grad is not None)
    check("AsymmetricResidualHeads receives gradients", grad_norm > 0, detail=f"grad_norm={grad_norm}")


def test_compute_ard_losses():
    torch.manual_seed(0)
    batch, chunk, arm_dim = 4, 6, 7
    per_element_loss = torch.rand(batch, chunk, 2 * arm_dim, requires_grad=True)
    stabilizer_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_pred = torch.randn(batch, chunk, arm_dim, requires_grad=True)
    actuator_is_first = torch.tensor([True, False, True, False])

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
    check("compute_ard_losses.total is a finite scalar", torch.isfinite(out.total).item() and out.total.ndim == 0)

    out.total.backward()
    check("compute_ard_losses.total is backprop-able", per_element_loss.grad is not None)

    # alpha=1, beta=0 should reduce to (approximately) the stabilizer-only loss.
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
        "alpha=1, beta=0 isolates the stabilizer loss",
        torch.isclose(stab_only.total, stab_only.stabilizer_loss, atol=1e-5).item(),
    )

    # Force loss should be exactly zero when no force_target is supplied.
    check("force_loss defaults to zero without a force_target", stab_only.force_loss.item() == 0.0)

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
    check("force_loss becomes nonzero once a force_target is supplied", with_force.force_loss.item() > 0.0)

    # Role auxiliary loss only kicks in when both a classifier prediction and a label are given.
    role_logits = torch.randn(batch, requires_grad=True)
    role_label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    with_role = compute_ard_losses(
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
        role_logits=role_logits,
        role_label=role_label,
        role_loss_weight=0.1,
    )
    check("role_loss is populated when logits + label + weight>0 are all given", with_role.role_loss is not None)


def main():
    test_config_validation()
    test_split_combine_roundtrip()
    test_resolve_actuator_is_first_priority()
    test_role_classifier_and_pooling()
    test_asymmetric_residual_heads_zero_init()
    test_compute_ard_losses()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("[OK] All ARD unit tests passed.")
    print(
        "Note: this does not construct SmolVLAPolicy end-to-end — that requires downloading the "
        "SmolVLM2 backbone config from the Hugging Face Hub, which this environment's network "
        "policy blocks. The ard.py module and its wiring into modeling_smolvla.py's forward/"
        "sample_actions/denoise_step are exercised directly with synthetic tensors instead."
    )


if __name__ == "__main__":
    main()
