# Copyright 2026 The ARD-VLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Asymmetric Role Decomposition (ARD) for bimanual tool-use fine-tuning of SmolVLA.

Human tool use is asymmetric: one hand (the Stabilizer) holds the workpiece still while the
other (the Actuator) performs the precise manipulation. Standard bimanual VLA training treats
both arms symmetrically. ARD instead splits the leading `2 * ard_arm_dim` action channels into a
Stabilizer block and an Actuator block, gives each its own lightweight residual head, and trains
them with different loss terms combined asymmetrically (alpha * L_stab + beta * L_act, default
0.3 / 0.7). See the ARD-VLA research plan for the full method.

Convention: within those `2 * ard_arm_dim` channels, the first `ard_arm_dim` belong to the left
arm and the next `ard_arm_dim` to the right arm — this matches LeRobot's usual bimanual action
layout (e.g. bi_so_follower). Which physical arm plays the Actuator role for a given sample is
resolved by `resolve_actuator_is_first`, in priority order: an explicit per-sample dataset label,
then a learned `RoleClassifier` prediction from the language instruction, then the config's
static default.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Optional per-sample batch keys a dataset/environment may provide.
ARD_ROLE_LABEL = "ard_actuator_is_first"  # bool/float, shape (batch,): True if the left arm is the Actuator
ARD_FORCE_TARGET = "ard_force_target"  # float, shape (batch,) or (batch, chunk_size): target contact force/torque


class RoleClassifier(nn.Module):
    """Predicts, from a pooled language-instruction embedding, whether the left arm (first
    `ard_arm_dim` action channels) is the Actuator for this instruction."""

    def __init__(self, hidden_size: int, classifier_hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, pooled_lang_emb: Tensor) -> Tensor:
        """pooled_lang_emb: (batch, hidden_size). Returns logits of shape (batch,);
        positive => left arm predicted as Actuator."""
        return self.net(pooled_lang_emb).squeeze(-1)


def pool_language_embedding(lang_emb: Tensor, lang_masks: Tensor) -> Tensor:
    """Mean-pool token embeddings over valid (non-padding) language positions.

    lang_emb: (batch, seq_len, hidden_size). lang_masks: (batch, seq_len) bool/int, True/1 = valid token.
    """
    mask = lang_masks.to(dtype=lang_emb.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (lang_emb * mask).sum(dim=1) / counts


class AsymmetricResidualHeads(nn.Module):
    """Per-role residual heads that refine the shared flow-matching output on each arm's channels.

    The base model already produces a full `max_action_dim`-wide prediction via a single shared
    linear projection (`action_out_proj`); these heads add a role-specialized correction on top of
    the Stabilizer and Actuator channel blocks, giving each role dedicated learnable capacity
    without duplicating the whole action expert transformer.
    """

    def __init__(self, expert_hidden_size: int, arm_dim: int, mlp_hidden_dim: int | None = None):
        super().__init__()
        mlp_hidden_dim = mlp_hidden_dim or max(expert_hidden_size // 2, arm_dim)
        self.stabilizer_head = nn.Sequential(
            nn.Linear(expert_hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, arm_dim),
        )
        self.actuator_head = nn.Sequential(
            nn.Linear(expert_hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, arm_dim),
        )
        # Zero-init the last layer of each head so ARD starts as a pure identity/no-op on top of
        # the pretrained shared projection, and only diverges from it as training proceeds.
        nn.init.zeros_(self.stabilizer_head[-1].weight)
        nn.init.zeros_(self.stabilizer_head[-1].bias)
        nn.init.zeros_(self.actuator_head[-1].weight)
        nn.init.zeros_(self.actuator_head[-1].bias)

    def forward(self, suffix_features: Tensor) -> tuple[Tensor, Tensor]:
        """suffix_features: (batch, chunk_size, expert_hidden_size), the action expert's pre-projection
        transformer output. Returns (stabilizer_residual, actuator_residual), each
        (batch, chunk_size, arm_dim)."""
        return self.stabilizer_head(suffix_features), self.actuator_head(suffix_features)


def resolve_actuator_is_first(
    default_actuator_arm: str,
    batch_size: int,
    device: torch.device,
    role_label: Tensor | None = None,
    role_logits: Tensor | None = None,
) -> Tensor:
    """Decide, per sample, whether the left arm (first `ard_arm_dim` action channels) is the
    Actuator. Priority: explicit dataset role label > learned role-classifier prediction > the
    config's static default. Returns a bool tensor of shape (batch_size,)."""
    if role_label is not None:
        return role_label.to(device=device).bool()
    if role_logits is not None:
        return role_logits > 0
    default_is_left = default_actuator_arm == "left"
    return torch.full((batch_size,), default_is_left, dtype=torch.bool, device=device)


def split_by_role(x: Tensor, arm_dim: int, actuator_is_first: Tensor) -> tuple[Tensor, Tensor]:
    """Split the leading `2 * arm_dim` channels of `x` into (stabilizer, actuator) sub-tensors,
    routing the left-arm or right-arm block to each role per-sample.

    x: (batch, ..., action_dim >= 2 * arm_dim). actuator_is_first: (batch,) bool — True if the
    left-arm block (channels [0:arm_dim]) is this sample's Actuator.
    """
    left = x[..., :arm_dim]
    right = x[..., arm_dim : 2 * arm_dim]
    mask = actuator_is_first.view(-1, *([1] * (x.ndim - 1)))
    actuator = torch.where(mask, left, right)
    stabilizer = torch.where(mask, right, left)
    return stabilizer, actuator


def combine_by_role(stabilizer: Tensor, actuator: Tensor, actuator_is_first: Tensor) -> tuple[Tensor, Tensor]:
    """Inverse of `split_by_role`: route the per-role (stabilizer, actuator) tensors back to
    per-sample (left, right) arm order. Returns (left, right), each (batch, ..., arm_dim)."""
    mask = actuator_is_first.view(-1, *([1] * (stabilizer.ndim - 1)))
    left = torch.where(mask, actuator, stabilizer)
    right = torch.where(mask, stabilizer, actuator)
    return left, right


@dataclass
class ARDLossOutput:
    total: Tensor
    stabilizer_loss: Tensor
    actuator_loss: Tensor
    pos_loss: Tensor
    smooth_loss: Tensor
    force_loss: Tensor
    traj_loss: Tensor
    role_loss: Tensor | None


def compute_ard_losses(
    per_element_loss: Tensor,
    stabilizer_pred: Tensor,
    actuator_pred: Tensor,
    actuator_is_first: Tensor,
    arm_dim: int,
    alpha: float,
    beta: float,
    lambda_smooth: float,
    lambda_force: float,
    lambda_traj: float,
    force_target: Tensor | None = None,
    role_logits: Tensor | None = None,
    role_label: Tensor | None = None,
    role_loss_weight: float = 0.0,
) -> ARDLossOutput:
    """Combine the base flow-matching regression loss with ARD's role-specific regularizers.

    per_element_loss: (batch, chunk_size, 2 * arm_dim) — the base model's per-channel flow-matching
        MSE(u_t, v_t) on the leading left-arm/right-arm channels, in the raw (unrouted) left/right
        layout. Routed to (stabilizer, actuator) here via `split_by_role` before being reduced. This
        stands in for the L_pos term from the ARD-VLA plan: it is the actual signal that trains the
        network to track the target trajectory, so ARD reuses it rather than recomputing a separate
        L1 position loss.
    stabilizer_pred / actuator_pred: (batch, chunk_size, arm_dim) — the predicted velocity field on
        the channels already routed to that role for each sample (see `split_by_role`). Used for the
        smoothness / trajectory regularizers below, since flow-matching training only produces a
        predicted velocity field per diffusion step (not a fully denoised action sequence), so its
        temporal profile across the chunk is the closest available proxy for "the predicted
        trajectory".
    actuator_is_first: (batch,) bool, as returned by `resolve_actuator_is_first` — needed here to
        route `per_element_loss` the same way `stabilizer_pred`/`actuator_pred` were already routed.
    """
    stab_pos_per_elem, act_pos_per_elem = split_by_role(per_element_loss, arm_dim, actuator_is_first)
    stab_pos_loss = stab_pos_per_elem.mean()
    act_pos_loss = act_pos_per_elem.mean()

    # L_smooth = sum |s_t - s_{t-1}|^2 over the predicted stabilizer trajectory within the chunk.
    if stabilizer_pred.shape[1] > 1:
        smooth_loss = (stabilizer_pred[:, 1:] - stabilizer_pred[:, :-1]).pow(2).mean()
    else:
        smooth_loss = stabilizer_pred.new_zeros(())

    # L_traj = sum |a_t - 2a_{t-1} + a_{t-2}|^2, a second-order smoothness penalty on the actuator
    # trajectory (penalizes abrupt direction changes during precise tool manipulation).
    if actuator_pred.shape[1] > 2:
        second_diff = actuator_pred[:, 2:] - 2 * actuator_pred[:, 1:-1] + actuator_pred[:, :-2]
        traj_loss = second_diff.pow(2).mean()
    else:
        traj_loss = actuator_pred.new_zeros(())

    # L_force: optional contact-force/torque tracking, e.g. supplied by an Isaac Sim contact
    # sensor. Not part of any current dataset in this repo, so it defaults to zero — wiring in a
    # real force signal is future work (see the ARD-VLA research plan, "실물 실증").
    if force_target is not None:
        force_pred = actuator_pred[..., -1]  # (batch, chunk_size): last actuator channel as a force proxy
        target = force_target.to(force_pred.dtype)
        if target.ndim == 1:  # (batch,) -> broadcast the same target across the chunk
            target = target.unsqueeze(-1)
        force_loss = (force_pred - target).abs().mean()
    else:
        force_loss = actuator_pred.new_zeros(())

    stab_loss = stab_pos_loss + lambda_smooth * smooth_loss
    act_loss = act_pos_loss + lambda_force * force_loss + lambda_traj * traj_loss
    total = alpha * stab_loss + beta * act_loss

    role_loss = None
    if role_logits is not None and role_label is not None and role_loss_weight > 0:
        role_loss = F.binary_cross_entropy_with_logits(role_logits, role_label.to(role_logits.dtype))
        total = total + role_loss_weight * role_loss

    return ARDLossOutput(
        total=total,
        stabilizer_loss=stab_loss.detach(),
        actuator_loss=act_loss.detach(),
        pos_loss=((stab_pos_loss + act_pos_loss) / 2).detach(),
        smooth_loss=smooth_loss.detach(),
        force_loss=force_loss.detach(),
        traj_loss=traj_loss.detach(),
        role_loss=role_loss.detach() if role_loss is not None else None,
    )
