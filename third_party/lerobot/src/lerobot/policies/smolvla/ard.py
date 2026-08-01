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
"""SmolVLA의 양손 도구 조작 파인튜닝을 위한 ARD (Asymmetric Role Decomposition, 비대칭 역할 분리).

사람이 도구를 쓸 때는 비대칭적이다: 한 손(Stabilizer)은 작업물을 가만히 붙잡고, 다른 손
(Actuator)이 정밀한 조작을 수행한다. 일반적인 양손 VLA 학습은 두 팔을 대칭적으로 다루지만,
ARD는 선두 `2 * ard_arm_dim`개 액션 채널을 Stabilizer 블록과 Actuator 블록으로 나누고, 각각에
전용 경량 residual head를 부여한 뒤, 서로 다른 손실 항을 비대칭적으로 결합해 학습한다
(alpha * L_stab + beta * L_act, 기본값 0.3 / 0.7). 전체 방법론은 ARD-VLA 연구계획서를 참고.

규칙: 이 `2 * ard_arm_dim`개 채널 중 앞쪽 `ard_arm_dim`개는 왼팔, 다음 `ard_arm_dim`개는
오른팔에 해당한다 — LeRobot의 일반적인 양손 액션 레이아웃(예: bi_so_follower)과 동일하다.
어느 물리적 팔이 Actuator 역할을 맡을지는 `config.ard_default_actuator_arm`으로 고정된다
(이 프로젝트에서는 항상 오른팔 — `resolve_actuator_is_first` 참고).
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# 데이터셋/환경이 선택적으로 제공할 수 있는 배치 키.
ARD_FORCE_TARGET = "ard_force_target"  # float, shape (batch,) 또는 (batch, chunk_size): 목표 접촉력/토크


class AsymmetricResidualHeads(nn.Module):
    """두 팔 각자의 채널에 대해 flow-matching 공유 출력을 보정하는 역할별 residual head.

    베이스 모델은 이미 하나의 공유 선형 projection(`action_out_proj`)으로 `max_action_dim` 전체
    길이의 예측을 만들어낸다. 이 head들은 전체 액션 전문가(action expert) 트랜스포머를 통째로
    복제하지 않으면서도, Stabilizer/Actuator 채널 블록 위에 역할 특화 보정을 추가로 얹어 각
    역할에 전용 학습 용량을 부여한다.
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
        # 각 head의 마지막 레이어를 0으로 초기화해서, ARD가 처음에는 사전학습된 공유 projection
        # 위에 아무 영향도 주지 않는 항등(identity) 상태로 시작하고, 학습이 진행되면서만
        # 점진적으로 갈라져 나가도록 한다.
        nn.init.zeros_(self.stabilizer_head[-1].weight)
        nn.init.zeros_(self.stabilizer_head[-1].bias)
        nn.init.zeros_(self.actuator_head[-1].weight)
        nn.init.zeros_(self.actuator_head[-1].bias)

    def forward(self, suffix_features: Tensor) -> tuple[Tensor, Tensor]:
        """suffix_features: (batch, chunk_size, expert_hidden_size), 액션 전문가의 projection 이전
        트랜스포머 출력. (stabilizer_residual, actuator_residual)을 반환하며 각각 shape은
        (batch, chunk_size, arm_dim)이다."""
        return self.stabilizer_head(suffix_features), self.actuator_head(suffix_features)


def resolve_actuator_is_first(default_actuator_arm: str, batch_size: int, device: torch.device) -> Tensor:
    """Actuator 팔은 config로 고정되어 있다(이 프로젝트 설정에서는 항상 오른팔, 즉
    `default_actuator_arm == "right"`). 모든 샘플에 동일하게 적용되는 상수 bool 텐서
    (batch_size,)를 반환한다 — 왼팔(앞쪽 `ard_arm_dim` 채널)이 Actuator일 때만 True."""
    default_is_left = default_actuator_arm == "left"
    return torch.full((batch_size,), default_is_left, dtype=torch.bool, device=device)


def split_by_role(x: Tensor, arm_dim: int, actuator_is_first: Tensor) -> tuple[Tensor, Tensor]:
    """`x`의 선두 `2 * arm_dim`개 채널을 (stabilizer, actuator) 서브텐서로 분리하고, 샘플별로
    왼팔/오른팔 블록을 각 역할에 라우팅한다.

    x: (batch, ..., action_dim >= 2 * arm_dim). actuator_is_first: (batch,) bool — 왼팔 블록
    (채널 [0:arm_dim])이 해당 샘플의 Actuator이면 True.
    """
    left = x[..., :arm_dim]
    right = x[..., arm_dim : 2 * arm_dim]
    mask = actuator_is_first.view(-1, *([1] * (x.ndim - 1)))
    actuator = torch.where(mask, left, right)
    stabilizer = torch.where(mask, right, left)
    return stabilizer, actuator


def combine_by_role(stabilizer: Tensor, actuator: Tensor, actuator_is_first: Tensor) -> tuple[Tensor, Tensor]:
    """`split_by_role`의 역연산: 역할별(stabilizer, actuator) 텐서를 다시 샘플별 (left, right)
    팔 순서로 되돌린다. (left, right)를 반환하며 각각 shape은 (batch, ..., arm_dim)이다."""
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
) -> ARDLossOutput:
    """베이스 flow-matching 회귀 손실에 ARD의 역할별 정규화 항들을 결합한다.

    per_element_loss: (batch, chunk_size, 2 * arm_dim) — 베이스 모델이 계산한 채널별
        flow-matching MSE(u_t, v_t)를, 아직 역할로 라우팅하지 않은 원본 왼팔/오른팔 레이아웃
        그대로 넘겨받은 것. 여기서 `split_by_role`로 (stabilizer, actuator)로 라우팅한 뒤
        reduce한다. 이 값이 ARD-VLA 계획서의 L_pos 항 역할을 한다: 네트워크가 목표 궤적을
        추적하도록 실제로 학습시키는 신호이므로, 별도의 L1 위치 손실을 다시 계산하지 않고
        이 값을 재사용한다.
    stabilizer_pred / actuator_pred: (batch, chunk_size, arm_dim) — 각 샘플에서 이미 해당
        역할로 라우팅된 채널의 예측 velocity field (`split_by_role` 참고). 아래의 smoothness /
        trajectory 정규화 항에 사용된다. flow-matching 학습은 diffusion 스텝마다 예측
        velocity field만 만들어낼 뿐 완전히 노이즈 제거된 액션 시퀀스를 만들지는 않으므로,
        chunk 구간에 걸친 이 velocity field의 시간적 변화 양상이 "예측된 궤적"에 가장 가까운
        대용값(proxy)이다.
    actuator_is_first: (batch,) bool, `resolve_actuator_is_first`가 반환한 값 — `per_element_loss`를
        `stabilizer_pred`/`actuator_pred`와 동일한 방식으로 라우팅하기 위해 필요하다.
    """
    stab_pos_per_elem, act_pos_per_elem = split_by_role(per_element_loss, arm_dim, actuator_is_first)
    stab_pos_loss = stab_pos_per_elem.mean()
    act_pos_loss = act_pos_per_elem.mean()

    # L_smooth = sum |s_t - s_{t-1}|^2, chunk 구간 내 예측된 stabilizer 궤적에 대한 흔들림 페널티.
    if stabilizer_pred.shape[1] > 1:
        smooth_loss = (stabilizer_pred[:, 1:] - stabilizer_pred[:, :-1]).pow(2).mean()
    else:
        smooth_loss = stabilizer_pred.new_zeros(())

    # L_traj = sum |a_t - 2a_{t-1} + a_{t-2}|^2, actuator 궤적에 대한 2차 스무딩 페널티
    # (정밀한 도구 조작 중 급격한 방향 전환에 불이익을 준다).
    if actuator_pred.shape[1] > 2:
        second_diff = actuator_pred[:, 2:] - 2 * actuator_pred[:, 1:-1] + actuator_pred[:, :-2]
        traj_loss = second_diff.pow(2).mean()
    else:
        traj_loss = actuator_pred.new_zeros(())

    # L_force: 예를 들어 Isaac Sim의 접촉 센서 등에서 얻는 선택적 접촉력/토크 추적 항.
    # 현재 이 레포의 어떤 데이터셋에도 없는 신호라 기본값은 0이다 — 실제 force 신호 연결은
    # 향후 과제다 (ARD-VLA 연구계획서의 "실물 실증" 참고).
    if force_target is not None:
        force_pred = actuator_pred[..., -1]  # (batch, chunk_size): actuator의 마지막 채널을 force 대용값으로 사용
        target = force_target.to(force_pred.dtype)
        if target.ndim == 1:  # (batch,) -> chunk 전체에 동일한 타겟을 브로드캐스트
            target = target.unsqueeze(-1)
        force_loss = (force_pred - target).abs().mean()
    else:
        force_loss = actuator_pred.new_zeros(())

    stab_loss = stab_pos_loss + lambda_smooth * smooth_loss
    act_loss = act_pos_loss + lambda_force * force_loss + lambda_traj * traj_loss
    total = alpha * stab_loss + beta * act_loss

    return ARDLossOutput(
        total=total,
        stabilizer_loss=stab_loss.detach(),
        actuator_loss=act_loss.detach(),
        pos_loss=((stab_pos_loss + act_pos_loss) / 2).detach(),
        smooth_loss=smooth_loss.detach(),
        force_loss=force_loss.detach(),
        traj_loss=traj_loss.detach(),
    )
