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


class GradNormLambdas(nn.Module):
    """GradNorm(Chen et al., 2018)으로 lambda_smooth/force/traj를 학습 중에 자동 조정한다.

    핵심 아이디어: 세 정규화 항(smooth/force/traj)이 "공유 표현"(actuator/stabilizer head
    바로 직전의 `suffix_out` — 액션 전문가 트랜스포머 출력, action_out_proj와 ard_heads가
    둘 다 이 텐서를 입력으로 받는다)에 만드는 그래디언트의 크기(norm)를 서로 균형 잡히게
    맞춘다. 학습 초반보다 유난히 느리게 줄어드는(=상대적으로 여전히 큰) 항일수록 그래디언트
    norm 목표치를 더 크게 잡아서, 해당 lambda가 커지도록 유도한다.

    중요: lambda(`self.weights`)는 반드시 이 클래스가 만드는 `compute_grad_loss()`의 반환값
    (L_grad)으로만 업데이트되어야 한다 — 메인 total loss의 backward로 직접 업데이트되게 두면
    lambda는 그냥 0으로 수렴해버린다(그래야 해당 항의 기여가 사라져서 total이 작아지므로).
    그래서 `compute_ard_losses()`는 total loss를 만들 때 `gradnorm.lambda_*`를 항상
    `.detach()`해서 쓴다 — 실제 파라미터 업데이트는 학습 루프가 별도 옵티마이저로
    `compute_grad_loss()`의 결과를 가지고 수행한다 (train_ard.py 참고). 이 클래스 자체는
    옵티마이저를 갖지 않는다 — `nn.Module.to(device)`가 파라미터의 실제 텐서를 바꿔치기하는데,
    모듈 안에서 미리 만든 옵티마이저는 그 변화를 모르고 옛 텐서를 계속 참조하게 되는 흔한
    버그를 피하기 위함이다.
    """

    def __init__(self, alpha: float = 1.5, init_value: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.task_names = ("smooth", "force", "traj")
        self.weights = nn.Parameter(torch.full((len(self.task_names),), float(init_value)))
        self.register_buffer("initial_losses", torch.zeros(len(self.task_names)))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def lambda_smooth(self) -> Tensor:
        return self.weights[0]

    @property
    def lambda_force(self) -> Tensor:
        return self.weights[1]

    @property
    def lambda_traj(self) -> Tensor:
        return self.weights[2]

    def compute_grad_loss(self, task_losses: list[Tensor], shared_activation: Tensor) -> Tensor:
        """GradNorm 손실(L_grad)을 계산한다 — `self.weights`에 대해서만 미분 가능하도록 만들어졌다
        (학습 루프에서 `grad_loss.backward(inputs=[self.weights], ...)`로 다른 파라미터는 건드리지
        않고 lambda만 업데이트한다).

        task_losses: [smooth_loss, force_loss, traj_loss] — 그래프가 살아있는(detach 안 된) 값.
        shared_activation: 공유 표현(예: suffix_out). requires_grad=True여야 하고, 이 텐서까지
            거슬러 올라가는 계산 그래프가 아직 free되지 않은 상태여야 한다(즉 이 함수를 호출하는
            시점의 forward pass 도중 — retain_graph로 나중에 메인 loss.backward()도 그래프를
            재사용할 수 있게 해줘야 한다).
        """
        task_losses = list(task_losses)
        if len(task_losses) != len(self.weights):
            raise ValueError(
                f"task_losses는 {len(self.weights)}개(smooth/force/traj)여야 합니다. 받은 개수: {len(task_losses)}"
            )

        if not bool(self.initialized):
            self.initial_losses = torch.stack([loss.detach() for loss in task_losses])
            self.initialized.fill_(True)

        grad_norms = []
        for weight, loss in zip(self.weights, task_losses, strict=True):
            weighted = weight * loss
            # allow_unused=True: force_target이 없으면 force_loss는 shared_activation과 연결되지
            # 않은 상수 0(actuator_pred.new_zeros(()))이라 그래프에 아예 안 잡힌다 — 그럴 때
            # autograd.grad는 기본적으로 에러를 내므로, "그 항은 그래디언트가 0"으로 명시적으로
            # 처리한다 (실제로 그 항이 shared_activation에 아무 영향을 안 준다는 뜻이므로 맞는 처리).
            (grad,) = torch.autograd.grad(
                weighted, shared_activation, retain_graph=True, create_graph=True, allow_unused=True
            )
            if grad is None:
                grad = torch.zeros_like(shared_activation)
            grad_norms.append(grad.norm(2))
        grad_norms = torch.stack(grad_norms)  # self.weights에 대해 미분 가능

        with torch.no_grad():
            current_losses = torch.stack([loss.detach() for loss in task_losses])
            loss_ratios = current_losses / self.initial_losses.clamp_min(1e-8)
            inverse_train_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
            target_grad_norms = grad_norms.mean().detach() * inverse_train_rates.pow(self.alpha)

        return (grad_norms - target_grad_norms).abs().sum()

    def renormalize(self) -> None:
        """GradNorm 논문의 표준 스텝: lambda 업데이트 뒤 합이 항상 태스크 개수(=3)가 되도록
        재정규화한다 — 안 그러면 옵티마이저가 그냥 전부 줄여버려서 L_grad를 트리비얼하게
        낮출 수 있다. `torch.optim.Optimizer.step()` 직후 학습 루프에서 호출해야 한다."""
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            self.weights.mul_(len(self.weights) / self.weights.sum())


@dataclass
class ARDLossOutput:
    total: Tensor
    stabilizer_loss: Tensor
    actuator_loss: Tensor
    pos_loss: Tensor
    smooth_loss: Tensor
    force_loss: Tensor
    traj_loss: Tensor
    grad_loss: Tensor | None = None  # GradNorm 활성화 시에만 채워짐 (ARD-VLA 학습 루프가 별도로 backward)


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
    gradnorm: GradNormLambdas | None = None,
    shared_activation: Tensor | None = None,
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
    gradnorm / shared_activation: 둘 다 주어지면 `lambda_smooth`/`lambda_force`/`lambda_traj`
        인자 대신 `gradnorm.lambda_*`(GradNorm으로 학습되는 값)를 쓴다. `gradnorm`만 주고
        `shared_activation`을 안 주면 에러 — GradNorm 그래디언트 norm 계산에 반드시 필요하다.
    """
    if gradnorm is not None and shared_activation is None:
        raise ValueError("gradnorm을 쓰려면 shared_activation(예: suffix_out)도 같이 넘겨야 합니다.")
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

    grad_loss = None
    if gradnorm is not None:
        # GradNorm이 학습한 lambda로 total loss를 만든다 — 반드시 detach해서 쓴다: 안 그러면
        # total.backward()가 lambda 파라미터에도 직접 그래디언트를 흘려서(해당 항의 손실을
        # 줄이려고 lambda를 그냥 0으로 밀어버리는 방향), GradNorm의 "그래디언트 norm을
        # 맞춘다"는 목적과 별개로 lambda가 오염된다. 진짜 업데이트는 아래 compute_grad_loss()가
        # 만드는 grad_loss로만 (학습 루프가 별도 옵티마이저로) 수행되어야 한다.
        used_lambda_smooth = gradnorm.lambda_smooth.detach()
        used_lambda_force = gradnorm.lambda_force.detach()
        used_lambda_traj = gradnorm.lambda_traj.detach()
        grad_loss = gradnorm.compute_grad_loss([smooth_loss, force_loss, traj_loss], shared_activation)
    else:
        used_lambda_smooth, used_lambda_force, used_lambda_traj = lambda_smooth, lambda_force, lambda_traj

    stab_loss = stab_pos_loss + used_lambda_smooth * smooth_loss
    act_loss = act_pos_loss + used_lambda_force * force_loss + used_lambda_traj * traj_loss
    total = alpha * stab_loss + beta * act_loss

    return ARDLossOutput(
        total=total,
        stabilizer_loss=stab_loss.detach(),
        actuator_loss=act_loss.detach(),
        pos_loss=((stab_pos_loss + act_pos_loss) / 2).detach(),
        smooth_loss=smooth_loss.detach(),
        force_loss=force_loss.detach(),
        traj_loss=traj_loss.detach(),
        grad_loss=grad_loss,
    )
