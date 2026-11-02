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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# 데이터셋/환경이 선택적으로 제공할 수 있는 배치 키.
ARD_FORCE_TARGET = "ard_force_target"  # float, shape (batch,) 또는 (batch, chunk_size): 목표 접촉력/토크
ARD_JOINT_TORQUE = "observation.joint_torque"  # float, shape (batch, torque_dim): 관절 토크 관측값.
# `config.ard_use_joint_torque=True`면 `SmolVLAPolicy.prepare_state()`가 이 값을 `observation.state`
# 뒤에 concat해서 `state_proj`에 넘긴다 — force_target(손실 supervision 타겟)과는 완전히 다른
# 개념이다: 이건 "모델이 보는 입력"이고, force_target은 "ForceHead 출력이 맞혀야 할 정답"이다.


class BridgeAttention(nn.Module):
    """VLA-Adapter(Wang et al., 2025)식 Bridge Attention.

    기존 ARD head는 action expert 트랜스포머의 마지막 지점 출력(`suffix_out`) 하나만 조건으로
    받는다. Bridge Attention은 여기에 더해 SmolLM2 백본의 **여러 중간 레이어**(초반/중반/후반 등
    서로 다른 깊이) hidden state를 KV로, `suffix_out`을 Q로 하는 cross-attention을 계산해서
    "백본이 여러 깊이에서 만든 표현"을 한 번에 조건으로 주입한다.

    출력은 `tanh(gate)`로 스케일되는 학습 가능한 스칼라 게이트를 거친다 — `gate`는 0으로
    초기화되므로(`tanh(0) = 0`) 학습 초반에는 이 브랜치가 아무 영향도 주지 않는다
    (LLaMA-Adapter의 zero-init attention과 같은 아이디어). 기존 `AsymmetricResidualHeads`가
    마지막 레이어를 0으로 초기화해 "처음엔 항등"으로 시작하는 것과 같은 설계 원칙을 그대로
    따른 것이다 — `use_bridge_attention=True`로 켜도 학습 시작 시점의 모델 출력은 꺼져 있을
    때와 완전히 동일하다.

    주의 — 이 구현은 VLA-Adapter 논문을 그대로 옮긴 게 아니라, 사용자가 요청한 설계(여러 레이어
    특징 + cross-attention 조건 주입 + tanh(gate) 스케일)를 직접 구현한 것이다. 특히 "여러
    레이어의 특징"을 어떻게 KV로 합칠지는 논문에 정확히 명시되지 않은 구현 세부사항이라, 각
    레이어의 prefix 토큰 시퀀스를 (레이어를 구분하는 학습 가능한 임베딩을 더해서) 그대로
    이어붙이는 방식을 택했다 — 토큰 단위의 세밀한 정보(예: 어느 이미지 패치/언어 토큰인지)를
    레이어별로 유지하면서도, 레이어 수(보통 3~4개)만큼만 KV 길이가 늘어나 계산 비용이 과하게
    커지지 않는다.
    """

    def __init__(
        self,
        expert_hidden_size: int,
        vlm_hidden_size: int,
        num_bridge_layers: int,
        num_heads: int = 4,
    ):
        super().__init__()
        if expert_hidden_size % num_heads != 0:
            raise ValueError(
                f"expert_hidden_size({expert_hidden_size})는 num_heads({num_heads})로 나누어 "
                f"떨어져야 합니다 (멀티헤드로 쪼갤 수 있어야 함)."
            )
        self.num_heads = num_heads
        self.head_dim = expert_hidden_size // num_heads
        # 레이어별로 "어느 깊이에서 왔는지" 구분하는 학습 가능한 임베딩 — 0-init이라 gate가
        # 열리기 전까지는(그리고 열린 직후 당분간도) 레이어 간에 인위적인 차이를 만들지 않는다.
        self.layer_embed = nn.Parameter(torch.zeros(num_bridge_layers, vlm_hidden_size))
        self.q_proj = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.k_proj = nn.Linear(vlm_hidden_size, expert_hidden_size)
        self.v_proj = nn.Linear(vlm_hidden_size, expert_hidden_size)
        self.out_proj = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)

    def forward(self, query: Tensor, kv_layers: list[Tensor]) -> Tensor:
        """query: (batch, chunk_size, expert_hidden_size) — 보통 `suffix_out`.
        kv_layers: 길이 num_bridge_layers인 리스트, 각 원소는
        (batch, prefix_len, vlm_hidden_size) — SmolLM2 백본의 서로 다른 레이어에서 뽑은
        prefix(이미지+언어) hidden state (`smolvlm_with_expert.py`의
        `SmolVLMWithExpertModel.forward(collect_layer_indices=...)` 참고).

        반환: (batch, chunk_size, expert_hidden_size) — `query`에 그대로 더해서 쓰면 되도록 이미
        `tanh(gate)`로 스케일된 값이다.
        """
        if len(kv_layers) != self.layer_embed.shape[0]:
            raise ValueError(
                f"kv_layers 길이({len(kv_layers)})가 BridgeAttention이 초기화된 "
                f"num_bridge_layers({self.layer_embed.shape[0]})와 다릅니다."
            )
        batch = query.shape[0]
        kv_input = torch.cat(
            [feat + self.layer_embed[i] for i, feat in enumerate(kv_layers)], dim=1
        )  # (batch, num_bridge_layers * prefix_len, vlm_hidden_size)

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(kv_input))
        v = self._split_heads(self.v_proj(kv_input))

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # (batch, heads, chunk, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(batch, -1, self.num_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)
        return torch.tanh(self.gate) * attn_out


def resolve_bridge_layer_indices(num_vlm_layers: int, requested: list[int] | None) -> list[int]:
    """Bridge Attention이 SmolLM2 백본의 어느(트림된) 레이어들에서 조건 특징을 뽑을지 결정한다.

    requested가 주어지면 그대로 쓰고(범위/중복만 검증), None이면 자동으로 4개 지점
    (1/4, 1/2, 3/4, 마지막)을 골라 초반/중반/후반/마지막 깊이를 대표하게 한다 —
    "중간 레이어 + 마지막 레이어" 요청을 SmolVLA가 실제로 쓰는(트림된) 레이어 수에 맞게 구현한
    기본값이다. 인덱스는 `num_vlm_layers`(원본 32개가 아니라 SmolVLA가 실제로 쓰는, 보통 16개)
    기준이다 — layer_importance.py가 분석하는 원본 레이어 인덱싱과는 다르다는 점에 주의.
    """
    if requested is not None:
        if len(requested) == 0:
            raise ValueError("ard_bridge_layer_indices가 비어 있습니다 — 최소 1개 이상의 레이어 인덱스가 필요합니다.")
        if len(set(requested)) != len(requested):
            raise ValueError(f"ard_bridge_layer_indices에 중복된 인덱스가 있습니다: {requested}")
        if any(i < 0 or i >= num_vlm_layers for i in requested):
            raise ValueError(
                f"ard_bridge_layer_indices는 0 이상 {num_vlm_layers - 1} 이하여야 합니다 "
                f"(트림된 VLM 레이어 수: {num_vlm_layers}). 받은 값: {requested}"
            )
        return sorted(requested)
    if num_vlm_layers < 4:
        return list(range(num_vlm_layers))
    quarter = max(num_vlm_layers // 4, 1)
    return sorted({quarter, num_vlm_layers // 2, (3 * num_vlm_layers) // 4, num_vlm_layers - 1})


class ForceHead(nn.Module):
    """`suffix_out`(액션 전문가 트랜스포머의 projection 이전 출력)으로부터 스칼라 접촉력/토크
    추정치를 예측하는 보조(auxiliary) MLP head.

    기존 설계는 `actuator_traj_pred[..., -1]`(actuator 블록의 마지막 "액션" 채널, 즉 보통
    그리퍼/마지막 관절의 속도 또는 x0_hat 기반 위치 추정치)을 힘의 대용값으로 재활용했는데,
    이건 의미가 맞지 않는다 — 그 채널은 실제로는 관절 위치/속도 채널이지 힘 센서 신호가
    아니다. `ard_use_force_head=True`면 이 클래스가 `suffix_out`에서 별도의 전용 스칼라
    출력을 직접 예측하도록 분리한다.

    중요 — 이 head의 출력은 액션 채널에 더해지지 않는 순수 보조(auxiliary) 출력이다:
    `compute_ard_losses`의 `force_loss` 계산에만 쓰이고, `v_t`/최종 액션 예측에는 전혀
    관여하지 않는다. 그래서 추론 경로(`sample_actions`/`denoise_step`)에서는 아예 호출되지
    않는다 — 학습 때(`VLAFlowMatching.forward`)만 쓰인다.

    GradNorm과의 연결: `GradNormLambdas.compute_grad_loss`는 `shared_activation`(=`suffix_out`)에
    대한 각 손실 항의 그래디언트 norm을 잰다. 이 head의 입력이 정확히 `suffix_out`이므로,
    `force_loss`가 `suffix_out`까지 역전파로 이어지는 계산 그래프를 자연스럽게 갖는다 —
    과거 설계(`actuator_traj_pred` 경유)도 결국 `suffix_out`까지 이어지긴 했지만, 의미가
    맞지 않는 채널을 거쳐서였다. 이 클래스를 쓰면 그 경로가 "진짜 force 예측"을 거치게 된다.

    주의 — 기존 ARD head(`AsymmetricResidualHeads`)와 달리 **zero-init을 쓰지 않는다**.
    `AsymmetricResidualHeads`의 zero-init은 "사전학습된 공유 projection 위에 처음엔 아무
    영향도 주지 않는 항등(identity)으로 시작한다"는 구조적 이유(그 출력이 `v_t`에 더해지기
    때문)가 있지만, `ForceHead`의 출력은 애초에 어디에도 더해지지 않는 독립 보조 출력이라
    "항등"이라는 개념 자체가 없다. 오히려 마지막 레이어를 0으로 초기화하면 `suffix_out`까지
    역전파되는 그래디언트가 초기화 시점에 정확히 0이 되어(zero 가중치 행렬을 통과하므로),
    GradNorm이 학습 첫 스텝부터 force 항의 그래디언트 norm을 제대로 못 보는 문제가 생긴다 —
    그래서 PyTorch 기본 초기화(Kaiming uniform)를 그대로 쓴다.
    """

    def __init__(self, expert_hidden_size: int, mlp_hidden_dim: int | None = None):
        super().__init__()
        mlp_hidden_dim = mlp_hidden_dim or max(expert_hidden_size // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(expert_hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, suffix_out: Tensor) -> Tensor:
        """suffix_out: (batch, chunk_size, expert_hidden_size) -> (batch, chunk_size)."""
        return self.net(suffix_out).squeeze(-1)


class AsymmetricResidualHeads(nn.Module):
    """두 팔 각자의 채널에 대해 flow-matching 공유 출력을 보정하는 역할별 residual head.

    베이스 모델은 이미 하나의 공유 선형 projection(`action_out_proj`)으로 `max_action_dim` 전체
    길이의 예측을 만들어낸다. 이 head들은 전체 액션 전문가(action expert) 트랜스포머를 통째로
    복제하지 않으면서도, Stabilizer/Actuator 채널 블록 위에 역할 특화 보정을 추가로 얹어 각
    역할에 전용 학습 용량을 부여한다.

    `use_bridge_attention=True`면 각 head 앞에 전용 `BridgeAttention`을 하나씩 붙여서(그래서
    Stabilizer/Actuator가 백본 특징 중 서로 다른 부분에 주목하도록 독립적으로 학습될 수 있다),
    `suffix_features`(마지막 지점 조건) 하나만이 아니라 `bridge_kv_layers`(백본 여러 레이어
    조건)까지 함께 반영한다. 꺼져 있으면(기본값) 기존과 완전히 동일하게 동작한다.
    """

    def __init__(
        self,
        expert_hidden_size: int,
        arm_dim: int,
        mlp_hidden_dim: int | None = None,
        use_bridge_attention: bool = False,
        vlm_hidden_size: int | None = None,
        num_bridge_layers: int = 4,
        bridge_num_heads: int = 4,
    ):
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

        self.use_bridge_attention = use_bridge_attention
        self.stabilizer_bridge: BridgeAttention | None = None
        self.actuator_bridge: BridgeAttention | None = None
        if use_bridge_attention:
            if vlm_hidden_size is None:
                raise ValueError("use_bridge_attention=True면 vlm_hidden_size를 반드시 지정해야 합니다.")
            self.stabilizer_bridge = BridgeAttention(expert_hidden_size, vlm_hidden_size, num_bridge_layers, bridge_num_heads)
            self.actuator_bridge = BridgeAttention(expert_hidden_size, vlm_hidden_size, num_bridge_layers, bridge_num_heads)

    def forward(self, suffix_features: Tensor, bridge_kv_layers: list[Tensor] | None = None) -> tuple[Tensor, Tensor]:
        """suffix_features: (batch, chunk_size, expert_hidden_size), 액션 전문가의 projection 이전
        트랜스포머 출력. bridge_kv_layers: `use_bridge_attention=True`일 때, SmolLM2 백본 여러
        레이어에서 뽑은 prefix hidden state 리스트(`BridgeAttention.forward` 참고) — 안 주거나
        `use_bridge_attention=False`면 기존과 동일하게 `suffix_features`만 사용한다.
        (stabilizer_residual, actuator_residual)을 반환하며 각각 shape은
        (batch, chunk_size, arm_dim)이다."""
        if self.use_bridge_attention and bridge_kv_layers is not None:
            stabilizer_input = suffix_features + self.stabilizer_bridge(suffix_features, bridge_kv_layers)
            actuator_input = suffix_features + self.actuator_bridge(suffix_features, bridge_kv_layers)
        else:
            stabilizer_input = actuator_input = suffix_features
        return self.stabilizer_head(stabilizer_input), self.actuator_head(actuator_input)


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
        # 과거엔 스칼라 하나(전역 "초기화됐는지")였다 — force처럼 학습 중간에 꺼졌다 켜질 수
        # 있는 태스크는 "그 태스크가 처음 active가 된 시점"의 손실을 L_i(0)으로 따로 기록해야
        # 해서, 태스크별(smooth/force/traj) bool 벡터로 바꿨다.
        self.register_buffer("initialized", torch.zeros(len(self.task_names), dtype=torch.bool))

    @property
    def lambda_smooth(self) -> Tensor:
        return self.weights[0]

    @property
    def lambda_force(self) -> Tensor:
        return self.weights[1]

    @property
    def lambda_traj(self) -> Tensor:
        return self.weights[2]

    def compute_grad_loss(
        self, task_losses: list[Tensor], shared_activation: Tensor, active: Sequence[bool] | Tensor
    ) -> Tensor:
        """GradNorm 손실(L_grad)을 계산한다 — `self.weights`에 대해서만 미분 가능하도록 만들어졌다
        (학습 루프에서 `grad_loss.backward(inputs=[self.weights], ...)`로 다른 파라미터는 건드리지
        않고 lambda만 업데이트한다).

        task_losses: [smooth_loss, force_loss, traj_loss] — 그래프가 살아있는(detach 안 된) 값.
        shared_activation: 공유 표현(예: suffix_out). requires_grad=True여야 하고, 이 텐서까지
            거슬러 올라가는 계산 그래프가 아직 free되지 않은 상태여야 한다(즉 이 함수를 호출하는
            시점의 forward pass 도중 — retain_graph로 나중에 메인 loss.backward()도 그래프를
            재사용할 수 있게 해줘야 한다).
        active: [smooth_active, force_active, traj_active] — 그 태스크가 이번 스텝에 실제로 의미
            있는 신호인지(예: force_active = force_target is not None and force_pred is not None).
            비활성 태스크는 그래디언트 norm 계산/Ḡ 평균/r_i 정규화에서 전부 제외한다 — 예전처럼
            "비활성 태스크를 0으로 때워서 active 집합에 섞어 넣으면" Ḡ가 실제보다 작게, 나머지
            활성 태스크의 r_i가 실제보다 크게 편향된다(force_loss가 데이터셋에 없어 항상 0인
            경우가 정확히 이 상황이었다). `active=False`인 태스크의 lambda는 이 스텝에서 전혀
            갱신되지 않는다(`apply_update()` 참고).
        """
        task_losses = list(task_losses)
        if len(task_losses) != len(self.weights):
            raise ValueError(
                f"task_losses는 {len(self.weights)}개(smooth/force/traj)여야 합니다. 받은 개수: {len(task_losses)}"
            )
        active_t = torch.as_tensor(list(active), dtype=torch.bool, device=self.weights.device)
        if active_t.shape != self.weights.shape:
            raise ValueError(
                f"active는 {len(self.weights)}개(smooth/force/traj)여야 합니다. 받은 개수: {active_t.numel()}"
            )

        # 비활성이었다가(또는 한 번도 활성화된 적 없다가) 이번 스텝에 처음 active가 된 태스크만
        # L_i(0)을 기록한다 — 기존에 이미 초기화된 태스크는 덮어쓰지 않는다.
        newly_active = active_t & ~self.initialized
        if bool(newly_active.any()):
            with torch.no_grad():
                for i, is_new in enumerate(newly_active.tolist()):
                    if is_new:
                        self.initial_losses[i] = task_losses[i].detach()
                self.initialized = self.initialized | newly_active

        active_idx = active_t.nonzero(as_tuple=True)[0].tolist()
        if not active_idx:
            # 이번 스텝에 활성 태스크가 하나도 없음 — 업데이트할 신호 자체가 없다. weights와
            # 연결되지 않은(그래서 backward해도 아무 lambda도 안 건드리는) 0을 반환한다.
            return shared_activation.new_zeros(())

        grad_norms = []
        for i in active_idx:
            weighted = self.weights[i] * task_losses[i]
            (grad,) = torch.autograd.grad(
                weighted, shared_activation, retain_graph=True, create_graph=True, allow_unused=True
            )
            if grad is None:
                # active=True로 넘어왔다는 건 "이 태스크가 이번 스텝에 실제로 의미 있다"는
                # 호출자의 보장인데, 실제로는 shared_activation과 그래프가 끊겨 있다는 뜻 —
                # 조용히 0으로 때우면 과거처럼 Ḡ/r_i가 소리 없이 편향된다. active 플래그 자체가
                # 실제 상태와 어긋난 호출자 버그일 가능성이 높으므로 바로 에러를 낸다.
                raise RuntimeError(
                    f"GradNorm 태스크 '{self.task_names[i]}'가 active=True로 들어왔지만 손실이 "
                    "shared_activation 계산 그래프와 연결되어 있지 않습니다(gradient=None). "
                    "active 플래그가 실제 손실 상태와 어긋났을 가능성이 큽니다(예: force_target이 "
                    "없는데 force를 active로 표시)."
                )
            grad_norms.append(grad.norm(2))
        grad_norms = torch.stack(grad_norms)  # self.weights[active_idx]에 대해 미분 가능

        with torch.no_grad():
            current_losses = torch.stack([task_losses[i].detach() for i in active_idx])
            initial_losses = self.initial_losses[active_idx]
            loss_ratios = current_losses / initial_losses.clamp_min(1e-8)
            inverse_train_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
            target_grad_norms = grad_norms.mean().detach() * inverse_train_rates.pow(self.alpha)

        return (grad_norms - target_grad_norms).abs().sum()

    def apply_update(self, optimizer: torch.optim.Optimizer, active: Sequence[bool] | Tensor) -> None:
        """`compute_grad_loss(...)`의 결과로 `grad_loss.backward(inputs=[self.weights], ...)`까지
        끝난 뒤, `optimizer.step()` 대신 이 메서드를 호출한다.

        `optimizer.step()`은 `self.weights` 전체(3개)에 대해 호출하므로(Adam은 모멘텀을
        갖고 있어서, 이번 스텝 gradient가 정확히 0이어도 과거 모멘텀 때문에 값이 계속
        흔들릴 수 있다), active=False인 인덱스는 **step 이전 값으로 복원**해서 이번 스텝에
        정말로 전혀 변하지 않도록 만든다. 그 다음 active 태스크끼리만 `clamp(min=1e-3)` +
        합이 "활성 태스크 수"가 되도록 재정규화한다(GradNorm 논문의 표준 "합 보존" 스텝을
        비활성 태스크가 끼지 않은 부분집합에 적용한 것) — 안 그러면 옵티마이저가 전체 lambda를
        그냥 줄여서 L_grad를 트리비얼하게 낮출 수 있다."""
        active_t = torch.as_tensor(list(active), dtype=torch.bool, device=self.weights.device)
        if active_t.shape != self.weights.shape:
            raise ValueError(
                f"active는 {len(self.weights)}개(smooth/force/traj)여야 합니다. 받은 개수: {active_t.numel()}"
            )
        with torch.no_grad():
            pre_step = self.weights.detach().clone()
        optimizer.step()
        with torch.no_grad():
            self.weights[~active_t] = pre_step[~active_t]
            n_active = int(active_t.sum().item())
            if n_active > 0:
                active_w = self.weights[active_t].clamp(min=1e-3)
                self.weights[active_t] = active_w * (n_active / active_w.sum())


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
    # GradNorm 활성화 시에만 채워짐: (smooth_active, force_active, traj_active). 학습 루프가
    # optimizer.step() 뒤 `GradNormLambdas.apply_update(optimizer, active=...)`에 그대로 넘겨서
    # 비활성 태스크의 lambda를 이번 스텝에 전혀 건드리지 않게 하는 데 쓴다.
    gradnorm_active: tuple[bool, bool, bool] | None = None


def _reduce_reg_loss(sq_diff: Tensor, time_weights: Tensor | None) -> Tensor:
    """smooth_loss/traj_loss의 제곱차(`sq_diff`, shape (batch, ..., arm_dim))를 스칼라로
    줄인다. `time_weights`가 None이면(기본, `ard_reg_time_weighting="none"`) 기존과 완전히
    동일하게 전체 원소 평균(`sq_diff.mean()`)을 쓴다 — 이 분기는 과거 코드와 bit-for-bit
    동일해야 하므로 절대 건드리지 않는다. `time_weights`(shape (batch,))가 주어지면 샘플별
    평균을 먼저 낸 뒤 그 가중 평균을 쓴다(`ard_reg_time_weighting="one_minus_t"`가 여기로
    `1 - time`을 넘긴다 — modeling_smolvla.py의 VLAFlowMatching.forward 참고)."""
    if time_weights is None:
        return sq_diff.mean()
    per_sample = sq_diff.mean(dim=tuple(range(1, sq_diff.ndim)))
    return (per_sample * time_weights).mean()


def _exclude_gripper(traj_pred: Tensor, gripper_dim: int) -> Tensor:
    """smooth_loss/traj_loss의 1차/2차 차분을 계산하기 전에, 각 팔 블록의 마지막
    `gripper_dim`개 채널(그리퍼)을 제외한다.

    그리퍼는 "열림/닫힘"에 가까운 bang-bang성 신호라, 관절 채널과 달리 빠르게 전환되는 것이
    정상적인 동작이다 — 관절 궤적에 적용하는 1차(smooth)/2차(traj) 차분 벌점을 그리퍼 채널에
    그대로 걸면, 정상적으로 빨리 열고 닫아야 하는 상황에서도 "느리게 움직이라"는 잘못된 신호를
    준다. `gripper_dim=0`(기본값)이면 과거와 완전히 동일하게 전체 채널에 벌점을 건다 — 이
    분기는 bit-for-bit 동일성을 위해 절대 건드리지 않는다.

    L_pos(기본 flow-matching 회귀 손실)는 이 함수와 무관하게 전체 채널(그리퍼 포함)에 그대로
    걸린다 — `compute_ard_losses`/`compute_symmetric_losses`의 `per_element_loss` 처리 참고.
    즉 그리퍼 채널은 "L_pos만 받고 smooth/traj 추가 벌점은 받지 않는다"는 설계가 된다.
    """
    if gripper_dim <= 0:
        return traj_pred
    return traj_pred[..., : traj_pred.shape[-1] - gripper_dim]


def compute_timestep_norm_sigma(
    a: float | list[float],
    s: float | list[float],
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    pad_to: int | None = None,
) -> Tensor:
    """`config.per_timestep_norm`용 σ(t) = a + s*sqrt(t), t=0..chunk_size-1.

    `a`/`s`가 스칼라면 반환 shape (chunk_size, 1)이라 액션의 모든 채널에 동일하게 broadcast된다.
    리스트(길이 D)면 (chunk_size, D)라서 채널마다 다른 σ(t)를 쓴다. `pad_to`가 D보다 크면(실제
    액션 차원만큼만 피팅했는데 actions 텐서가 max_action_dim으로 패딩돼 있는 경우) 나머지
    채널은 σ(t)=1(identity, a=1·s=0)로 채워서 broadcast가 깨지지 않게 한다. 호출자
    (modeling_smolvla.py)가 `actions / sigma`(정규화) / `x_t * sigma`(역정규화) 형태로
    (batch, chunk_size, action_dim)에 바로 곱/나눗셈하면 된다(LeHome 우승 보고서, arXiv
    2606.27163 참고)."""
    t = torch.arange(chunk_size, device=device, dtype=dtype)
    sqrt_t = torch.sqrt(t)
    a_t = torch.as_tensor(a, device=device, dtype=dtype).reshape(-1)
    s_t = torch.as_tensor(s, device=device, dtype=dtype).reshape(-1)
    if pad_to is not None and pad_to > a_t.shape[0]:
        pad = pad_to - a_t.shape[0]
        a_t = torch.cat([a_t, torch.ones(pad, device=device, dtype=dtype)])
        s_t = torch.cat([s_t, torch.zeros(pad, device=device, dtype=dtype)])
    return a_t[None, :] + s_t[None, :] * sqrt_t[:, None]  # (chunk_size, len(a) or 1)


def apply_timestep_norm(actions: Tensor, sigma: Tensor) -> Tensor:
    """actions: (batch, chunk_size, D). sigma: compute_timestep_norm_sigma()의 (chunk_size, 1|D)."""
    return actions / sigma[None, : actions.shape[1], :]


def invert_timestep_norm(scaled_actions: Tensor, sigma: Tensor) -> Tensor:
    """apply_timestep_norm의 역연산."""
    return scaled_actions * sigma[None, : scaled_actions.shape[1], :]


def compute_ard_losses(
    per_element_loss: Tensor,
    stabilizer_traj_pred: Tensor,
    actuator_traj_pred: Tensor,
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
    reg_time_weights: Tensor | None = None,
    force_pred: Tensor | None = None,
    gripper_dim: int = 0,
) -> ARDLossOutput:
    """베이스 flow-matching 회귀 손실에 ARD의 역할별 정규화 항들을 결합한다.

    per_element_loss: (batch, chunk_size, 2 * arm_dim) — 베이스 모델이 계산한 채널별
        flow-matching MSE(u_t, v_t)를, 아직 역할로 라우팅하지 않은 원본 왼팔/오른팔 레이아웃
        그대로 넘겨받은 것. 여기서 `split_by_role`로 (stabilizer, actuator)로 라우팅한 뒤
        reduce한다. 이 값이 ARD-VLA 계획서의 L_pos 항 역할을 한다: 네트워크가 목표 궤적을
        추적하도록 실제로 학습시키는 신호이므로, 별도의 L1 위치 손실을 다시 계산하지 않고
        이 값을 재사용한다.
    stabilizer_traj_pred / actuator_traj_pred: (batch, chunk_size, arm_dim) — 각 샘플에서 이미
        해당 역할로 라우팅된 채널의, **노이즈 제거된 액션 궤적 추정치** `x0_hat`(`split_by_role`
        참고). 아래의 smoothness / trajectory 정규화 항에 사용된다.

        주의 — 과거 버전에서는 여기에 예측 velocity field(`v_t`) 자체를 넘겼는데, 이건
        개념적으로 틀렸다: flow-matching 목표 velocity `u_t = noise - actions`는 매 타임스텝
        독립적으로 샘플된 `noise` 때문에 시간축으로 원래 거칠다(스무스해야 할 이유가 없다).
        `v_t`에 1차/2차 차분 스무딩 벌점을 주면 "이 velocity는 매끄러워야 한다"고 강요하는
        셈이라, flow-matching 회귀 목표(정답 `u_t`를 맞히는 것) 자체와 정면으로 충돌한다.
        `x0_hat = x_t - t * v_t`(직선 보간 `x_t = t*noise + (1-t)*actions`이므로, `v_t == u_t`일
        때 `x0_hat`은 `actions`와 정확히 같아진다)는 velocity가 아니라 "현재 시점에서 모델이
        추정하는 노이즈 제거된 액션 그 자체"라서, 이 값의 시간적 변화(1차/2차 차분)에 벌점을
        주는 것이 smooth_loss/traj_loss가 원래 의도한 "예측 궤적이 매끄럽다"는 개념과 실제로
        들어맞는다. `x0_hat` 계산 자체는 이 함수가 아니라 `modeling_smolvla.py`의
        `VLAFlowMatching.forward`에서 이뤄진다 — 이 함수는 이미 계산되어 역할별로 라우팅된
        값만 받는다.
    actuator_is_first: (batch,) bool, `resolve_actuator_is_first`가 반환한 값 — `per_element_loss`를
        `stabilizer_traj_pred`/`actuator_traj_pred`와 동일한 방식으로 라우팅하기 위해 필요하다.
    gradnorm / shared_activation: 둘 다 주어지면 `lambda_smooth`/`lambda_force`/`lambda_traj`
        인자 대신 `gradnorm.lambda_*`(GradNorm으로 학습되는 값)를 쓴다. `gradnorm`만 주고
        `shared_activation`을 안 주면 에러 — GradNorm 그래디언트 norm 계산에 반드시 필요하다.
    reg_time_weights: (batch,) 또는 None(기본). `config.ard_reg_time_weighting="one_minus_t"`일
        때 `modeling_smolvla.py`가 `1 - time`을 넘긴다 — `x0_hat`은 flow-matching 타임스텝
        `t`가 클수록(거의 순수 노이즈에 가까운 샘플일수록) 오차/분산이 커지는 경향이 있어서,
        smooth_loss/traj_loss에서 그런 샘플의 영향을 줄이는 용도다. None이면(기본) 기존과
        완전히 동일하게 가중치 없는 평균을 쓴다(`_reduce_reg_loss` 참고).
    force_pred: (batch, chunk_size) 또는 None. `config.ard_use_force_head=True`일 때
        `modeling_smolvla.py`가 `ForceHead(suffix_out)`의 출력을 넘긴다 — `force_target`과
        비교해 `force_loss`를 만드는 데 쓰인다. `force_target`은 주어졌는데 `force_pred`가
        None이면(즉 `ard_use_force_head=False`인데 `force_target`이 들어온 경우) 힘을 예측할
        방법이 없으므로 경고를 한 번 띄우고 `force_loss`를 0으로 처리한다 — 과거처럼 관절
        속도/위치 채널을 대신 쓰지 않는다.
    gripper_dim: 각 팔 블록(`arm_dim`개 채널)의 마지막 `gripper_dim`개를 그리퍼로 간주해서
        smooth_loss/traj_loss 계산에서 제외한다(`_exclude_gripper` 참고). 기본값 0이면 과거와
        완전히 동일하게 전체 채널에 벌점을 건다. `L_pos`(아래 `stab_pos_loss`/`act_pos_loss`)는
        이 인자와 무관하게 항상 전체 채널(그리퍼 포함)에 걸린다 — 즉 그리퍼는 "L_pos만 받고
        smooth/traj 추가 벌점은 받지 않는다".
    """
    if gradnorm is not None and shared_activation is None:
        raise ValueError("gradnorm을 쓰려면 shared_activation(예: suffix_out)도 같이 넘겨야 합니다.")
    stab_pos_per_elem, act_pos_per_elem = split_by_role(per_element_loss, arm_dim, actuator_is_first)
    stab_pos_loss = stab_pos_per_elem.mean()
    act_pos_loss = act_pos_per_elem.mean()

    # L_smooth = sum |s_t - s_{t-1}|^2, chunk 구간 내 예측된(노이즈 제거된) stabilizer 액션
    # 궤적 추정치(x0_hat 기반)에 대한 흔들림 페널티. gripper_dim>0이면 그리퍼 채널은 제외한다.
    smooth_input = _exclude_gripper(stabilizer_traj_pred, gripper_dim)
    if smooth_input.shape[1] > 1:
        smooth_loss = _reduce_reg_loss(
            (smooth_input[:, 1:] - smooth_input[:, :-1]).pow(2), reg_time_weights
        )
    else:
        smooth_loss = stabilizer_traj_pred.new_zeros(())

    # L_traj = sum |a_t - 2a_{t-1} + a_{t-2}|^2, actuator 액션 궤적 추정치에 대한 2차 스무딩
    # 페널티(정밀한 도구 조작 중 급격한 방향 전환에 불이익을 준다). gripper_dim>0이면 그리퍼
    # 채널은 제외한다.
    traj_input = _exclude_gripper(actuator_traj_pred, gripper_dim)
    if traj_input.shape[1] > 2:
        second_diff = traj_input[:, 2:] - 2 * traj_input[:, 1:-1] + traj_input[:, :-2]
        traj_loss = _reduce_reg_loss(second_diff.pow(2), reg_time_weights)
    else:
        traj_loss = actuator_traj_pred.new_zeros(())

    # L_force: 예를 들어 Isaac Sim의 접촉 센서 등에서 얻는 선택적 접촉력/토크 추적 항.
    # 현재 이 레포의 어떤 데이터셋에도 없는 신호라 force_target이 없으면 항상 0이다.
    # force_pred는 ard_use_force_head=True일 때 modeling_smolvla.py가 ForceHead(suffix_out)로
    # 계산해서 넘긴다 — 과거처럼 actuator_traj_pred의 마지막 채널(관절 위치/속도 채널이지
    # 힘 센서 신호가 아니다)을 대신 쓰지 않는다.
    if force_target is not None and force_pred is not None:
        target = force_target.to(force_pred.dtype)
        if target.ndim == 1:  # (batch,) -> chunk 전체에 동일한 타겟을 브로드캐스트
            target = target.unsqueeze(-1)
        force_loss = (force_pred - target).abs().mean()
    elif force_target is not None and force_pred is None:
        # ard_use_force_head=False인데 force_target이 들어온 경우 — 힘을 예측할 방법이 없다.
        # 과거처럼 관절 속도/위치 채널을 대신 쓰지 않고, 경고 후 0으로 처리한다. warnings.warn은
        # 기본 필터("default")가 동일한 (메시지, 위치) 조합을 프로세스당 한 번만 보여주므로,
        # 매 학습 스텝마다 호출돼도 자연스럽게 "한 번만" 출력된다.
        warnings.warn(
            "force_target이 주어졌지만 ard_use_force_head=False라 힘을 예측할 방법이 없습니다 — "
            "force_loss를 0으로 처리합니다. 힘 추적 손실을 쓰려면 ard_use_force_head=True로 "
            "ForceHead를 활성화하세요.",
            stacklevel=2,
        )
        force_loss = actuator_traj_pred.new_zeros(())
    else:
        force_loss = actuator_traj_pred.new_zeros(())

    grad_loss = None
    gradnorm_active = None
    if gradnorm is not None:
        # smooth_loss/traj_loss가 바로 위에서 0으로 폴백하는 조건(shape[1]<=1/<=2)과 force_loss가
        # 0으로 폴백하는 조건(force_target/force_pred 부재)을 그대로 재사용한다 — "이 태스크가
        # 이번 스텝에 실제로 의미 있는 신호인가"의 유일한 판단 기준이 두 군데서 따로 갈리지
        # 않게 하기 위함이다. GradNorm은 이 active 플래그로 비활성 태스크를 Ḡ/r_i/재정규화에서
        # 완전히 제외한다(GradNormLambdas.compute_grad_loss 참고) — 예전에는 force_loss가 데이터셋
        # 부재로 거의 항상 0인데도 이 셋에 섞여 들어가 Ḡ·r_i를 편향시켰었다.
        gradnorm_active = (
            smooth_input.shape[1] > 1,
            force_target is not None and force_pred is not None,
            traj_input.shape[1] > 2,
        )
        # GradNorm이 학습한 lambda로 total loss를 만든다 — 반드시 detach해서 쓴다: 안 그러면
        # total.backward()가 lambda 파라미터에도 직접 그래디언트를 흘려서(해당 항의 손실을
        # 줄이려고 lambda를 그냥 0으로 밀어버리는 방향), GradNorm의 "그래디언트 norm을
        # 맞춘다"는 목적과 별개로 lambda가 오염된다. 진짜 업데이트는 아래 compute_grad_loss()가
        # 만드는 grad_loss로만 (학습 루프가 별도 옵티마이저로) 수행되어야 한다.
        used_lambda_smooth = gradnorm.lambda_smooth.detach()
        used_lambda_force = gradnorm.lambda_force.detach()
        used_lambda_traj = gradnorm.lambda_traj.detach()
        grad_loss = gradnorm.compute_grad_loss(
            [smooth_loss, force_loss, traj_loss], shared_activation, active=gradnorm_active
        )
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
        gradnorm_active=gradnorm_active,
    )


def compute_symmetric_losses(
    per_element_loss: Tensor,
    stabilizer_traj_pred: Tensor,
    actuator_traj_pred: Tensor,
    actuator_is_first: Tensor,
    arm_dim: int,
    lambda_smooth: float,
    lambda_force: float,
    lambda_traj: float,
    force_target: Tensor | None = None,
    gradnorm: GradNormLambdas | None = None,
    shared_activation: Tensor | None = None,
    reg_time_weights: Tensor | None = None,
    force_pred: Tensor | None = None,
    gripper_dim: int = 0,
) -> ARDLossOutput:
    """`ard_symmetric=True`일 때 쓰는 대칭 대조군(ablation) 손실 결합. `compute_ard_losses`와
    달리 ARD의 핵심 가정("Stabilizer는 smooth만, Actuator는 force+traj만") 자체를 빼고, 두
    팔 모두에 똑같은 형태 `L_pos + lambda_smooth*L_smooth + lambda_traj*L_traj`를 적용한다 —
    즉 smooth(1차 차분)와 traj(2차 차분)를 **양쪽 팔 모두**에 건다(ARD처럼 한쪽에만 걸지
    않는다). 호출하는 쪽(`alpha=beta=0.5`는 `configuration_smolvla.py`에서 강제됨)이
    `total = 0.5 * stab_loss + 0.5 * act_loss`로 결합한다고 가정하고, 이 함수 자체도 내부적으로
    0.5/0.5로 결합한다.

    head 구조/파라미터 수를 ARD와 동일하게 유지하는 건 이 함수의 책임이 아니다 — 호출하는 쪽
    (`modeling_smolvla.py`)이 `ard_symmetric` 여부와 무관하게 항상 같은 `AsymmetricResidualHeads`
    인스턴스를 재사용하고, 이 함수는 그 출력(`stabilizer_traj_pred`/`actuator_traj_pred`)을
    어떻게 손실로 결합할지만 바꾼다 — 그래서 두 모드의 파라미터 수가 정확히 같다.

    force 항은 ARD 모드와 동일하게 actuator 쪽에만 유지한다(대칭으로 만들지 않음) — 접촉력/
    토크는 도구를 조작하는 Actuator 팔에만 물리적으로 의미가 있는 신호이고, 작업물을 가만히
    붙잡고 있는 Stabilizer 팔에는 애초에 "힘을 추적한다"는 과제 자체가 성립하지 않는다.
    smooth/traj(궤적의 일반적인 매끄러움)는 양팔 모두에 자연스럽게 의미가 있는 것과 달리
    force는 그렇지 않다고 판단해서, 이 항목만큼은 대칭화하지 않기로 결정했다 — 이건 판단이
    갈릴 수 있는 지점이라 여기 명시해둔다.

    gripper_dim: `compute_ard_losses`와 동일한 의미 — 각 팔 블록의 마지막 `gripper_dim`개
        채널(그리퍼)을 smooth/traj 계산에서 제외한다(양쪽 팔 모두에 동일하게 적용된다, 대칭
        모드답게). 기본값 0이면 과거와 완전히 동일하다.

    다른 인자들의 의미는 `compute_ard_losses`와 동일하다.
    """
    if gradnorm is not None and shared_activation is None:
        raise ValueError("gradnorm을 쓰려면 shared_activation(예: suffix_out)도 같이 넘겨야 합니다.")
    stab_pos_per_elem, act_pos_per_elem = split_by_role(per_element_loss, arm_dim, actuator_is_first)
    stab_pos_loss = stab_pos_per_elem.mean()
    act_pos_loss = act_pos_per_elem.mean()

    def _smooth(traj_pred: Tensor) -> Tensor:
        smooth_input = _exclude_gripper(traj_pred, gripper_dim)
        if smooth_input.shape[1] > 1:
            return _reduce_reg_loss((smooth_input[:, 1:] - smooth_input[:, :-1]).pow(2), reg_time_weights)
        return traj_pred.new_zeros(())

    def _traj(traj_pred: Tensor) -> Tensor:
        traj_input = _exclude_gripper(traj_pred, gripper_dim)
        if traj_input.shape[1] > 2:
            second_diff = traj_input[:, 2:] - 2 * traj_input[:, 1:-1] + traj_input[:, :-2]
            return _reduce_reg_loss(second_diff.pow(2), reg_time_weights)
        return traj_pred.new_zeros(())

    smooth_stab = _smooth(stabilizer_traj_pred)
    smooth_act = _smooth(actuator_traj_pred)
    traj_stab = _traj(stabilizer_traj_pred)
    traj_act = _traj(actuator_traj_pred)

    # force_loss: compute_ard_losses와 완전히 동일한 로직(actuator 쪽에만, force_pred 필요).
    if force_target is not None and force_pred is not None:
        target = force_target.to(force_pred.dtype)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        force_loss = (force_pred - target).abs().mean()
    elif force_target is not None and force_pred is None:
        warnings.warn(
            "force_target이 주어졌지만 ard_use_force_head=False라 힘을 예측할 방법이 없습니다 — "
            "force_loss를 0으로 처리합니다. 힘 추적 손실을 쓰려면 ard_use_force_head=True로 "
            "ForceHead를 활성화하세요.",
            stacklevel=2,
        )
        force_loss = actuator_traj_pred.new_zeros(())
    else:
        force_loss = actuator_traj_pred.new_zeros(())

    grad_loss = None
    gradnorm_active = None
    if gradnorm is not None:
        used_lambda_smooth = gradnorm.lambda_smooth.detach()
        used_lambda_force = gradnorm.lambda_force.detach()
        used_lambda_traj = gradnorm.lambda_traj.detach()
        # _smooth/_traj 내부의 0-폴백 조건(shape[1]<=1/<=2)과 동일한 기준 — gripper 제외는
        # 시간축(shape[1])에 영향이 없으므로 stabilizer_traj_pred.shape[1]을 그대로 쓴다.
        gradnorm_active = (
            stabilizer_traj_pred.shape[1] > 1,
            force_target is not None and force_pred is not None,
            stabilizer_traj_pred.shape[1] > 2,
        )
        # 대칭 모드에서는 smooth/traj가 양팔 각각에 걸리므로, GradNorm에 넘기는 "대표" smooth/traj
        # 손실은 양팔 평균을 쓴다 — GradNormLambdas.task_names는 ("smooth", "force", "traj")
        # 3개 고정이라, ARD 모드와 동일한 인터페이스를 유지하기 위한 선택이다.
        grad_loss = gradnorm.compute_grad_loss(
            [(smooth_stab + smooth_act) / 2, force_loss, (traj_stab + traj_act) / 2],
            shared_activation,
            active=gradnorm_active,
        )
    else:
        used_lambda_smooth, used_lambda_force, used_lambda_traj = lambda_smooth, lambda_force, lambda_traj

    stab_loss = stab_pos_loss + used_lambda_smooth * smooth_stab + used_lambda_traj * traj_stab
    act_loss = (
        act_pos_loss
        + used_lambda_smooth * smooth_act
        + used_lambda_force * force_loss
        + used_lambda_traj * traj_act
    )
    total = 0.5 * stab_loss + 0.5 * act_loss

    return ARDLossOutput(
        total=total,
        stabilizer_loss=stab_loss.detach(),
        actuator_loss=act_loss.detach(),
        pos_loss=((stab_pos_loss + act_pos_loss) / 2).detach(),
        smooth_loss=((smooth_stab + smooth_act) / 2).detach(),
        force_loss=force_loss.detach(),
        traj_loss=((traj_stab + traj_act) / 2).detach(),
        grad_loss=grad_loss,
        gradnorm_active=gradnorm_active,
    )
