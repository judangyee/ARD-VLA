# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.ard import ARD_JOINT_TORQUE
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    # `vlm_layer_indices`가 주어지면 "앞에서부터 num_vlm_layers개"라는 기본 규칙 대신 이 원본
    # 레이어 인덱스 조합을 그대로 쓴다 (예: scripts/layer_importance.py의 코사인 유사도 기준
    # 중요도 순위로 고른 레이어들). None이면 기존처럼 num_vlm_layers가 그대로 적용된다.
    vlm_layer_indices: list[int] | None = None
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # --- ARD: Asymmetric Role Decomposition (비대칭 역할 분리, 양손 도구 조작 파인튜닝용) ---
    # ARD-VLA 연구계획서 참고. 선두 `2 * ard_arm_dim`개 액션 채널을 Stabilizer 팔(작업물을
    # 붙잡아 고정)과 Actuator 팔(정밀한 도구 조작 수행)로 나누고, 각각 전용 residual head로
    # 보정한 뒤 서로 다른 손실 항으로 비대칭 학습한다. 규칙: 이 `2 * ard_arm_dim`개 채널 중
    # 앞쪽 `ard_arm_dim`개는 왼팔, 다음 `ard_arm_dim`개는 오른팔이다. Actuator는 항상
    # `ard_default_actuator_arm`으로 고정되며(샘플별로 바뀌지 않음), 전체 설정에 동일하게 적용된다.
    use_ard: bool = False
    # ARD-Gen 확정 스펙(2026-10, OpenArm + 공식 그리퍼): 팔 하나당 7 관절 + 그리퍼 1 = 8 DoF.
    ard_arm_dim: int = 8  # 팔 하나당 자유도 (관절 7 + 그리퍼 1)
    # ard_arm_dim개 채널 중 마지막 ard_gripper_dim개를 그리퍼로 간주해서 smooth_loss/traj_loss
    # (1차/2차 차분 벌점)에서 제외한다 — 그리퍼는 bang-bang성 신호라 빠른 개폐에 매끄러움
    # 벌점을 거는 게 부적절하다고 판단했다(lerobot.policies.smolvla.ard._exclude_gripper 참고).
    # L_pos(기본 flow-matching 회귀 손실)는 이 값과 무관하게 그리퍼 채널에도 그대로 걸린다 —
    # 즉 그리퍼는 "L_pos만 받고 smooth/traj 추가 벌점은 받지 않는다". 0이면 과거와 완전히
    # 동일하게 그리퍼도 smooth/traj에 포함된다.
    ard_gripper_dim: int = 1
    ard_default_actuator_arm: str = "right"  # "left" 또는 "right"; 이 팔이 항상 Actuator 역할
    ard_alpha: float = 0.3  # Stabilizer 손실 가중치
    ard_beta: float = 0.7  # Actuator 손실 가중치
    ard_lambda_smooth: float = 1.0  # Stabilizer 흔들림(smoothness) 페널티 가중치 (use_gradnorm=True면 무시됨 — GradNorm은 항상 1.0에서 시작)
    ard_lambda_force: float = 1.0  # Actuator 힘 추적(force-tracking) 페널티 가중치 (use_gradnorm=True면 무시됨 — GradNorm은 항상 1.0에서 시작)
    ard_lambda_traj: float = 1.0  # Actuator 궤적 스무딩(trajectory-smoothness) 페널티 가중치 (use_gradnorm=True면 무시됨 — GradNorm은 항상 1.0에서 시작)

    # smooth_loss/traj_loss가 거는 대상인 x0_hat = x_t - t*v_t는, t(flow-matching 타임스텝)가
    # 클수록(거의 순수 노이즈에 가까운 샘플일수록) 오차/분산이 커지는 경향이 있다(실측:
    # README의 "ARD" 절 참고). "one_minus_t"면 샘플별로 (1-t) 가중치를 곱해서 t≈1 샘플의
    # 영향을 줄인다 — "none"(기본값)이면 가중치 없이(기존과 완전히 동일하게) 그냥 평균낸다.
    ard_reg_time_weighting: str = "none"  # "none" 또는 "one_minus_t"

    # 기존 force_loss는 actuator_traj_pred(x0_hat 기반 액션 추정치)의 마지막 채널(관절 위치/
    # 속도 채널)을 힘의 대용값으로 재활용했는데, 의미가 맞지 않는다. ard_use_force_head=True면
    # suffix_out -> 스칼라 힘을 예측하는 별도 ForceHead(lerobot.policies.smolvla.ard 참고)를
    # 만들어서 force_loss를 거기서 계산한다. 기본값 False면 기존 "force_target이 없으면
    # force_loss=0" 동작과 완전히 동일하다 — 단, force_target이 주어졌는데 이 플래그가 꺼져
    # 있으면(즉 힘을 예측할 방법이 없으면) 과거처럼 엉뚱한 채널을 쓰는 대신 경고 후 0으로
    # 처리한다(ard.py의 compute_ard_losses 참고).
    ard_use_force_head: bool = False

    # 관절 토크 관측값을 모델 입력으로 쓸지 여부. 켜면 배치의 `ard.ARD_JOINT_TORQUE`
    # ("observation.joint_torque") 키를 `observation.state`(그 "뒤에" concat, 기존 상태 채널은
    # 그대로 둠)에 이어붙여서 `state_proj`에 넘긴다(SmolVLAPolicy.prepare_state 참고) — 별도
    # 인코더/브랜치를 새로 만들지 않고, 이미 범용 선형 projection으로 처리되는 state 벡터의
    # 남는 패딩 공간(max_state_dim)에 토크 채널을 추가로 싣는 방식이다. 그래서 새 학습
    # 파라미터가 전혀 생기지 않는다(state_proj의 입력 폭은 항상 max_state_dim으로 고정돼 있어
    # 이미 패딩되던 자리가 토크로 채워질 뿐이다). 기본값 False면 기존과 완전히 동일하다 — 켜져
    # 있는데 배치에 해당 키가 없으면 에러를 낸다(조용히 무시하지 않음).
    ard_use_joint_torque: bool = False

    # --- 대칭 대조군(symmetric ablation) 모드 — ARD의 "팔마다 다른 정규화" 가정 자체가 실제로
    # 도움이 되는지 공정하게 비교하기 위한 ablation용 코드. 켜면 head 구조/파라미터 수는 ARD와
    # 동일하게 유지한 채(AsymmetricResidualHeads를 그대로 재사용), 손실 "결합 방식"만 바뀐다:
    # 양팔 모두 L_pos + lambda_smooth*L_smooth + lambda_traj*L_traj를 동일하게 적용하고
    # (ARD처럼 stabilizer=smooth만/actuator=force+traj로 쪼개지 않음), alpha/beta는 항상
    # 0.5/0.5로 강제된다(사용자가 다르게 줬으면 경고 후 덮어씀). force 항은 ARD 모드와
    # 동일하게 actuator 쪽에만 둔다 — 접촉력/토크는 도구를 조작하는 팔에만 물리적으로 의미가
    # 있는 신호라서(Stabilizer는 작업물을 붙잡고 있을 뿐이라 힘 추적이라는 과제 자체가 성립
    # 안 함), smooth/traj와 달리 대칭으로 만들지 않기로 판단했다(lerobot.policies.smolvla.ard의
    # compute_symmetric_losses 참고).
    ard_symmetric: bool = False

    # --- VLA-Adapter(Wang et al., 2025)식 Bridge Attention — ARD head가 suffix_out(마지막 지점)
    # 하나만이 아니라 SmolLM2 백본의 여러 중간 레이어 특징도 cross-attention으로 조건받게 함 ---
    use_bridge_attention: bool = False  # use_ard=True일 때만 의미가 있다
    # None이면 [num_vlm_layers의 1/4, 1/2, 3/4, 마지막] 4개 지점을 자동으로 고른다 (초반/중반/
    # 후반/마지막 깊이 대표). 인덱스는 원본 32개가 아니라 SmolVLA가 실제로 쓰는(트림된)
    # num_vlm_layers 기준이다 — lerobot.policies.smolvla.ard.resolve_bridge_layer_indices 참고.
    ard_bridge_layer_indices: list[int] | None = None
    ard_bridge_num_heads: int = 4  # Bridge Attention cross-attention의 헤드 수

    # --- GradNorm (Chen et al., 2018) — lambda_smooth/force/traj를 고정값 대신 자동 조정 ---
    # 켜면 세 lambda가 nn.Parameter가 되어 매 스텝, "공유 표현"(actuator/stabilizer head
    # 바로 직전의 suffix_out)에 각 정규화 항이 만드는 그래디언트 norm이 서로 균형 잡히도록
    # 학습 루프가 별도로 업데이트한다 (lerobot.policies.smolvla.ard.GradNormLambdas 참고).
    use_gradnorm: bool = False
    gradnorm_alpha: float = 1.5  # GradNorm의 asymmetry 하이퍼파라미터 (논문 기본값)
    gradnorm_lr: float = 0.025  # lambda 전용 옵티마이저 학습률 (메인 옵티마이저와 별개)

    # --- EfficientVLA(Yang et al., 2025)식 Task-Relevance/Diversity 비전 토큰 프루닝 ---
    # 켜면 프레임당 비전 토큰(pixel shuffle 직후, 보통 64개)을 언어 지시문과의 cross-attention
    # 관련성 점수 기준으로 token_pruning_k_final개까지 줄인다 — 상위 token_pruning_k_key개는
    # 무조건 남기고(핵심 세트), 나머지는 절반 관련성/절반 다양성(코사인 거리 최대)으로 채운다.
    # lerobot.policies.smolvla.token_pruning 참고.
    use_token_pruning: bool = False
    token_pruning_k_final: int = 32  # 프레임당 최종 비전 토큰 수 (실험용 — 32/48/64 등으로 바꿔가며 비교)
    token_pruning_k_key: int = 6  # 무조건 남기는 핵심(최고 관련성) 토큰 수, 논문 권장 4~8

    # --- FreqPolicy(2025)식 주파수 영역 일관성 손실 ---
    # 액션 청크(u_t/v_t)를 DCT로 주파수 성분으로 분해해서, 저주파(궤적 전체의 형태)에 더 큰
    # 가중치를 준 MSE를 기존 flow-matching 회귀 손실에 추가로 더한다. FreqPolicy 논문의
    # "coarse-to-fine 자기회귀 생성" 아키텍처 자체를 이식한 게 아니라(SmolVLA의 병렬 flow-
    # matching 디코딩은 그대로 유지), 그 논문의 "저주파 우선" 통찰만 additive loss로 가져온
    # 것이다. use_ard와 무관하게(둘 다 꺼도/켜도) 독립적으로 켤 수 있다.
    # lerobot.policies.smolvla.freq_policy 참고.
    use_freq_policy: bool = False
    freq_lambda: float = 1.0  # 주파수 일관성 손실을 total loss에 더할 때의 가중치
    # decay=0이면 전체 주파수 bin에 균등 가중(=시간 영역 MSE와 수학적으로 동일, Parseval 정리).
    # decay가 클수록 저주파에 더 쏠린다 — chunk_size에 무관하게, 최고 주파수 bin의 가중치가
    # 최저 주파수 bin의 exp(-decay)배가 되도록 정규화되어 있다(예: decay=2 -> 약 13.5%).
    freq_decay: float = 2.0

    # --- 폐루프(closed-loop) 구조 — Judge(성공 판정)/Selector(다음 서브태스크 선택) head.
    # 하나의 prefix(이미지+언어+state) forward를 Judge/Selector/action expert가 공유한다
    # (lerobot.policies.smolvla.judge.Judge, SmolVLAPolicy.predict_action_chunk_with_decisions
    # 참고). 이번 작업은 구조 구현만 — 데이터 생성/실제 학습/캘리브레이션 실측은 범위 밖.
    use_judge: bool = False
    judge_hidden_dim: int = 256
    # -1이면 prefix 마지막 레이어(post-norm, sample_actions()가 버리던 반환값을 재사용 —
    # 추가 연산 없음). 그 외 값이면 Bridge Attention과 같은 `collect_layer_indices` 경로로
    # 뽑은 중간 레이어(pre-norm) — 상한(트림된 num_vlm_layers)은 실제 모델을 로드해야 알 수
    # 있어서 modeling_smolvla.py의 VLAFlowMatching.__init__에서 검증한다.
    judge_feature_layer: int = -1
    judge_num_failure_types: int = 0  # 0이면 실패 유형 분류 head 없음
    judge_use_progress: bool = False  # 진행도(0~1) 회귀 head 추가 여부
    # Judge/Selector 손실이 백본/action expert로 역전파되지 않게 pooled feature를 detach한다
    # — 1차 연구에서는 백본을 고정한 채로만 Judge/Selector를 다루는 게 전제라서 기본 True.
    judge_detach_features: bool = True
    use_selector: bool = False
    selector_num_options: int = 0  # use_selector=True면 1 이상이어야 함
    judge_loss_weight: float = 1.0
    selector_loss_weight: float = 1.0

    # --- LAC(Latent-Aware Controller, Co-VLA arXiv 2606.20285) — 학습 없는 추론 시점 EMA
    # 필터. ARD의 "공유(action_out_proj)+잔차(ARD 헤드)" 가산 구조에서 팔별 공유/잔차 에너지
    # 비율(ρ)과 협동 방향(ω)을 보고 매 실행 스텝마다 필터 강도 α를 조절한다
    # (lerobot.policies.smolvla.lac.LatentAwareController 참고). "none"이면 기존 출력과
    # bit-for-bit 동일하다. "covla"면 Co-VLA 원본(두 팔에 같은 α), "role_aware"면 ARD의
    # Stabilizer/Actuator 역할별로 α를 따로 둔다(둘 다 use_ard=True 필요 — LAC가 가정하는
    # "공유+잔차" 분해 자체가 ARD head의 구조라서).
    lac_mode: str = "none"  # "none" | "covla" | "role_aware"
    lac_alpha_base: float = 0.4
    lac_delta_macro: float = 0.4
    lac_delta_prec: float = 0.3
    lac_delta_noise: float = 0.1
    lac_tau_rho: float = 0.01
    lac_tau_omega: float = 0.3
    lac_beta: float = 0.2
    lac_alpha_min: float = 0.05
    lac_alpha_max: float = 0.95
    lac_epsilon: float = 1e-6
    # role_aware 전용 — Stabilizer는 상한을 낮게 고정, Actuator는 협동(ω>tau_omega)일 때만
    # 높인다(lac.py의 LatentAwareController._role_alpha_hat 참고).
    lac_alpha_stab_max: float = 0.5
    lac_alpha_act_cooperative: float = 0.9
    # 오른팔 관절 축 미러링 보정(ω 계산에만 영향, 역할과 무관하게 항상 오른팔에 적용) — 길이는
    # `ard_arm_dim - ard_gripper_dim`(관절 수)과 같아야 한다. None이면 전부 1.0(미러링 없음)
    # — ARD-Gen의 OpenArm MJCF 비교로 얻은 기본 추정값은 [-1,-1,1,1,1,-1,1](joint1~7)이지만,
    # **실제 forward kinematics로 검증하지 않았다** (lac.py 모듈 docstring 참고, "미검증").
    lac_mirror_sign: list[float] | None = None
    lac_log: bool = False  # 매 실행 스텝 rho/omega/alpha/regime을 policy.lac_debug_log에 기록

    # --- "설정 가능한 ARD" (ARD-VLA → RoCo Challenge 등 매 회차 바뀌는 과제에 대응) ---
    # role_map: {"stabilizer": "left"|"right", "actuator": "left"|"right"} — ard_default_actuator_arm의
    # 사용자 친화적 별칭. None(기본)이면 ard_default_actuator_arm을 그대로 쓴다(기존과 동일). 지정하면
    # __post_init__에서 ard_default_actuator_arm = role_map["actuator"]로 덮어써서, 실제 분리/결합
    # 로직(resolve_actuator_is_first 등)은 여전히 ard_default_actuator_arm 하나만 읽는다(단일 소스 유지).
    role_map: dict[str, str] | None = None
    # 액션 공간 종류. "joint"(기본, 기존과 동일)면 채널이 관절 각도/속도라서 lac_mirror_sign(관절 축
    # 미러링 보정)이 그대로 적용된다. "ee"(end-effector, 예: RoCo Challenge 공식 데이터셋)면 채널이
    # 엔드이펙터 목표 pose+그리퍼라서 "관절 축 미러링"이라는 개념이 성립하지 않는다 — __post_init__이
    # lac_mirror_sign을 강제로 None(미러링 없음)으로 덮어쓰고, 사용자가 값을 줬으면 경고한다.
    action_space: str = "joint"  # "joint" 또는 "ee"
    # 메인 flow-matching 회귀 손실(L_pos = F.*_loss(u_t, v_t), modeling_smolvla.py)의 종류. "l2"
    # (기본, 기존과 완전히 동일한 MSE)는 목표에 가까워질수록 기울기가 0으로 사라져서 끝까지 밀어
    # 넣지 않는 경향이 있다 — RoCo Challenge 1회 우승팀이 "l1"로 바꿔 이 문제를 해결했다고 보고
    # 했다(arXiv 2603.15469). "smooth_l1"은 그 중간(큰 오차엔 L1처럼, 작은 오차엔 L2처럼) 동작.
    pos_loss_type: str = "l2"  # "l2" | "l1" | "smooth_l1"
    # 액션 청크 내 위치(실행 스텝 인덱스, 시간이 아님)별로 정규화 스케일을 다르게 준다 — 뒤쪽
    # 위치일수록(먼 미래일수록) 예측 분산이 커지는 경향을 σ(t) = a + s*sqrt(t) (t=0..chunk_size-1)
    # 로 반영한다. 기본 False면 기존 정규화(차원별 평균/표준편차만)와 비트 단위로 완전히 동일하다
    # (LeHome 우승 보고서, arXiv 2606.27163 참고). scripts/fit_timestep_norm.py로 a/s를 피팅한다.
    per_timestep_norm: bool = False
    per_timestep_norm_a: float | list[float] = 1.0  # 스칼라 또는 (실제 액션 차원,) 리스트
    per_timestep_norm_s: float | list[float] = 0.0

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.vlm_layer_indices is not None:
            if len(self.vlm_layer_indices) == 0:
                raise ValueError("`vlm_layer_indices`가 비어 있습니다 — 최소 1개 이상의 레이어 인덱스가 필요합니다.")
            if len(set(self.vlm_layer_indices)) != len(self.vlm_layer_indices):
                raise ValueError(f"`vlm_layer_indices`에 중복된 인덱스가 있습니다: {self.vlm_layer_indices}")
            if any(i < 0 for i in self.vlm_layer_indices):
                raise ValueError(f"`vlm_layer_indices`는 음수를 포함할 수 없습니다: {self.vlm_layer_indices}")
            # 원본 VLM 레이어 수(예: 32)를 넘는 인덱스인지는 실제 모델을 로드해야 알 수 있어서
            # (config 단계에선 Hub 접근 없이 알 수 없음) 여기서는 형식만 검증하고, 상한 체크는
            # smolvlm_with_expert.py의 SmolVLMWithExpertModel.__init__에서 한다.
        if self.role_map is not None:
            if set(self.role_map.keys()) != {"stabilizer", "actuator"}:
                raise ValueError(
                    f"`role_map`은 정확히 'stabilizer'와 'actuator' 키만 가져야 합니다. "
                    f"현재 키: {sorted(self.role_map.keys())}"
                )
            stabilizer_arm, actuator_arm = self.role_map["stabilizer"], self.role_map["actuator"]
            if stabilizer_arm not in ("left", "right") or actuator_arm not in ("left", "right"):
                raise ValueError(f"`role_map`의 값은 'left' 또는 'right'여야 합니다. 현재 값: {self.role_map}")
            if stabilizer_arm == actuator_arm:
                raise ValueError(f"`role_map`의 stabilizer와 actuator는 서로 다른 팔이어야 합니다: {self.role_map}")
            self.ard_default_actuator_arm = actuator_arm
        if self.action_space not in ("joint", "ee"):
            raise ValueError(f"`action_space`는 'joint' 또는 'ee'여야 합니다. 현재 값: {self.action_space!r}")
        if self.pos_loss_type not in ("l2", "l1", "smooth_l1"):
            raise ValueError(
                f"`pos_loss_type`은 'l2'/'l1'/'smooth_l1' 중 하나여야 합니다. 현재 값: {self.pos_loss_type!r}"
            )
        if self.action_space == "ee" and self.lac_mirror_sign is not None:
            logging.warning(
                "action_space='ee'에서는 '관절 축 미러링' 전제가 성립하지 않아 `lac_mirror_sign`"
                "(현재 값: %s)을 무시하고 전부 1.0(미러링 없음)으로 강제합니다.",
                self.lac_mirror_sign,
            )
            self.lac_mirror_sign = None
        if self.per_timestep_norm:
            for name, value in (
                ("per_timestep_norm_a", self.per_timestep_norm_a),
                ("per_timestep_norm_s", self.per_timestep_norm_s),
            ):
                if isinstance(value, list) and len(value) == 0:
                    raise ValueError(f"`{name}`가 리스트면 비어 있을 수 없습니다.")
            a_values = (
                self.per_timestep_norm_a if isinstance(self.per_timestep_norm_a, list) else [self.per_timestep_norm_a]
            )
            if any(a <= 0 for a in a_values):
                raise ValueError(
                    f"`per_timestep_norm_a`는 모두 양수여야 합니다(sigma(t=0)=a). 현재 값: {self.per_timestep_norm_a}"
                )
        if self.use_ard:
            if self.ard_arm_dim <= 0:
                raise ValueError(f"`ard_arm_dim`은 양수여야 합니다. 현재 값: {self.ard_arm_dim}")
            if 2 * self.ard_arm_dim > self.max_action_dim:
                raise ValueError(
                    f"ARD는 `2 * ard_arm_dim`({2 * self.ard_arm_dim})개 채널이 필요하지만 "
                    f"`max_action_dim`이 {self.max_action_dim}밖에 되지 않습니다."
                )
            if self.ard_default_actuator_arm not in ("left", "right"):
                raise ValueError(
                    f"`ard_default_actuator_arm`은 'left' 또는 'right'여야 합니다. "
                    f"현재 값: {self.ard_default_actuator_arm!r}"
                )
            if self.ard_reg_time_weighting not in ("none", "one_minus_t"):
                raise ValueError(
                    f"`ard_reg_time_weighting`은 'none' 또는 'one_minus_t'여야 합니다. "
                    f"현재 값: {self.ard_reg_time_weighting!r}"
                )
            if not (0 <= self.ard_gripper_dim < self.ard_arm_dim):
                raise ValueError(
                    f"`ard_gripper_dim`은 0 이상 `ard_arm_dim`({self.ard_arm_dim}) 미만이어야 "
                    f"합니다(최소 1개의 관절 채널은 남아야 함). 현재 값: {self.ard_gripper_dim}"
                )
        if self.use_gradnorm and not self.use_ard:
            raise ValueError("`use_gradnorm`은 `use_ard=True`일 때만 의미가 있습니다.")
        if self.ard_use_force_head and not self.use_ard:
            raise ValueError("`ard_use_force_head`는 `use_ard=True`일 때만 의미가 있습니다.")
        if self.ard_symmetric:
            if not self.use_ard:
                raise ValueError("`ard_symmetric`은 `use_ard=True`일 때만 의미가 있습니다.")
            if self.ard_alpha != 0.5 or self.ard_beta != 0.5:
                logging.warning(
                    "`ard_symmetric=True`에서는 alpha/beta가 항상 0.5/0.5로 강제됩니다 — "
                    "직접 지정하신 ard_alpha=%s, ard_beta=%s는 무시되고 0.5/0.5로 덮어씁니다.",
                    self.ard_alpha, self.ard_beta,
                )
            self.ard_alpha = 0.5
            self.ard_beta = 0.5
        if self.use_bridge_attention:
            if not self.use_ard:
                raise ValueError("`use_bridge_attention`은 `use_ard=True`일 때만 의미가 있습니다.")
            if self.ard_bridge_num_heads <= 0:
                raise ValueError(f"`ard_bridge_num_heads`는 양수여야 합니다. 현재 값: {self.ard_bridge_num_heads}")
            if self.ard_bridge_layer_indices is not None:
                if len(self.ard_bridge_layer_indices) == 0:
                    raise ValueError("`ard_bridge_layer_indices`가 비어 있습니다 — 최소 1개 이상의 레이어 인덱스가 필요합니다.")
                if len(set(self.ard_bridge_layer_indices)) != len(self.ard_bridge_layer_indices):
                    raise ValueError(f"`ard_bridge_layer_indices`에 중복된 인덱스가 있습니다: {self.ard_bridge_layer_indices}")
                if any(i < 0 for i in self.ard_bridge_layer_indices):
                    raise ValueError(f"`ard_bridge_layer_indices`는 음수를 포함할 수 없습니다: {self.ard_bridge_layer_indices}")
                # num_vlm_layers 상한 체크는 실제 모델을 로드해야 알 수 있어서(트림 후 레이어 수),
                # modeling_smolvla.py의 VLAFlowMatching.__init__에서 resolve_bridge_layer_indices로 한다.
        if self.use_token_pruning:
            if self.token_pruning_k_final <= 0:
                raise ValueError(f"`token_pruning_k_final`은 양수여야 합니다. 현재 값: {self.token_pruning_k_final}")
            if self.token_pruning_k_key <= 0:
                raise ValueError(f"`token_pruning_k_key`는 양수여야 합니다. 현재 값: {self.token_pruning_k_key}")
        if self.use_freq_policy:
            if self.chunk_size < 2:
                raise ValueError(f"`use_freq_policy`는 `chunk_size`가 2 이상이어야 의미가 있습니다. 현재 값: {self.chunk_size}")
            if self.freq_lambda < 0:
                raise ValueError(f"`freq_lambda`는 음수일 수 없습니다. 현재 값: {self.freq_lambda}")
            if self.freq_decay < 0:
                raise ValueError(f"`freq_decay`는 음수일 수 없습니다. 현재 값: {self.freq_decay}")
        if self.judge_feature_layer != -1 and self.judge_feature_layer < 0:
            raise ValueError(
                f"`judge_feature_layer`는 -1(마지막 레이어) 또는 0 이상이어야 합니다. "
                f"현재 값: {self.judge_feature_layer}"
            )
        if self.use_judge or self.use_selector:
            if self.judge_hidden_dim <= 0:
                raise ValueError(f"`judge_hidden_dim`은 양수여야 합니다. 현재 값: {self.judge_hidden_dim}")
        if self.judge_num_failure_types > 0 and not self.use_judge:
            raise ValueError("`judge_num_failure_types`는 `use_judge=True`일 때만 의미가 있습니다.")
        if self.judge_use_progress and not self.use_judge:
            raise ValueError("`judge_use_progress`는 `use_judge=True`일 때만 의미가 있습니다.")
        if self.use_judge and self.judge_num_failure_types < 0:
            raise ValueError(f"`judge_num_failure_types`는 음수일 수 없습니다. 현재 값: {self.judge_num_failure_types}")
        if self.use_selector and self.selector_num_options < 1:
            raise ValueError(
                f"`use_selector=True`면 `selector_num_options`가 1 이상이어야 합니다. "
                f"현재 값: {self.selector_num_options}"
            )
        if not self.use_selector and self.selector_num_options != 0:
            raise ValueError("`selector_num_options`는 `use_selector=True`일 때만 설정할 수 있습니다.")
        if self.lac_mode not in ("none", "covla", "role_aware"):
            raise ValueError(f"`lac_mode`는 'none'/'covla'/'role_aware' 중 하나여야 합니다. 현재 값: {self.lac_mode!r}")
        if self.lac_mode != "none":
            if not self.use_ard:
                raise ValueError(
                    "`lac_mode`는 `use_ard=True`일 때만 의미가 있습니다 — LAC의 공유/잔차 분해가 "
                    "ARD head(action_out_proj 출력=공유, ARD 헤드 출력=잔차) 구조를 전제합니다."
                )
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError(
                    "`lac_mode`와 RTC(`rtc_config.enabled=True`)는 함께 쓸 수 없습니다 — RTC가 여러 "
                    "denoise 후보 velocity를 블렌딩해서 최종 v_t를 만들기 때문에, 공유/잔차를 denoise "
                    "스텝마다 누적해 분해하는 LAC의 전제(선형 적분)가 깨집니다(미지원, PROGRESS.md 참고)."
                )
            if self.lac_mirror_sign is not None:
                num_joints = self.ard_arm_dim - self.ard_gripper_dim
                if len(self.lac_mirror_sign) != num_joints:
                    raise ValueError(
                        f"`lac_mirror_sign` 길이({len(self.lac_mirror_sign)})가 관절 수"
                        f"(`ard_arm_dim - ard_gripper_dim` = {num_joints})와 다릅니다."
                    )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

        if self.use_ard and self.action_feature is not None:
            real_action_dim = self.action_feature.shape[0]
            if real_action_dim < 2 * self.ard_arm_dim:
                raise ValueError(
                    f"ARD를 사용하려면 데이터셋의 실제 액션 차원(현재 {real_action_dim})이 최소한 "
                    f"`2 * ard_arm_dim`({2 * self.ard_arm_dim}) 이상이어야, 선두 채널이 왼팔/오른팔 "
                    "블록으로 균등하게 나뉩니다."
                )

        if self.per_timestep_norm and self.action_feature is not None:
            real_action_dim = self.action_feature.shape[0]
            for name, value in (
                ("per_timestep_norm_a", self.per_timestep_norm_a),
                ("per_timestep_norm_s", self.per_timestep_norm_s),
            ):
                if isinstance(value, list) and len(value) != real_action_dim:
                    raise ValueError(
                        f"`{name}` 길이({len(value)})가 실제 액션 차원({real_action_dim})과 다릅니다."
                    )

        if self.ard_use_joint_torque:
            # 데이터셋이 observation.state와 관절 토크를 별도 feature로 제공하는 경우에만 미리
            # 검증할 수 있다(둘 다 input_features에 등록돼 있을 때) — 토크 차원 자체는 데이터셋
            # 스펙이라 config 단계에서는 알 수 없으므로, 둘 다 있을 때만 "concat 후
            # max_state_dim을 넘지 않는지"를 확인한다. 배치에 키가 없는 경우는 런타임에
            # (SmolVLAPolicy.prepare_state) 걸린다.
            state_feat = self.input_features.get(OBS_STATE)
            torque_feat = self.input_features.get(ARD_JOINT_TORQUE)
            if state_feat is not None and torque_feat is not None:
                combined_dim = state_feat.shape[0] + torque_feat.shape[0]
                if combined_dim > self.max_state_dim:
                    raise ValueError(
                        f"`ard_use_joint_torque=True`로 관절 토크를 state에 concat하면 실제 "
                        f"차원이 {combined_dim}(state {state_feat.shape[0]} + torque "
                        f"{torque_feat.shape[0]})이 되는데, `max_state_dim`이 {self.max_state_dim}"
                        "밖에 되지 않습니다."
                    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
