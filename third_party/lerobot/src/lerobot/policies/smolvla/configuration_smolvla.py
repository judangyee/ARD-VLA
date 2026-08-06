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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


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
    ard_arm_dim: int = 7  # 팔 하나당 자유도 (예: 관절 6 + 그리퍼 1)
    ard_default_actuator_arm: str = "right"  # "left" 또는 "right"; 이 팔이 항상 Actuator 역할
    ard_alpha: float = 0.3  # Stabilizer 손실 가중치
    ard_beta: float = 0.7  # Actuator 손실 가중치
    ard_lambda_smooth: float = 1.0  # Stabilizer 흔들림(smoothness) 페널티 가중치
    ard_lambda_force: float = 1.0  # Actuator 힘 추적(force-tracking) 페널티 가중치
    ard_lambda_traj: float = 1.0  # Actuator 궤적 스무딩(trajectory-smoothness) 페널티 가중치

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
