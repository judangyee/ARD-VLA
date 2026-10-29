#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    # angle_unwrap_dims가 있으면(기본 None) 표준 정규화 전에, raw 액션의 회전 채널을 시간축
    # (학습 청크 내부)으로 연속으로 펴서 ±pi 경계 불연속을 없앤다 — 없으면(기본) 이 단계가
    # 파이프라인에 들어가지도 않아 bit-exact다. configuration_smolvla.py의 angle_unwrap_dims
    # 필드 주석 참고(회전벡터/오일러각 여부가 미확인이라 기본 꺼짐).
    if config.angle_unwrap_dims:
        input_steps.append(AngleUnwrapProcessorStep(angle_dims=tuple(config.angle_unwrap_dims)))
    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    # Unwrap의 역연산 — 로봇에 보내기 직전(여기가 마지막 단계), 연속 표현으로 다루던 회전
    # 채널을 로봇 API가 기대하는 [-pi, pi) 범위로 다시 감싼다.
    if config.angle_unwrap_dims:
        output_steps.append(AngleRewrapProcessorStep(angle_dims=tuple(config.angle_unwrap_dims)))
    output_steps.append(DeviceProcessorStep(device="cpu"))
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """x를 [-pi, pi) 구간으로 감싼다 — 각도 차이의 2*pi 경계 불연속(예: 3.14 vs -3.14가
    실제로는 거의 같은 각도인데 raw 차이는 ~6.28)을 없앤다. 입력이 실제 라디안 단위일
    때만 의미가 있다(아래 AngleUnwrapProcessorStep/AngleRewrapProcessorStep 참고)."""
    return x - 2 * math.pi * torch.round(x / (2 * math.pi))


def _unwrap_angles_along_time(x: torch.Tensor, angle_dims: tuple[int, ...]) -> torch.Tensor:
    """x: (..., T, D). `angle_dims` 채널만 시간축(dim=-2)을 따라 연속으로 편다 — np.unwrap과
    같은 원리로, 연속된 두 프레임의 wrap_to_pi(차이)를 누적합해서 재구성한다. 다른 채널은
    그대로. T<2(시간축이 퇴화)면 변화 없음.

    `x`는 항상 하나의 에피소드 내부에서 뽑힌 연속 구간이라(LeRobotDataset의 delta_timestamps
    윈도우는 에피소드 경계를 넘지 않음), 이 함수가 "에피소드 경계를 넘어 펴지 않는다"는
    요구를 자동으로 만족한다 — 전체 에피소드를 한 번에 펴지 않고 "이 청크 안에서"만 펴는
    것으로 충분하다(모델은 청크 내부의 상대적 연속성만 필요함)."""
    if x.shape[-2] < 2:
        return x
    idx = list(angle_dims)
    sub = x[..., idx]  # (..., T, len(angle_dims))
    deltas = _wrap_to_pi(sub[..., 1:, :] - sub[..., :-1, :])
    unwrapped = torch.cat([sub[..., :1, :], sub[..., :1, :] + torch.cumsum(deltas, dim=-2)], dim=-2)
    result = x.clone()
    result[..., idx] = unwrapped
    return result


@ProcessorStepRegistry.register(name="smolvla_angle_unwrap")
@dataclass
class AngleUnwrapProcessorStep(ProcessorStep):
    """학습 전처리: `angle_dims`(액션 벡터의 절대 채널 인덱스, 예: RoCo의 양팔 rx,ry,rz =
    [3,4,5,10,11,12]) 채널을 액션 청크의 시간축을 따라 연속으로 편다(np.unwrap과 동일한
    원리) — ±pi 경계에서의 거짓 불연속(예: 3.14 다음 프레임이 -3.14, 실제로는 거의 같은
    각도)을 없앤다. `NormalizerProcessorStep`보다 먼저(raw 값 기준으로) 적용해야 한다.

    **주의**: 이 전처리는 각 채널이 독립적으로 주기적인 표현(예: 오일러각의 각 축)일 때만
    올바르다 — 회전벡터(axis-angle)라면 세 채널을 한 벡터로 다뤄야 해서 축별 unwrap은
    수학적으로 맞지 않다(docs/configurable_ard.md 참고, 아직 RoCo의 rx/ry/rz가 어느 쪽인지
    확인 못 함). `angle_dims`가 비어 있으면(기본) 아무것도 하지 않는다.

    `NormalizerProcessorStep`과 같은 패턴으로, 배치에 액션이 없으면(실제 로봇 추론 시
    입력에는 액션이 없음) 조용히 통과시킨다 — 에러를 내지 않는다."""

    angle_dims: tuple[int, ...] = ()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not self.angle_dims or action.ndim < 2:
            return new_transition
        new_transition[TransitionKey.ACTION] = _unwrap_angles_along_time(action, self.angle_dims)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # shape/의미는 바뀌지 않음(값만 연속 표현으로 변함)


@ProcessorStepRegistry.register(name="smolvla_angle_rewrap")
@dataclass
class AngleRewrapProcessorStep(ProcessorStep):
    """추론 후처리: `AngleUnwrapProcessorStep`의 역연산 — 로봇에 보내기 직전(파이프라인의
    마지막 단계), 연속 표현으로 다루던 `angle_dims` 채널을 로봇 API가 기대하는 표준 범위
    [-pi, pi)로 다시 감싼다. `angle_dims`가 비어 있으면(기본) 아무것도 하지 않는다."""

    angle_dims: tuple[int, ...] = ()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not self.angle_dims:
            return new_transition
        idx = list(self.angle_dims)
        result = action.clone()
        result[..., idx] = _wrap_to_pi(action[..., idx])
        new_transition[TransitionKey.ACTION] = result
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
