"""`--smoke` 모드/통합 테스트 전용 합성 데이터 생성기.

실제 RoCo 데이터셋(44차원 state, 14차원 action, 카메라 3대) 스키마를 그대로 흉내 내되
(state_select 적용 후 30차원, 카메라는 head/left_hand/right_hand 3대), 이미지 해상도는
`_tiny_vlm.py`의 합성 SmolVLM vision config(`image_size=64`)와 맞춰 작게 둔다 —
스모크의 목적은 "파이프라인이 안 깨지는지"이고 실제 해상도/통계는 범위 밖이다."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CAMERA_NAMES = ("head", "left_hand", "right_hand")
STATE_DIM = 30  # state_select 적용 후
ACTION_DIM = 14


def image_key(camera: str) -> str:
    from lerobot.utils.constants import OBS_IMAGES

    return f"{OBS_IMAGES}.{camera}"


def build_smoke_features(
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    for cam in camera_names:
        input_features[image_key(cam)] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return input_features, output_features


def build_smoke_batch(
    batch_size: int = 2,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    chunk_size: int = 8,
    lang_len: int = 6,
) -> dict:
    """`preprocessor(batch)`에 바로 넣을 수 있는 **raw**(전처리 전) 배치를 만든다 — 토큰
    텐서가 아니라 `"task"`(문자열 리스트)를 준다. `TokenizerProcessorStep`이
    `complementary_data["task"]`에서 문자열을 읽어 그 자리에서 토크나이즈하기
    때문이다(실측 확인 — 사전 토큰화된 텐서를 미리 넣어두는 방식이 아니라, raw 문자열을
    파이프라인이 직접 토크나이즈하는 구조다). `lang_len`은 이제 쓰이지 않지만 호출부
    호환을 위해 인자로만 남겨둔다."""
    from lerobot.utils.constants import ACTION, OBS_STATE

    del lang_len
    batch = {
        OBS_STATE: torch.randn(batch_size, state_dim),
        ACTION: torch.randn(batch_size, chunk_size, action_dim),
        "task": ["스모크 테스트용 가짜 지시문"] * batch_size,
    }
    for cam in camera_names:
        batch[image_key(cam)] = torch.rand(batch_size, *image_shape)
    return batch


def build_smoke_frames(num_frames: int, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> list[dict]:
    """`norm_stats.compute_train_stats`용 — 단일 프레임(청크 없음) 딕셔너리 목록."""
    from lerobot.utils.constants import ACTION, OBS_STATE

    rng = torch.Generator().manual_seed(0)
    return [
        {
            OBS_STATE: torch.randn(state_dim, generator=rng),
            ACTION: torch.randn(action_dim, generator=rng),
        }
        for _ in range(num_frames)
    ]


def build_smoke_eval_samples(
    num_samples: int = 3,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    image_shape: tuple[int, int, int] = (3, 64, 64),
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    chunk_size: int = 8,
    lang_len: int = 6,
):
    from eval_offline import EvalSample  # noqa: PLC0415 (경량 스크립트, 순환 임포트 회피)

    samples = []
    for _ in range(num_samples):
        batch = build_smoke_batch(
            batch_size=1, camera_names=camera_names, image_shape=image_shape, state_dim=state_dim,
            action_dim=action_dim, chunk_size=1, lang_len=lang_len,
        )
        del batch["action"]  # 추론 입력에는 정답 action이 없음(실제 배치와 동일)
        samples.append(
            EvalSample(
                batch=batch,
                target_chunk=torch.randn(chunk_size, action_dim),
                prev_action=torch.randn(action_dim),
            )
        )
    return samples
