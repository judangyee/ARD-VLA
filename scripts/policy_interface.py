#!/usr/bin/env python3
"""RoCo 평가용 정책 실행 어댑터 — 순수 파이썬 인터페이스.

공식 RoCo 평가 환경의 실제 API(관측을 주는 함수 시그니처, 액션을 받는 함수 시그니처, 호출
빈도 등)는 이 레포 어디에도 없고 이 세션에서 열어볼 수도 없었다(HF Hub 차단) — 이 모듈은
**그 어떤 가정도 하지 않는다**. 대신 명시적 인자(이미지 dict, state 배열, task 문자열)를
입력으로 받고 torch.Tensor를 출력으로 돌려주는, 공식 환경과 무관하게 그 자체로 테스트 가능한
어댑터를 제공한다. 공식 환경과 실제로 연결할 때 확인해야 할 항목은 docs/policy_interface.md의
"미확인 항목" 절에 모아뒀다.

내부적으로는 `SmolVLAPolicy.select_action()`(third_party/lerobot/src/lerobot/policies/smolvla/
modeling_smolvla.py:508)을 그대로 재사용한다 — state_select/정규화(min_std 포함)/
n_action_steps 단위 청크 큐/역정규화/LAC를 전부 이미 구현하고 있어서, 이 어댑터가 다시
구현할 필요가 없다(ponytail: 이미 있는 걸 또 만들지 않는다). 이 어댑터가 새로 하는 일은
딱 두 가지뿐이다: (1) uint8 HWC 이미지 -> policy가 기대하는 float32 CHW [0,1] 텐서 변환,
(2) `make_smolvla_pre_post_processors()`가 만드는 전/후처리 파이프라인과 `select_action()`을
하나의 `act()` 호출로 묶기.

Usage:
    from policy_interface import RoCoPolicyAdapter
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)
    adapter.reset()
    action = adapter.act(images={"head": img_hwc_uint8, "left_hand": ..., "right_hand": ...},
                          state=state_44d, task="insert the bolt into the board")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lerobot" / "src"))

from lerobot.utils.constants import OBS_STATE  # noqa: E402

# 미확인(docs/policy_interface.md 참고): 공식 RoCo 환경이 실제로 쓰는 카메라 관측 키 이름.
# roco_board.yaml 주석은 "대부분 왼팔만 사용"이라고만 적혀 있지, 카메라 피처 키 이름 자체는
# 어디에도 없다 — 아래는 이 어댑터가 기본값으로 거는 추정 이름일 뿐이다.
DEFAULT_CAMERA_KEYS = {
    "head": "observation.images.head",
    "left_hand": "observation.images.left_hand",
    "right_hand": "observation.images.right_hand",
}

STATE_DIM = 44  # roco_board.yaml 상단 주석(info.json 채널 이름으로 확인됨) 기준
ACTION_DIM = 14  # 팔당 [ee pose 6 + 그리퍼 1] = 7, 양팔 14 (roco_board.yaml 기준)


def _image_hwc_uint8_to_chw_float(img: "np.ndarray | torch.Tensor") -> torch.Tensor:
    """(H, W, 3) uint8 [0,255] -> (3, H, W) float32 [0,1]. `SmolVLAPolicy.prepare_images()`
    (modeling_smolvla.py:710-712)가 그 다음 단계(리사이즈+패딩+[-1,1] 변환)를 맡아서 하므로,
    이 함수는 딱 dtype/레이아웃/값 범위만 맞춰준다 — 크기를 특정 해상도로 바꾸지 않는다
    (원본이 240x320이어도 그대로 CHW로만 바꿔서 넘긴다)."""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.dtype != torch.uint8:
        raise ValueError(f"이미지는 uint8이어야 합니다 (받은 dtype: {img.dtype})")
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"이미지는 (H, W, 3) 모양이어야 합니다 (받은 shape: {tuple(img.shape)})")
    return img.permute(2, 0, 1).to(dtype=torch.float32) / 255.0


class RoCoPolicyAdapter:
    """RoCo 평가 하네스와 `SmolVLAPolicy` 사이의 어댑터. 입력/출력 포맷은 docs/policy_interface.md
    1절("입출력 인터페이스")에 정확히 정의돼 있다 — 요약: 이미지 3장(head/left_hand/right_hand,
    240x320 uint8) + state(44,) float + task 문자열 -> action(14,) float(EE 절대 목표, 연속
    각도, wrap 없음)."""

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        camera_keys: dict[str, str] | None = None,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.camera_keys = dict(camera_keys or DEFAULT_CAMERA_KEYS)

    def reset(self) -> None:
        """에피소드 시작 시 호출한다. `policy.reset()`(modeling_smolvla.py:270-293)이 액션 큐
        (`n_action_steps` 길이 deque)와 LAC 필터 상태(`lac_mode != "none"`일 때)를 전부 새로
        만든다 — 이 어댑터가 따로 들고 있는 상태는 없다. task-graph 수준의 "폐루프 상태"
        (서브태스크 진행/재시도 횟수 등, closed_loop.py의 `EpisodeState`)는 이 어댑터의 책임
        밖이다 — Judge/Selector로 서브태스크를 오가는 평가가 필요하면
        `lerobot.policies.smolvla.closed_loop.run_closed_loop()`를 이 어댑터 대신(또는 위에)
        쓴다(docs/policy_interface.md 참고)."""
        self.policy.reset()

    def act(self, images: dict[str, Any], state: "np.ndarray | torch.Tensor", task: str) -> torch.Tensor:
        """한 스텝의 관측을 받아 한 스텝의 액션을 돌려준다.

        images: {"head": (240,320,3) uint8, "left_hand": ..., "right_hand": ...} — 키 이름은
            `self.camera_keys`로 설정 가능(기본값은 미확인 추정치, 위 DEFAULT_CAMERA_KEYS 참고).
        state: (44,) float — EE 위치+쿼터니언 14 + 관절 위치 14 + 관절 속도 14 + 그리퍼 2
            (roco_board.yaml 상단 주석 기준 순서). state_select(여기서는 관절 속도 28-41 제외)는
            policy 내부(`prepare_state`)에서 적용되므로 여기서는 44차원 그대로 넘긴다.
        task: 자연어 지시문 문자열.

        반환: (14,) float — EE 절대 목표(위치3+회전3+그리퍼1, 양팔), 연속 각도(wrap 없음,
            action_space="ee"+angle_unwrap_dims=None 기준)."""
        obs: dict[str, Any] = {}
        for role, key in self.camera_keys.items():
            if role not in images:
                raise ValueError(f"images에 '{role}' 키가 없습니다 (camera_keys={self.camera_keys})")
            obs[key] = _image_hwc_uint8_to_chw_float(images[role])

        state_t = state if isinstance(state, torch.Tensor) else torch.as_tensor(np.asarray(state))
        state_t = state_t.to(dtype=torch.float32)
        if state_t.shape != (STATE_DIM,):
            raise ValueError(f"state는 ({STATE_DIM},) 모양이어야 합니다 (받은 shape: {tuple(state_t.shape)})")
        obs[OBS_STATE] = state_t
        obs["task"] = task

        batch = self.preprocessor(obs)
        with torch.no_grad():
            action = self.policy.select_action(batch)  # (1, action_dim) 또는 (action_dim,) -- 아래서 통일
        action = self.postprocessor(action)
        if action.ndim == 2:
            action = action[0]
        if action.shape[-1] != ACTION_DIM:
            raise RuntimeError(
                f"policy 출력 차원({action.shape[-1]})이 기대한 {ACTION_DIM}과 다릅니다 — "
                "policy.config.action_feature.shape 확인 필요."
            )
        return action


def replay_episode(adapter: RoCoPolicyAdapter, frames: list[dict[str, Any]]) -> dict[str, Any]:
    """오프라인 리플레이 하네스: LeRobot 포맷 에피소드(프레임 dict의 리스트, 시간 순서)를
    한 프레임씩 `adapter.act()`에 넣고, 예측 액션과 그 프레임의 정답 액션(`frame["action"]`)
    사이의 차원별 L1(|pred - true|, 프레임 평균)을 계산한다.

    각 frame dict은 `adapter.camera_keys`의 값들(기본 "observation.images.head" 등)을 키로
    갖는 이미지((H,W,3) uint8), `lerobot.utils.constants.OBS_STATE`(기본
    "observation.state", (44,) float), `"action"`((14,) float, 정답), `"task"`(문자열)을
    담고 있어야 한다 — 실제 LeRobot 데이터셋(`LeRobotDataset`)의 `__getitem__`이 주는 것과
    같은 키 이름이다(이 함수 자체는 `LeRobotDataset`을 import하지 않는다 — 호출자가 이미
    로드한 프레임 리스트를 넘긴다는 전제).

    환경(실제 로봇/시뮬레이터)과 아예 통신하지 않는다 — "정답 액션을 다시 실행했을 때 실제로
    어떻게 되는지"가 아니라 "정책이 그 상황에서 무엇을 예측했을지"만 본다(오프라인 모방학습
    평가의 표준적인 한계, RoCo 실제 리더보드 채점 방식과는 다를 수 있음 — 미확인).

    반환: {"per_dim_l1": (14,) ndarray, "mean_l1": float, "n_frames": int,
    "predicted_actions": (n_frames,14) ndarray, "true_actions": (n_frames,14) ndarray}"""
    if not frames:
        raise ValueError("frames가 비어 있습니다")

    predicted, true = [], []
    adapter.reset()
    for frame in frames:
        images = {role: frame[key] for role, key in adapter.camera_keys.items()}
        pred = adapter.act(images=images, state=frame[OBS_STATE], task=frame["task"])
        predicted.append(pred.detach().cpu().numpy())
        true.append(np.asarray(frame["action"], dtype=np.float32))

    predicted = np.stack(predicted)
    true = np.stack(true)
    per_dim_l1 = np.abs(predicted - true).mean(axis=0)
    return {
        "per_dim_l1": per_dim_l1,
        "mean_l1": float(per_dim_l1.mean()),
        "n_frames": len(frames),
        "predicted_actions": predicted,
        "true_actions": true,
    }
