#!/usr/bin/env python3
"""검증 에피소드의 "접촉 직전 구간"만 따로 잘라서 정밀도를 본다.

전체 에피소드 평균 L1만 보면, 접촉/삽입처럼 정밀도가 실제로 중요한 짧은 구간의 오차가
나머지 긴 이동 구간 평균에 묻혀 버린다. 이 스크립트는 `policy_interface.replay_episode()`
(이미 테스트된, 정책으로 매 프레임 액션을 예측하는 리플레이 하네스)를 그대로 재사용해서
예측/정답 액션 시퀀스를 얻은 뒤, 그 위에 **구간 자동 분리 + mm 환산 + 팔별 분리**만
추가한다(ponytail: L1 계산 자체는 다시 만들지 않는다 — `scripts/colab/eval_offline.py`의
`ACTION_DIM_GROUPS`를 그대로 재사용).

**구간 분리 기준** (코드에 기본값 명시, `--window-criterion`으로 변경 가능):
- `gripper_change`(기본): 그리퍼 값이 연속 프레임 사이에서 가장 크게 바뀌는 지점을
  "접촉 이벤트"로 보고, 그 프레임을 중심으로 `--window-frames`(기본 20)프레임을 자른다.
  그리퍼가 열림->닫힘(또는 반대)으로 바뀌는 순간이 접촉/파지 시도와 가장 직접적으로
  연결된 신호라서 기본값으로 골랐다.
- `min_height`: 대신 끝점(z) 높이가 최소인 프레임(보드에 가장 가까워진 지점)을 중심으로
  자른다 — 그리퍼 상태 변화가 없는 과제(예: 누르기만 하는 동작)에 대한 대안.

**접촉 팔 선택** (`--contact-arm`, 기본 `auto`): auto면 그리퍼 값의 총 변화량
(`sum(|Δgripper|)`)이 더 큰 팔을 "실제로 접촉 동작을 하는 팔"로 자동 선택한다
(RoCo 프리셋은 왼팔=Actuator라 보통 왼팔이 선택될 것으로 예상 — 자동 판별이라 과제마다
달라져도 그대로 동작한다).

사용 예:
    python scripts/eval_precision.py --checkpoint-path outputs/ard_train/step_0001000 \\
        --dataset-repo-id rocochallenge2025/rocochallenge2026_Industrial_Assembly --episode-index 0

주의 — 이 레포 샌드박스는 Hugging Face Hub 접근이 막혀 있어 `--dataset-repo-id`로 실제
에피소드를 받는 경로(`load_episode_frames()`)는 실행해보지 못했다(미검증) — 특히
`LeRobotDataset`이 돌려주는 이미지 텐서가 실제로 CHW float [0,1]인지(HWC uint8로 되돌리는
`_tensor_to_hwc_uint8`의 전제)는 실제 데이터셋을 열어봐야 확정된다. 구간 분리/mm 환산/
팔별 집계 로직(`compute_precision_report()` 이하)은 `tests/test_eval_precision.py`가
합성 프레임으로 직접 검증했다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (policy_interface 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parent / "colab"))  # scripts/colab (eval_offline 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lerobot" / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from eval_offline import ACTION_DIM_GROUPS  # noqa: E402
from policy_interface import RoCoPolicyAdapter, replay_episode  # noqa: E402

ARMS = ("left", "right")
Z_OFFSET_IN_POS = 2  # ACTION_DIM_GROUPS["{arm}_pos"]는 (x,y,z) 3채널 — z는 그 안의 세 번째


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True, help="SmolVLAPolicy.from_pretrained()로 불러올 저장 경로")
    parser.add_argument("--dataset-repo-id", default="rocochallenge2025/rocochallenge2026_Industrial_Assembly")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--task", default=None, help="주면 데이터셋의 task 대신 모든 프레임에 이 문자열을 쓴다")
    parser.add_argument(
        "--window-criterion", choices=["gripper_change", "min_height"], default="gripper_change",
        help="접촉 직전 구간을 고르는 기준 (기본: 그리퍼 상태가 가장 크게 바뀌는 프레임 중심)",
    )
    parser.add_argument(
        "--contact-arm", choices=["left", "right", "auto"], default="auto",
        help="auto면 그리퍼 총 변화량이 더 큰 팔을 '접촉 팔'로 자동 선택",
    )
    parser.add_argument("--window-frames", type=int, default=20, help="접촉 이벤트 프레임을 중심으로 앞뒤로 자를 총 프레임 수")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-path", default="outputs/eval_precision.json")
    return parser.parse_args()


def detect_gripper_change_frame(gripper_values: np.ndarray) -> int:
    """그리퍼 값이 연속 프레임 사이에서 가장 크게 바뀐 **직후** 프레임의 인덱스."""
    if len(gripper_values) < 2:
        return 0
    deltas = np.abs(np.diff(gripper_values))
    return int(np.argmax(deltas)) + 1


def detect_min_height_frame(z_values: np.ndarray) -> int:
    """끝점(z) 높이가 최소인 프레임의 인덱스."""
    return int(np.argmin(z_values))


def select_contact_arm(true_actions: np.ndarray) -> str:
    """그리퍼 총 변화량(`sum(|Δgripper|)`)이 더 큰 팔을 고른다."""
    left_gripper = true_actions[:, ACTION_DIM_GROUPS["left_gripper"]][:, 0]
    right_gripper = true_actions[:, ACTION_DIM_GROUPS["right_gripper"]][:, 0]
    left_var = float(np.abs(np.diff(left_gripper)).sum())
    right_var = float(np.abs(np.diff(right_gripper)).sum())
    return "left" if left_var >= right_var else "right"


def select_contact_window(true_actions: np.ndarray, arm: str, criterion: str, window_frames: int) -> tuple[int, int]:
    """[start, end) 인덱스 — `true_actions`(정답, (T,14))에서 지정된 팔/기준으로 접촉 직전
    구간을 고른다(정답 기준으로 고르는 이유: 모델 예측은 틀릴 수 있어서 "실제로 접촉이
    일어난 지점"은 정답 쪽에서 찾아야 함)."""
    if criterion == "gripper_change":
        gripper = true_actions[:, ACTION_DIM_GROUPS[f"{arm}_gripper"]][:, 0]
        center = detect_gripper_change_frame(gripper)
    elif criterion == "min_height":
        z = true_actions[:, ACTION_DIM_GROUPS[f"{arm}_pos"]][:, Z_OFFSET_IN_POS]
        center = detect_min_height_frame(z)
    else:
        raise ValueError(f"알 수 없는 window_criterion: {criterion!r}")
    half = window_frames // 2
    n = true_actions.shape[0]
    start = max(0, min(center - half, n - 1))
    end = max(start + 1, min(center + half, n))
    return start, end


def arm_errors(pred: np.ndarray, true: np.ndarray, arm: str) -> dict:
    """`pred`/`true`: (T, 14). 지정된 팔의 위치 L1(mm)/회전 L1(rad)/그리퍼 L1 평균
    (슬라이스는 `eval_offline.ACTION_DIM_GROUPS`를 그대로 재사용)."""
    pos_sl, rot_sl, grip_sl = (ACTION_DIM_GROUPS[f"{arm}_{g}"] for g in ("pos", "rot", "gripper"))
    pos_l1_m = np.abs(pred[:, pos_sl] - true[:, pos_sl]).mean()
    rot_l1_rad = np.abs(pred[:, rot_sl] - true[:, rot_sl]).mean()
    gripper_l1 = np.abs(pred[:, grip_sl] - true[:, grip_sl]).mean()
    return {
        "pos_l1_mm": float(pos_l1_m * 1000.0),
        "rot_l1_rad": float(rot_l1_rad),
        "gripper_l1": float(gripper_l1),
        "n_frames": int(pred.shape[0]),
    }


def compute_precision_report(
    adapter: RoCoPolicyAdapter,
    frames: list[dict],
    window_criterion: str = "gripper_change",
    contact_arm: str = "auto",
    window_frames: int = 20,
) -> dict:
    replay = replay_episode(adapter, frames)
    true_actions = replay["true_actions"]
    pred_actions = replay["predicted_actions"]

    arm = select_contact_arm(true_actions) if contact_arm == "auto" else contact_arm
    start, end = select_contact_window(true_actions, arm, window_criterion, window_frames)

    return {
        "n_frames_total": int(true_actions.shape[0]),
        "contact_arm": arm,
        "window_criterion": window_criterion,
        "window": [start, end],
        "full_episode": {a: arm_errors(pred_actions, true_actions, a) for a in ARMS},
        "contact_window": {a: arm_errors(pred_actions[start:end], true_actions[start:end], a) for a in ARMS},
    }


def _tensor_to_hwc_uint8(img) -> np.ndarray:
    """LeRobotDataset이 돌려주는 이미지를 `RoCoPolicyAdapter`가 기대하는 (H,W,3) uint8으로
    되돌린다. **미검증**: 실제 LeRobotDataset 이미지 텐서가 CHW float [0,1]이라는 전제는
    이 샌드박스에서 실제로 열어본 적이 없다(Hub 차단) — 이미 uint8 HWC라면 그대로 통과."""
    if torch.is_tensor(img):
        img = img.numpy()
    if img.dtype != np.uint8:
        img = (np.transpose(img, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    return img


def load_episode_frames(
    dataset_repo_id: str, episode_index: int, camera_keys: dict[str, str], task_override: str | None
) -> list[dict]:
    """`--dataset-repo-id`/`--episode-index`로 실제 에피소드 하나를 받아
    `policy_interface.replay_episode()`가 기대하는 프레임 dict 리스트로 바꾼다."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, episodes=[episode_index])
    frames = []
    for item in dataset:
        frame = {key: _tensor_to_hwc_uint8(item[key]) for key in camera_keys.values()}
        state = item[OBS_STATE]
        frame[OBS_STATE] = state.numpy() if torch.is_tensor(state) else np.asarray(state)
        action = item["action"]
        frame["action"] = action.numpy() if torch.is_tensor(action) else np.asarray(action)
        frame["task"] = task_override if task_override is not None else item.get("task", "")
        frames.append(frame)
    return frames


def print_report(report: dict) -> None:
    start, end = report["window"]
    print(
        f"접촉 구간: {report['window_criterion']} 기준, contact_arm={report['contact_arm']}, "
        f"프레임 [{start}, {end}) (총 {report['n_frames_total']}프레임 중)"
    )
    print(f"\n{'':6} {'구간':10} {'위치 L1(mm)':>12} {'회전 L1(rad)':>13} {'그리퍼 L1':>10}")
    for arm in ARMS:
        for group, label in (("full_episode", "전체 평균"), ("contact_window", "접촉 구간")):
            e = report[group][arm]
            print(f"{arm:6} {label:10} {e['pos_l1_mm']:>12.3f} {e['rot_l1_rad']:>13.4f} {e['gripper_l1']:>10.4f}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_path)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)

    frames = load_episode_frames(args.dataset_repo_id, args.episode_index, adapter.camera_keys, args.task)
    report = compute_precision_report(
        adapter, frames,
        window_criterion=args.window_criterion, contact_arm=args.contact_arm, window_frames=args.window_frames,
    )
    print_report(report)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
