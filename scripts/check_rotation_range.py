#!/usr/bin/env python3
"""회전 채널(rx/ry/rz 등)이 실제 데이터에서 ±pi 경계를 넘는지 점검하는 스크립트.

`configuration_smolvla.py`의 `angle_unwrap_dims`(processor_smolvla.py의
`AngleUnwrapProcessorStep`/`AngleRewrapProcessorStep`)를 켤지 말지는, 실제 데이터가
경계를 넘는 케이스가 있는지에 따라 결정해야 한다 — 안 넘으면 unwrap은 아무 효과가
없고(또는 거의 없고), 넘는다면 켜는 게 맞다. 이 레포의 샌드박스는 Hugging Face Hub
접근이 막혀 있어 이 스크립트를 실행해보지 못했다 — Hub 접근이 되는 환경(로컬/CI)에서
실행해서 결정해야 한다.

에피소드별로:
- 지정 채널(`--angle-dims`, 기본 [3,4,5,10,11,12] — RoCo의 왼팔/오른팔 rx,ry,rz)의
  최소/최대값
- 연속 프레임 간 차이가 `--jump-threshold`(기본 pi)를 넘는 "경계 도약" 횟수
- (채널 수가 3의 배수면) 3개씩 묶어 "삼중값" 노름(||[rx,ry,rz]||)의 최소/최대/평균 —
  **오일러각 vs 회전벡터(axis-angle) 판별에 참고**: 회전벡터는 보통 노름이 회전각 자체라서
  정준(canonical) 표현이면 거의 항상 [0, pi] 안에 들어온다. 노름이 거의 항상 이 범위
  안에 있고 각 축이 "독립적으로" 풀 레인지를 쓰는 게 아니라 셋이 얽혀 움직인다면 회전벡터
  쪽에 더 가깝다. 반대로 노름이 pi를 자주/크게 넘거나 각 채널이 서로 무관하게 독립적인
  범위를 쓰면 오일러각(혹은 다른 분해)에 더 가깝다. **이 추론은 참고용 힌트일 뿐 확정적
  증거가 아니다** — 실제 데이터 생성 코드/스키마 문서를 직접 확인하는 게 최종 근거다
  (docs/configurable_ard.md 참고).
를 출력한다. 이미지/비디오는 전혀 디코딩하지 않는다(액션은 비디오가 아니라 일반
테이블 컬럼이라 `dataset.hf_dataset`에서 직접 읽는다 — 빠름).

사용 예:
    python scripts/check_rotation_range.py --dataset-repo-id <HF_USER>/<DATASET>
    python scripts/check_rotation_range.py \
        --dataset-repo-id rocochallenge2025/rocochallenge2026_Industrial_Assembly \
        --angle-dims 3 4 5 10 11 12
"""

from __future__ import annotations

import argparse
import logging
import math

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobotDataset repo id (예: <user>/<dataset>)")
    parser.add_argument(
        "--angle-dims", type=int, nargs="+", default=[3, 4, 5, 10, 11, 12],
        help="회전(라디안) 채널의 절대 인덱스. 기본은 RoCo 14차원 스키마의 왼팔/오른팔 rx,ry,rz.",
    )
    parser.add_argument(
        "--jump-threshold", type=float, default=math.pi,
        help="연속 프레임 간 절댓값 차이가 이 값을 넘으면 '경계 도약'으로 센다(기본: pi).",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="앞쪽 N개 에피소드만 점검(큰 데이터셋용)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    logging.info("데이터셋 로드 중(테이블 컬럼만, 비디오 디코딩 없음): %s", args.dataset_repo_id)
    dataset = LeRobotDataset(args.dataset_repo_id)

    if ACTION not in dataset.hf_dataset.column_names:
        raise ValueError(f"데이터셋에 '{ACTION}' 컬럼이 없습니다. 컬럼: {dataset.hf_dataset.column_names}")

    actions = torch.stack([torch.as_tensor(a) for a in dataset.hf_dataset[ACTION]])  # (N, action_dim)
    episode_index = torch.as_tensor(dataset.hf_dataset["episode_index"])  # (N,)
    action_dim = actions.shape[-1]
    if any(d < 0 or d >= action_dim for d in args.angle_dims):
        raise ValueError(f"--angle-dims {args.angle_dims}가 액션 차원({action_dim}) 범위를 벗어납니다.")

    episode_ids = episode_index.unique().tolist()
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    logging.info(
        "점검 대상: 에피소드 %d개, 채널 %s, jump-threshold=%.4f rad",
        len(episode_ids), args.angle_dims, args.jump_threshold,
    )

    overall_min = torch.full((len(args.angle_dims),), float("inf"))
    overall_max = torch.full((len(args.angle_dims),), float("-inf"))
    overall_jumps = [0] * len(args.angle_dims)
    episodes_with_jumps = 0

    for ep in episode_ids:
        mask = episode_index == ep
        ep_actions = actions[mask][:, args.angle_dims]  # (T, len(angle_dims))
        if ep_actions.shape[0] == 0:
            continue

        ep_min = ep_actions.min(dim=0).values
        ep_max = ep_actions.max(dim=0).values
        overall_min = torch.minimum(overall_min, ep_min)
        overall_max = torch.maximum(overall_max, ep_max)

        if ep_actions.shape[0] >= 2:
            diffs = (ep_actions[1:] - ep_actions[:-1]).abs()  # (T-1, len(angle_dims))
            jumps_per_dim = (diffs > args.jump_threshold).sum(dim=0)
            if jumps_per_dim.sum() > 0:
                episodes_with_jumps += 1
            for i, j in enumerate(jumps_per_dim.tolist()):
                overall_jumps[i] += j

    print(f"\n=== {args.dataset_repo_id} — 회전 채널 범위/경계 도약 점검 ===")
    print(f"에피소드 수: {len(episode_ids)}, 전체 프레임 수: {actions.shape[0]}\n")
    for i, dim in enumerate(args.angle_dims):
        print(
            f"  채널[{dim}]: min={overall_min[i]:.4f} max={overall_max[i]:.4f} "
            f"(범위 {overall_max[i] - overall_min[i]:.4f} rad) — 경계 도약(>{args.jump_threshold:.3f} rad) "
            f"{overall_jumps[i]}회"
        )
    print(f"\n경계 도약이 1회 이상 발생한 에피소드 수: {episodes_with_jumps} / {len(episode_ids)}")

    if len(args.angle_dims) % 3 == 0:
        print("\n--- 삼중값(rx,ry,rz) 노름 — 오일러각/회전벡터 판별 참고용 ---")
        all_rot = actions[:, args.angle_dims]  # (N, len(angle_dims))
        for g in range(len(args.angle_dims) // 3):
            triplet = all_rot[:, 3 * g : 3 * g + 3]
            norms = triplet.norm(dim=-1)
            over_pi = (norms > math.pi).float().mean().item() * 100
            dims = args.angle_dims[3 * g : 3 * g + 3]
            print(
                f"  채널{dims}: ||v|| min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f} "
                f"— 전체 프레임의 {over_pi:.1f}%가 ||v||>pi"
            )
        print(
            "  (참고) ||v||가 거의 항상 <= pi면 회전벡터(axis-angle)일 가능성 — 그렇다면 "
            "angle_unwrap_dims를 이 채널들에 켜는 건 부적절하다(세 채널을 독립적으로 "
            "unwrap하면 안 됨, docs/configurable_ard.md 참고). ||v||가 pi를 자주 넘거나 "
            "각 채널 범위가 서로 독립적이면 오일러각에 더 가깝다."
        )

    print(
        "\n판단 기준: 도약이 거의 없으면(0에 가까움) angle_unwrap_dims를 켤 필요가 없다 — "
        "실제로 pi 경계를 넘는 궤적이 드물다는 뜻이다. 여러 에피소드에서 반복적으로 도약이 "
        "보이면, 해당 채널이 ±pi를 넘나드는 회전을 포함한다는 뜻이라 켜는 쪽을 검토해야 한다. "
        "단, 위 삼중값 노름 결과가 회전벡터를 가리키면 도약이 있어도 켜면 안 된다 — 그 경우는 "
        "SO(3)-인식 처리(쿼터니언/회전행렬 연속성 등, 이 스크립트의 범위 밖)가 필요하다. "
        "(이 스크립트는 '넘는지 여부'와 '어느 표현인지 힌트'만 보여준다 — 최종 판단은 데이터 "
        "생성 코드 확인 + 켰을 때 실제로 학습이 좋아지는지 A/B 비교로 해야 한다.)"
    )


if __name__ == "__main__":
    main()
