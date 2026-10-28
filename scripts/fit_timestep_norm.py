#!/usr/bin/env python3
"""`per_timestep_norm`(configuration_smolvla.py)용 σ(t) = a + s*sqrt(t) 피팅 스크립트.

청크 내 위치(실행 스텝 인덱스) t가 클수록(먼 미래일수록) 정규화된 액션의 분산이 커지는
경향이 있다는 관찰(LeHome 우승 보고서, arXiv 2606.27163)을 바탕으로, 실제 데이터셋에서
위치별 표준편차 std(t)를 측정하고 std(t) ≈ a + s*sqrt(t) 형태로 최소제곱 피팅한다.

**중요**: 이 피팅은 표준 차원별(평균/표준편차) 정규화를 이미 거친 액션(= policy가 실제로
보는 입력) 위에서 수행한다 — dataset.meta.stats[ACTION]의 mean/std로 직접 정규화해서
계산하고(NormalizerProcessorStep과 동일한 계산, 이미지/토크나이저 파이프라인은 거치지
않는다), 전체 이미지/언어 파이프라인을 돌리지 않아 빠르다.

이 레포의 샌드박스는 Hugging Face Hub 접근이 막혀 있어 RoCo Challenge 공식 데이터셋
(`rocochallenge2025/rocochallenge2026_Industrial_Assembly`)으로 **실행해보지 못했다** —
실제 GPU/Hub 접근이 있는 로컬 환경에서 돌려서 configs/presets/roco_board.yaml의
per_timestep_norm_a/_s 값을 갱신해야 한다(README/docs의 "검증됨 vs 미검증" 참고).

사용 예:
    python scripts/fit_timestep_norm.py --dataset-repo-id <HF_USER>/<DATASET> --chunk-size 20

출력(JSON)을 그대로 SmolVLAConfig(per_timestep_norm_a=..., per_timestep_norm_s=...)에
붙여넣거나 configs/presets/*.yaml에 넣으면 된다.
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobotDataset repo id (예: <user>/<dataset>)")
    parser.add_argument("--chunk-size", type=int, default=50, help="SmolVLAConfig.chunk_size와 동일한 값을 줘야 한다")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="전체 데이터셋이 너무 크면 앞쪽 N배치만 사용")
    parser.add_argument(
        "--per-dim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="채널마다 다른 a/s를 피팅한다(기본은 전체 채널에 동일한 스칼라 a/s 하나).",
    )
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 경로 (생략하면 stdout에만 출력)")
    return parser.parse_args()


def _fit_a_s(std_per_t: torch.Tensor) -> tuple[float, float]:
    """std_per_t: (chunk_size,). std(t) ≈ a + s*sqrt(t) 최소제곱 피팅, (a, s) 반환."""
    chunk_size = std_per_t.shape[0]
    t = torch.arange(chunk_size, dtype=torch.float64)
    design = torch.stack([torch.ones_like(t), torch.sqrt(t)], dim=1)  # (chunk_size, 2)
    solution = torch.linalg.lstsq(design, std_per_t.to(torch.float64)).solution
    a, s = solution.tolist()
    return a, s


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    logging.info("데이터셋 메타데이터 로드 중: %s", args.dataset_repo_id)
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    if ACTION not in output_features:
        raise ValueError(f"데이터셋 {args.dataset_repo_id!r}에 '{ACTION}' feature가 없습니다.")
    action_dim = output_features[ACTION].shape[0]

    # delta_timestamps 계산용 — chunk_size만큼 액션 시퀀스를 통째로 받아오기 위함
    # (scripts/train_ard.py와 동일한 이유, 그 파일의 주석 참고).
    config = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size=args.chunk_size)
    delta_timestamps = resolve_delta_timestamps(config, ds_meta)
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)

    mean = torch.as_tensor(dataset.meta.stats[ACTION]["mean"], dtype=torch.float64).reshape(-1)
    std = torch.as_tensor(dataset.meta.stats[ACTION]["std"], dtype=torch.float64).reshape(-1)
    std = std.clamp_min(1e-8)  # 0으로 나누기 방지 — 상수 채널(그리퍼 등)에서 발생 가능

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 위치별로 "정규화된 액션" 값을 전부 모아서 std(t)를 구한다. 메모리를 아끼려고 welford 같은
    # 누적 알고리즘을 쓰는 대신(chunk_size*action_dim이 작아서 괜찮음) 그냥 리스트로 모은다 —
    # ponytail: 데이터셋이 매우 크면(--max-batches로 캡 걸 수 있음) 메모리가 부족할 수 있다.
    collected: list[torch.Tensor] = []
    for i, batch in enumerate(dataloader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        actions = batch[ACTION].to(torch.float64)  # (B, chunk_size, action_dim)
        normalized = (actions - mean) / std
        collected.append(normalized)
    if not collected:
        raise RuntimeError("데이터셋에서 배치를 하나도 읽지 못했습니다.")
    all_actions = torch.cat(collected, dim=0)  # (N, chunk_size, action_dim)
    logging.info("표준편차 계산에 사용한 샘플 수: %d (chunk_size=%d, action_dim=%d)", all_actions.shape[0], args.chunk_size, action_dim)

    if args.per_dim:
        std_per_t_dim = all_actions.std(dim=0)  # (chunk_size, action_dim)
        a_list, s_list = [], []
        for d in range(action_dim):
            a, s = _fit_a_s(std_per_t_dim[:, d])
            a_list.append(a)
            s_list.append(s)
        result = {"per_timestep_norm_a": a_list, "per_timestep_norm_s": s_list}
    else:
        std_per_t = all_actions.std(dim=(0, 2))  # (chunk_size,) — 채널 평균
        a, s = _fit_a_s(std_per_t)
        result = {"per_timestep_norm_a": a, "per_timestep_norm_s": s}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info("결과를 %s에 저장했습니다.", args.output)


if __name__ == "__main__":
    main()
