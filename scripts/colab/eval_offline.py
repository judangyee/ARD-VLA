"""보류 평가 에피소드에서 청크 예측 vs 정답의 차원별 L1, 그리고 두 베이스라인
("평균 action 상수 예측", "직전 action 유지")을 같이 계산해서 JSON으로 남긴다.

action 14차원 그룹은 `docs/configurable_ard.md`가 확인한 순서
(`left_ee_x..rz, left_gripper, right_ee_x..rz, right_gripper`)를 그대로 쓴다."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

import torch

ACTION_DIM_GROUPS: dict[str, slice] = {
    "left_pos": slice(0, 3),
    "left_rot": slice(3, 6),
    "left_gripper": slice(6, 7),
    "right_pos": slice(7, 10),
    "right_rot": slice(10, 13),
    "right_gripper": slice(13, 14),
}


@dataclasses.dataclass
class EvalSample:
    """평가 1건(보통 평가 에피소드 하나에서 뽑은 청크 하나) — 모델 추론 입력(`batch`,
    `preprocessor` 적용 전 raw 배치, batch_size=1 가정)과, 그 청크 구간의 정답
    (`target_chunk`, (chunk_size, action_dim), raw 단위), 그 청크가 시작되기 바로 직전에
    실제로 관측된 action(`prev_action`, (action_dim,), raw 단위 — "직전 action 유지"
    베이스라인용)."""

    batch: dict
    target_chunk: torch.Tensor
    prev_action: torch.Tensor


def l1_by_group(pred: torch.Tensor, target: torch.Tensor, groups: dict[str, slice] = ACTION_DIM_GROUPS) -> dict[str, float]:
    """`pred`/`target`: (..., action_dim). 그룹별 평균 L1 + `overall`을 돌려준다."""
    diff = (pred - target).abs()
    out = {name: float(diff[..., sl].mean().item()) for name, sl in groups.items()}
    out["overall"] = float(diff.mean().item())
    return out


def _average_group_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def compute_baselines(
    train_mean_action: torch.Tensor, samples: list[EvalSample], groups: dict[str, slice] = ACTION_DIM_GROUPS
) -> dict[str, dict[str, float]]:
    """`train_mean_action`: 학습 에피소드만으로 계산한 action의 채널별 평균, (action_dim,).
    "평균 action 상수 예측"/"직전 action 유지" 두 베이스라인을 각 샘플의 청크 길이에
    맞춰 브로드캐스트해서 정답과 비교한다."""
    const_l1, repeat_l1 = [], []
    for sample in samples:
        chunk_size = sample.target_chunk.shape[0]
        const_pred = train_mean_action.unsqueeze(0).expand(chunk_size, -1)
        repeat_pred = sample.prev_action.unsqueeze(0).expand(chunk_size, -1)
        const_l1.append(l1_by_group(const_pred, sample.target_chunk, groups))
        repeat_l1.append(l1_by_group(repeat_pred, sample.target_chunk, groups))
    return {
        "constant_mean": _average_group_dicts(const_l1),
        "repeat_last": _average_group_dicts(repeat_l1),
    }


@dataclasses.dataclass
class OfflineEvalResult:
    num_samples: int
    model: dict[str, float]
    constant_mean_baseline: dict[str, float]
    repeat_last_baseline: dict[str, float]

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_offline_eval(
    policy,
    preprocessor,
    postprocessor,
    samples: Iterable[EvalSample],
    train_mean_action: torch.Tensor,
    device: torch.device | str = "cpu",
) -> OfflineEvalResult:
    """실제 정책으로 추론해서 모델 L1 + 두 베이스라인 L1을 계산한다.

    `policy.predict_action_chunk(batch)`는 정규화된 공간의 액션을 돌려주므로(학습 때
    쓰는 `dataset_stats` 기준), `postprocessor`로 raw 단위(m/rad)로 되돌린 뒤에야
    `target_chunk`(raw 단위, 데이터셋에서 그대로 읽은 값)와 비교할 수 있다 —
    `closed_loop.py`가 실행 직전에 쓰는 것과 같은 패턴(`postprocessor(actions)`)."""
    policy.eval()
    samples = list(samples)

    model_l1_per_sample: list[dict[str, float]] = []
    for sample in samples:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.batch.items()}
        batch = preprocessor(batch)
        pred_chunk = policy.predict_action_chunk(batch)
        pred_chunk = postprocessor(pred_chunk)
        pred_chunk = pred_chunk.squeeze(0).to("cpu")

        target = sample.target_chunk.to("cpu")
        chunk_size = min(pred_chunk.shape[0], target.shape[0])
        model_l1_per_sample.append(l1_by_group(pred_chunk[:chunk_size], target[:chunk_size]))

    baselines = compute_baselines(train_mean_action.to("cpu"), samples)

    return OfflineEvalResult(
        num_samples=len(samples),
        model=_average_group_dicts(model_l1_per_sample),
        constant_mean_baseline=baselines["constant_mean"],
        repeat_last_baseline=baselines["repeat_last"],
    )


def save_eval_report(path: str | Path, result: OfflineEvalResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))
