"""정규화 통계를 **학습 에피소드만**으로 계산하고, `action_min_std`/`state_min_std`
하한을 적용한 뒤 "어느 채널이 하한에 걸렸는지"를 사람이 읽을 수 있게 보고한다.

하한 적용 자체는 새로 만들지 않고 `processor_smolvla.py`의 `_apply_action_min_std`/
`_apply_state_min_std`를 그대로 재사용한다(이미 "원본은 안 바꾸고 새 dict를 반환" /
"state_select 인덱스로 raw 위치에 scatter" 같은 까다로운 부분이 검증돼 있는 함수들 —
`tests/test_configurable_ard.py` 참고, 여기서 또 만들 이유가 없다)."""

from __future__ import annotations

import dataclasses
from typing import Iterable

import torch

from lerobot.utils.constants import ACTION, OBS_STATE


def compute_train_stats(
    frames: Iterable[dict], keys: tuple[str, ...] = (OBS_STATE, ACTION)
) -> dict[str, dict[str, torch.Tensor]]:
    """`frames`(각 원소가 `{OBS_STATE: Tensor(D,), ACTION: Tensor(A,), ...}` 형태인 단일
    프레임 딕셔너리 — delta_timestamps 없이 `LeRobotDataset(..., episodes=train_ids)`를
    순회하면 이 형태가 나온다)에서 채널별 mean/std를 스트리밍으로 계산한다. 전체를
    메모리에 쌓지 않고 합/제곱합만 누적하므로 에피소드 수가 많아도 메모리에 비례하지
    않는다. `frames`가 실제 `LeRobotDataset`이 아니라 평범한 dict 리스트/제너레이터여도
    동작한다(단위 테스트는 이 형태로 한다 — 실제 데이터셋 순회는 Hub 접근이 필요해
    이 샌드박스에서 실행해보지 못했다, 미검증)."""
    sums: dict[str, torch.Tensor] = {}
    sumsqs: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for frame in frames:
        for key in keys:
            if key not in frame:
                continue
            value = torch.as_tensor(frame[key], dtype=torch.float64).reshape(-1)
            if key not in sums:
                sums[key] = torch.zeros_like(value)
                sumsqs[key] = torch.zeros_like(value)
                counts[key] = 0
            sums[key] += value
            sumsqs[key] += value * value
            counts[key] += 1

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key in keys:
        if key not in sums or counts[key] == 0:
            continue
        n = counts[key]
        mean = sums[key] / n
        variance = (sumsqs[key] / n - mean * mean).clamp_min(0.0)
        std = variance.sqrt()
        stats[key] = {"mean": mean.to(torch.float32), "std": std.to(torch.float32)}
    return stats


@dataclasses.dataclass
class FlooredDim:
    index: int
    name: str
    original_std: float
    floored_std: float


def _floored_dims(original_std: torch.Tensor, new_std: torch.Tensor, names: list[str] | None, eps: float = 1e-12) -> list[FlooredDim]:
    out = []
    for i in range(original_std.shape[-1]):
        old_v = float(original_std[i])
        new_v = float(new_std[i])
        if new_v > old_v + eps:
            label = names[i] if names and i < len(names) else f"dim_{i}"
            out.append(FlooredDim(index=i, name=label, original_std=old_v, floored_std=new_v))
    return out


@dataclasses.dataclass
class MinStdReport:
    action_floored: list[FlooredDim]
    state_floored: list[FlooredDim]

    def summary_lines(self) -> list[str]:
        lines = [f"action_min_std 하한에 걸린 채널: {len(self.action_floored)}개"]
        for d in self.action_floored:
            lines.append(f"  action[{d.index}] {d.name}: std {d.original_std:.6g} -> {d.floored_std:.6g}")
        lines.append(f"state_min_std 하한에 걸린 채널: {len(self.state_floored)}개")
        for d in self.state_floored:
            lines.append(f"  state[{d.index}] {d.name}: std {d.original_std:.6g} -> {d.floored_std:.6g}")
        return lines


def apply_min_std_with_report(
    stats: dict[str, dict[str, torch.Tensor]],
    action_min_std: float | list[float] | None,
    state_min_std: float | list[float] | None,
    state_select: list[int] | None,
    action_names: list[str] | None = None,
    state_names: list[str] | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], MinStdReport]:
    from lerobot.policies.smolvla.processor_smolvla import _apply_action_min_std, _apply_state_min_std

    original_action_std = stats.get(ACTION, {}).get("std")
    original_state_std = stats.get(OBS_STATE, {}).get("std")

    new_stats = _apply_action_min_std(stats, action_min_std)
    new_stats = _apply_state_min_std(new_stats, state_min_std, state_select)

    action_floored: list[FlooredDim] = []
    if original_action_std is not None and ACTION in new_stats:
        action_floored = _floored_dims(original_action_std, new_stats[ACTION]["std"], action_names)

    state_floored: list[FlooredDim] = []
    if original_state_std is not None and OBS_STATE in new_stats:
        state_floored = _floored_dims(original_state_std, new_stats[OBS_STATE]["std"], state_names)

    return new_stats, MinStdReport(action_floored=action_floored, state_floored=state_floored)
