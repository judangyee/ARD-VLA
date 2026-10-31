"""scripts/colab/norm_stats.py — 학습 전용 통계 계산 + min_std 하한 리포트."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

from norm_stats import apply_min_std_with_report, compute_train_stats  # noqa: E402

from lerobot.utils.constants import ACTION, OBS_STATE


def test_compute_train_stats_matches_closed_form_mean_std():
    torch.manual_seed(0)
    frames = [{OBS_STATE: torch.randn(4), ACTION: torch.randn(3)} for _ in range(500)]
    stats = compute_train_stats(frames)

    all_state = torch.stack([f[OBS_STATE] for f in frames])
    all_action = torch.stack([f[ACTION] for f in frames])

    assert torch.allclose(stats[OBS_STATE]["mean"], all_state.mean(dim=0), atol=1e-4)
    assert torch.allclose(stats[OBS_STATE]["std"], all_state.std(dim=0, unbiased=False), atol=1e-4)
    assert torch.allclose(stats[ACTION]["mean"], all_action.mean(dim=0), atol=1e-4)
    assert torch.allclose(stats[ACTION]["std"], all_action.std(dim=0, unbiased=False), atol=1e-4)


def test_compute_train_stats_ignores_frames_not_containing_keys():
    frames = [{OBS_STATE: torch.ones(2)} for _ in range(3)]  # action 키가 전혀 없는 경우
    stats = compute_train_stats(frames)
    assert ACTION not in stats
    assert OBS_STATE in stats


def test_apply_min_std_with_report_flags_floored_dims():
    stats = {
        ACTION: {"mean": torch.zeros(4), "std": torch.tensor([1e-6, 0.5, 1e-6, 2.0])},
        OBS_STATE: {"mean": torch.zeros(4), "std": torch.tensor([1e-6, 1e-6, 0.3, 0.3])},
    }
    new_stats, report = apply_min_std_with_report(
        stats, action_min_std=0.1, state_min_std=0.2, state_select=[0, 1, 2, 3],
        action_names=["a0", "a1", "a2", "a3"], state_names=["s0", "s1", "s2", "s3"],
    )
    floored_action_names = {d.name for d in report.action_floored}
    floored_state_names = {d.name for d in report.state_floored}
    assert floored_action_names == {"a0", "a2"}  # std가 1e-6인 두 채널만
    assert floored_state_names == {"s0", "s1"}
    assert new_stats[ACTION]["std"][1] == 0.5  # 이미 하한보다 큰 채널은 안 바뀜
    # 원본은 변형되지 않아야 한다(네스트 함수들이 이미 이 성질을 보장 — 여기선 사용자 관점에서 재확인).
    assert stats[ACTION]["std"][0].item() == pytest.approx(1e-6)


def test_apply_min_std_with_report_none_means_no_floor():
    stats = {ACTION: {"mean": torch.zeros(2), "std": torch.tensor([1e-6, 1e-6])}}
    new_stats, report = apply_min_std_with_report(stats, None, None, None)
    assert new_stats is stats
    assert report.action_floored == []
    assert report.state_floored == []


def test_min_std_report_summary_lines_are_human_readable():
    stats = {ACTION: {"mean": torch.zeros(2), "std": torch.tensor([1e-6, 0.5])}}
    _, report = apply_min_std_with_report(stats, action_min_std=0.05, state_min_std=None, state_select=None)
    lines = report.summary_lines()
    assert any("action_min_std" in line for line in lines)
    assert any("0.05" in line or "5e-02" in line for line in lines)
