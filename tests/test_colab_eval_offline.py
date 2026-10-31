"""scripts/colab/eval_offline.py — 차원별 L1 계산과 두 베이스라인("평균 action 상수
예측", "직전 action 유지")의 수치가 올바른지, 그리고 여러 평가 지점에 대한
평균/표준편차 집계가 맞는지 순수 수학으로 확인한다(실제 정책/데이터 없이 —
`run_offline_eval()`은 `build_tiny_policy`를 쓰는 통합 테스트
`tests/test_colab_smoke.py`에서 커버한다)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

from eval_offline import ACTION_DIM_GROUPS, EvalSample, compute_baselines, l1_by_group  # noqa: E402


def test_l1_by_group_matches_manual_computation():
    pred = torch.zeros(2, 14)
    target = torch.ones(2, 14) * 2.0  # 모든 채널에서 오차 2.0
    result = l1_by_group(pred, target)
    for name in ACTION_DIM_GROUPS:
        assert abs(result[name] - 2.0) < 1e-6
    assert abs(result["overall"] - 2.0) < 1e-6


def test_l1_by_group_distinguishes_groups():
    pred = torch.zeros(1, 14)
    target = torch.zeros(1, 14)
    target[0, 0:3] = 1.0  # left_pos만 오차
    result = l1_by_group(pred, target)
    assert abs(result["left_pos"] - 1.0) < 1e-6
    assert abs(result["left_rot"] - 0.0) < 1e-6
    assert abs(result["right_gripper"] - 0.0) < 1e-6


def test_compute_baselines_constant_mean_is_exact_when_target_equals_mean():
    train_mean = torch.full((14,), 3.0)
    target_chunk = torch.full((5, 14), 3.0)
    sample = EvalSample(batch={}, target_chunk=target_chunk, prev_action=torch.zeros(14))
    baselines = compute_baselines(train_mean, [sample])
    assert abs(baselines["constant_mean"]["overall"]["mean"]) < 1e-6
    assert abs(baselines["constant_mean"]["overall"]["std"]) < 1e-6  # 지점이 1개면 표준편차는 0


def test_compute_baselines_repeat_last_is_exact_when_action_is_constant():
    prev_action = torch.full((14,), 5.0)
    target_chunk = torch.full((6, 14), 5.0)  # 에피소드 내내 액션이 똑같이 유지되는 상황
    sample = EvalSample(batch={}, target_chunk=target_chunk, prev_action=prev_action)
    baselines = compute_baselines(torch.zeros(14), [sample])
    assert abs(baselines["repeat_last"]["overall"]["mean"]) < 1e-6


def test_compute_baselines_averages_across_samples():
    train_mean = torch.zeros(14)
    s1 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 1.0, prev_action=torch.zeros(14))
    s2 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 3.0, prev_action=torch.zeros(14))
    baselines = compute_baselines(train_mean, [s1, s2])
    assert abs(baselines["constant_mean"]["overall"]["mean"] - 2.0) < 1e-6  # (1+3)/2


def test_compute_baselines_std_reflects_spread_across_points():
    """지점별 L1이 서로 다르면(1.0과 3.0처럼), 표준편차가 0이 아니어야 하고 그 값이
    직접 계산한 모표준편차와 일치해야 한다."""
    train_mean = torch.zeros(14)
    s1 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 1.0, prev_action=torch.zeros(14))
    s2 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 3.0, prev_action=torch.zeros(14))
    baselines = compute_baselines(train_mean, [s1, s2])
    # 두 지점의 overall L1은 각각 1.0, 3.0 -> 모표준편차 = 1.0
    assert abs(baselines["constant_mean"]["overall"]["std"] - 1.0) < 1e-6


def test_compute_baselines_many_points_matches_torch_std():
    train_mean = torch.zeros(14)
    torch.manual_seed(0)
    samples = [
        EvalSample(batch={}, target_chunk=torch.rand(4, 14) * (i + 1), prev_action=torch.zeros(14))
        for i in range(8)
    ]
    baselines = compute_baselines(train_mean, samples)
    per_point = [l1_by_group(torch.zeros(4, 14), s.target_chunk)["overall"] for s in samples]
    expected_mean = sum(per_point) / len(per_point)
    expected_std = torch.tensor(per_point).std(unbiased=False).item()
    assert abs(baselines["constant_mean"]["overall"]["mean"] - expected_mean) < 1e-5
    assert abs(baselines["constant_mean"]["overall"]["std"] - expected_std) < 1e-5
