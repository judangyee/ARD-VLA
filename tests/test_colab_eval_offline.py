"""scripts/colab/eval_offline.py — 차원별 L1 계산과 두 베이스라인("평균 action 상수
예측", "직전 action 유지")의 수치가 올바른지 순수 수학으로 확인한다(실제 정책/데이터
없이 — `run_offline_eval()`은 `build_tiny_policy`를 쓰는 통합 테스트
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
    assert abs(baselines["constant_mean"]["overall"]) < 1e-6


def test_compute_baselines_repeat_last_is_exact_when_action_is_constant():
    prev_action = torch.full((14,), 5.0)
    target_chunk = torch.full((6, 14), 5.0)  # 에피소드 내내 액션이 똑같이 유지되는 상황
    sample = EvalSample(batch={}, target_chunk=target_chunk, prev_action=prev_action)
    baselines = compute_baselines(torch.zeros(14), [sample])
    assert abs(baselines["repeat_last"]["overall"]) < 1e-6


def test_compute_baselines_averages_across_samples():
    train_mean = torch.zeros(14)
    s1 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 1.0, prev_action=torch.zeros(14))
    s2 = EvalSample(batch={}, target_chunk=torch.ones(3, 14) * 3.0, prev_action=torch.zeros(14))
    baselines = compute_baselines(train_mean, [s1, s2])
    assert abs(baselines["constant_mean"]["overall"] - 2.0) < 1e-6  # (1+3)/2
