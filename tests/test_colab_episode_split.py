"""scripts/colab/episode_split.py — 학습/평가 분할의 재현성 + 저장/복원."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import episode_split  # noqa: E402

import pytest


def test_split_episodes_counts_and_no_overlap():
    split = episode_split.split_episodes(200, num_eval=10, seed=0)
    assert len(split.train_ids) == 190
    assert len(split.eval_ids) == 10
    assert set(split.train_ids) | set(split.eval_ids) == set(range(200))
    assert set(split.train_ids) & set(split.eval_ids) == set()


def test_split_episodes_is_reproducible_given_same_seed():
    a = episode_split.split_episodes(200, num_eval=10, seed=42)
    b = episode_split.split_episodes(200, num_eval=10, seed=42)
    assert a.train_ids == b.train_ids
    assert a.eval_ids == b.eval_ids


def test_split_episodes_different_seed_differs():
    a = episode_split.split_episodes(200, num_eval=10, seed=0)
    b = episode_split.split_episodes(200, num_eval=10, seed=1)
    assert a.eval_ids != b.eval_ids


def test_split_episodes_rejects_invalid_num_eval():
    with pytest.raises(ValueError):
        episode_split.split_episodes(10, num_eval=11, seed=0)
    with pytest.raises(ValueError):
        episode_split.split_episodes(10, num_eval=-1, seed=0)


def test_save_and_load_round_trip(tmp_path):
    split = episode_split.split_episodes(50, num_eval=5, seed=7)
    path = tmp_path / "split.json"
    episode_split.save_split(path, split)
    loaded = episode_split.load_split(path)
    assert loaded == split


def test_load_or_create_split_creates_once_then_reuses(tmp_path):
    path = tmp_path / "split.json"
    first = episode_split.load_or_create_split(path, num_episodes=200, num_eval=10, seed=0)
    assert path.exists()
    second = episode_split.load_or_create_split(path, num_episodes=200, num_eval=10, seed=999)
    # 파일이 이미 있으므로 seed=999는 무시되고 첫 호출의 분할을 그대로 재사용해야 한다.
    assert second.train_ids == first.train_ids
    assert second.eval_ids == first.eval_ids
    assert second.seed == first.seed == 0


def test_load_or_create_split_rejects_mismatched_num_episodes(tmp_path):
    path = tmp_path / "split.json"
    episode_split.load_or_create_split(path, num_episodes=200, num_eval=10, seed=0)
    with pytest.raises(ValueError):
        episode_split.load_or_create_split(path, num_episodes=150, num_eval=10, seed=0)
