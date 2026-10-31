"""scripts/colab/batch_source.py — 재시작 시 "같은 지점"으로 정확히 되돌아가는지
(fast_forward의 결정론성)를 검증한다. 체크포인트 재시작의 "데이터 순서 상태" 부분을
모델/데이터셋 없이 순수하게 테스트한다."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import pytest

from batch_source import BatchSource  # noqa: E402


def _make_source(seed_offset_tracker: list[int], steps_per_epoch: int = 4) -> BatchSource:
    def make_iter_for_epoch(epoch: int):
        seed_offset_tracker.append(epoch)
        return iter([f"e{epoch}_b{i}" for i in range(steps_per_epoch)])

    return BatchSource(make_iter_for_epoch, steps_per_epoch=steps_per_epoch)


def test_next_batch_runs_through_epoch_in_order():
    calls: list[int] = []
    source = _make_source(calls, steps_per_epoch=3)
    seq = [source.next_batch() for _ in range(7)]
    assert seq == ["e0_b0", "e0_b1", "e0_b2", "e1_b0", "e1_b1", "e1_b2", "e2_b0"]
    assert calls == [0, 1, 2]  # 생성자에서 epoch 0 한 번, 그 뒤 소진될 때마다 다음 epoch


def test_fast_forward_resumes_at_exact_next_batch():
    calls_a: list[int] = []
    source_a = _make_source(calls_a, steps_per_epoch=4)
    consumed = [source_a.next_batch() for _ in range(11)]  # 10개를 "이미 학습에 썼다"고 가정 + 비교용 11번째

    calls_b: list[int] = []
    source_b = _make_source(calls_b, steps_per_epoch=4)
    source_b.fast_forward(10)
    resumed_next = source_b.next_batch()

    # source_a의 11번째(0-indexed 10번째) 배치와 정확히 같아야 한다.
    direct = _make_source([], steps_per_epoch=4)
    for _ in range(11):
        value = direct.next_batch()
    assert resumed_next == value
    assert consumed[10] == value  # source_a를 끊지 않고 쭉 돌렸어도 같은 값이어야 함


def test_fast_forward_then_continue_matches_uninterrupted_run():
    """재시작 지점부터 쭉 이어간 결과가, 끊기지 않고 처음부터 쭉 돌린 결과와 동일한지."""
    steps_per_epoch = 5
    total = 17

    uninterrupted = _make_source([], steps_per_epoch=steps_per_epoch)
    full_sequence = [uninterrupted.next_batch() for _ in range(total)]

    split_point = 8
    first_half_source = _make_source([], steps_per_epoch=steps_per_epoch)
    first_half = [first_half_source.next_batch() for _ in range(split_point)]

    resumed_source = _make_source([], steps_per_epoch=steps_per_epoch)
    resumed_source.fast_forward(split_point)
    second_half = [resumed_source.next_batch() for _ in range(total - split_point)]

    assert first_half + second_half == full_sequence


def test_fast_forward_to_zero_restarts_from_beginning():
    source = _make_source([], steps_per_epoch=3)
    source.next_batch()
    source.next_batch()
    source.fast_forward(0)
    assert source.next_batch() == "e0_b0"


def test_fast_forward_rejects_negative():
    source = _make_source([], steps_per_epoch=3)
    with pytest.raises(ValueError):
        source.fast_forward(-1)


def test_steps_per_epoch_must_be_positive():
    with pytest.raises(ValueError):
        BatchSource(lambda epoch: iter([]), steps_per_epoch=0)
