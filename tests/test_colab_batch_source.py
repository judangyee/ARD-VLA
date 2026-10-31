"""scripts/colab/batch_source.py — 재시작 시 "같은 지점"으로 정확히 되돌아가는지,
그리고 **`fast_forward`가 실제 로드를 전혀 일으키지 않는지**(인덱스 계산만)를
검증한다."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import pytest

from batch_source import BatchSource  # noqa: E402


def _counting_loader(load_calls: list[list[int]]):
    def load_batch(indices: list[int]) -> list[int]:
        load_calls.append(list(indices))
        return list(indices)

    return load_batch


def test_next_batch_covers_every_index_exactly_once_per_epoch():
    calls: list[list[int]] = []
    source = BatchSource(dataset_len=8, batch_size=2, data_seed=0, load_batch=_counting_loader(calls))
    batches = [source.next_batch() for _ in range(4)]  # 정확히 한 에포크
    covered = sorted(i for b in batches for i in b)
    assert covered == list(range(8))
    assert len(calls) == 4


def test_next_batch_reseeds_deterministically_per_epoch():
    """같은 data_seed로 새로 만든 소스를 똑같이 진행하면 항상 같은 배치가 나와야 한다."""
    calls_a: list[list[int]] = []
    source_a = BatchSource(dataset_len=8, batch_size=2, data_seed=7, load_batch=_counting_loader(calls_a))
    seq_a = [source_a.next_batch() for _ in range(10)]

    calls_b: list[list[int]] = []
    source_b = BatchSource(dataset_len=8, batch_size=2, data_seed=7, load_batch=_counting_loader(calls_b))
    seq_b = [source_b.next_batch() for _ in range(10)]

    assert seq_a == seq_b


def test_different_data_seed_gives_different_order():
    a = BatchSource(dataset_len=20, batch_size=2, data_seed=1, load_batch=_counting_loader([]))
    b = BatchSource(dataset_len=20, batch_size=2, data_seed=2, load_batch=_counting_loader([]))
    seq_a = [a.next_batch() for _ in range(5)]
    seq_b = [b.next_batch() for _ in range(5)]
    assert seq_a != seq_b


def test_fast_forward_does_not_call_load_batch_at_all():
    """핵심 요구사항: 재개가 "읽고 버리기"가 아니라 순수 인덱스 계산이어야 한다."""
    calls: list[list[int]] = []
    source = BatchSource(dataset_len=20, batch_size=2, data_seed=3, load_batch=_counting_loader(calls))
    for _ in range(6):
        source.next_batch()
    assert len(calls) == 6

    calls.clear()
    source.fast_forward(6)
    assert calls == []  # fast_forward 자체는 load_batch를 단 한 번도 안 불러야 한다


def test_fast_forward_resumes_at_exact_next_batch():
    """끊김 없이 쭉 돌린 결과와, 중간에 재시작(fast_forward)한 결과가 그 지점부터
    정확히 같은 배치를 내야 한다."""
    uninterrupted = BatchSource(dataset_len=20, batch_size=2, data_seed=42, load_batch=_counting_loader([]))
    full_sequence = [uninterrupted.next_batch() for _ in range(17)]

    split_point = 8
    first_half_source = BatchSource(dataset_len=20, batch_size=2, data_seed=42, load_batch=_counting_loader([]))
    first_half = [first_half_source.next_batch() for _ in range(split_point)]

    resumed_source = BatchSource(dataset_len=20, batch_size=2, data_seed=42, load_batch=_counting_loader([]))
    resumed_source.fast_forward(split_point)
    second_half = [resumed_source.next_batch() for _ in range(17 - split_point)]

    assert first_half == full_sequence[:split_point]
    assert first_half + second_half == full_sequence


def test_fast_forward_across_epoch_boundary():
    """재시작 지점이 에포크 경계를 넘는 경우(2번째 에포크 중간)도 정확해야 한다."""
    steps_per_epoch = 4  # dataset_len=8, batch_size=2
    uninterrupted = BatchSource(dataset_len=8, batch_size=2, data_seed=5, load_batch=_counting_loader([]))
    full_sequence = [uninterrupted.next_batch() for _ in range(steps_per_epoch + 2)]  # 2번째 에포크 2번째 배치까지

    resumed = BatchSource(dataset_len=8, batch_size=2, data_seed=5, load_batch=_counting_loader([]))
    resumed.fast_forward(steps_per_epoch + 1)  # 그 다음(마지막) 배치부터 재개
    assert resumed.next_batch() == full_sequence[-1]


def test_fast_forward_to_zero_restarts_from_beginning():
    source = BatchSource(dataset_len=8, batch_size=2, data_seed=9, load_batch=_counting_loader([]))
    first_ever = BatchSource(dataset_len=8, batch_size=2, data_seed=9, load_batch=_counting_loader([])).next_batch()
    source.next_batch()
    source.next_batch()
    source.fast_forward(0)
    assert source.next_batch() == first_ever


def test_fast_forward_rejects_negative():
    source = BatchSource(dataset_len=8, batch_size=2, data_seed=0, load_batch=_counting_loader([]))
    with pytest.raises(ValueError):
        source.fast_forward(-1)


def test_rejects_batch_size_larger_than_dataset():
    with pytest.raises(ValueError):
        BatchSource(dataset_len=3, batch_size=4, data_seed=0, load_batch=_counting_loader([]))


def test_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        BatchSource(dataset_len=8, batch_size=0, data_seed=0, load_batch=_counting_loader([]))
