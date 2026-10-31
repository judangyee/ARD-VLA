"""재시작 가능한 배치 소스 — **인덱스만으로 재개한다(영상 디코딩 없이)**.

`torch.utils.data.DataLoader(shuffle=True)`는 "몇 번째 배치까지 썼는지"를 저장할 수
있는 상태를 노출하지 않는다. 이전 버전은 재시작 시 `make_iter_for_epoch(epoch)`로
실제 배치 이터레이터를 다시 만들고 그 자리에서 `next()`를 호출해 "읽고 버리는"
방식으로 같은 위치까지 되돌렸는데, 이러면 영상 디코딩까지 전부 다시 일어나서
재시작 지점이 늦을수록 느려졌다(이전 설계의 알려진 한계).

이번 버전은 `DataLoader(shuffle=True)`가 내부적으로 쓰는 `RandomSampler`와 똑같은
방식(`torch.randperm(n, generator=seeded_generator)`)으로 **인덱스 순열만** 매
에포크 `data_seed + epoch`로 재계산한다 — 이건 순수 RNG 연산이라 데이터셋/영상에
전혀 접근하지 않는다. 재시작(`fast_forward`)은 그 순열에서 "이미 소비한 개수만큼
건너뛴 시작 위치"만 계산하고 끝난다 — **실제로 배치를 읽는(`load_batch` 호출) 일은
전혀 일어나지 않는다.** 그 다음 `next_batch()`를 호출하는 시점에야 그 배치 하나만
실제로 로드한다."""

from __future__ import annotations

from typing import Callable

import torch


class BatchSource:
    def __init__(self, dataset_len: int, batch_size: int, data_seed: int, load_batch: Callable[[list[int]], dict]):
        if batch_size <= 0:
            raise ValueError(f"batch_size는 양수여야 합니다: {batch_size}")
        steps_per_epoch = dataset_len // batch_size  # DataLoader(drop_last=True)와 동일한 규칙
        if steps_per_epoch <= 0:
            raise ValueError(f"dataset_len({dataset_len})이 batch_size({batch_size})보다 작습니다.")

        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.load_batch = load_batch
        self.steps_per_epoch = steps_per_epoch
        self._epoch = 0
        self._pos = 0
        self._permutation = self._make_permutation(0)

    def _make_permutation(self, epoch: int) -> torch.Tensor:
        """`torch.utils.data.RandomSampler`가 `shuffle=True`일 때 쓰는 것과 정확히 같은
        방식(`torch.randperm(n, generator=...)`)으로, `data_seed + epoch`에서만 결정되는
        순열을 돌려준다 — 순수 RNG 연산, 데이터셋 접근 없음."""
        gen = torch.Generator()
        gen.manual_seed(self.data_seed + epoch)
        return torch.randperm(self.dataset_len, generator=gen)

    def _indices_for_pos(self, pos: int) -> list[int]:
        start = pos * self.batch_size
        return self._permutation[start : start + self.batch_size].tolist()

    def next_batch(self) -> dict:
        if self._pos >= self.steps_per_epoch:
            self._epoch += 1
            self._permutation = self._make_permutation(self._epoch)
            self._pos = 0
        indices = self._indices_for_pos(self._pos)
        self._pos += 1
        return self.load_batch(indices)

    def fast_forward(self, num_batches_already_consumed: int) -> None:
        """체크포인트가 기록한 `step`(=이미 소비한 배치 수)을 받아 그 다음 배치부터
        나오도록 되돌린다 — `load_batch`를 단 한 번도 호출하지 않는다(순열 재계산 +
        위치 계산뿐)."""
        if num_batches_already_consumed < 0:
            raise ValueError(f"num_batches_already_consumed는 0 이상이어야 합니다: {num_batches_already_consumed}")
        self._epoch, self._pos = divmod(num_batches_already_consumed, self.steps_per_epoch)
        self._permutation = self._make_permutation(self._epoch)
