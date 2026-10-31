"""재시작 가능한 배치 소스 — `torch.utils.data.DataLoader(shuffle=True)`는 "몇 번째
배치까지 썼는지"를 저장할 수 있는 상태를 노출하지 않으므로, 에포크 경계에서 **매
에포크를 `data_seed + epoch`로 명시적으로 재시드**해서 "에포크 번호만 알면 그 에포크의
배치 순서를 정확히 재구성할 수 있게" 만든다. 재시작 시에는 그 지점까지의 에포크를
다시 만들고, 이미 소비한 배치 수만큼 그 자리에서 버려서(fast-forward) 같은 위치로
돌아간다 — 데이터를 다시 "읽기"는 하지만 모델 forward/backward는 전혀 하지 않으므로
학습 결과에는 영향이 없다. 다만 데이터셋이 크거나 재시작 지점이 에포크 후반이면 그만큼
느리다(알려진 비용, `docs/colab_free_tier.md` 참고)."""

from __future__ import annotations

from typing import Callable, Iterator


class BatchSource:
    def __init__(self, make_iter_for_epoch: Callable[[int], Iterator[dict]], steps_per_epoch: int):
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch는 양수여야 합니다: {steps_per_epoch}")
        self.make_iter_for_epoch = make_iter_for_epoch
        self.steps_per_epoch = steps_per_epoch
        self._epoch = 0
        self._pos = 0
        self._iter = make_iter_for_epoch(0)

    def next_batch(self) -> dict:
        if self._pos >= self.steps_per_epoch:
            self._epoch += 1
            self._iter = self.make_iter_for_epoch(self._epoch)
            self._pos = 0
        batch = next(self._iter)
        self._pos += 1
        return batch

    def fast_forward(self, num_batches_already_consumed: int) -> None:
        """체크포인트가 기록한 `step`(=이미 소비한 배치 수)을 받아 그 다음 배치부터
        나오도록 되돌린다."""
        if num_batches_already_consumed < 0:
            raise ValueError(f"num_batches_already_consumed는 0 이상이어야 합니다: {num_batches_already_consumed}")
        self._epoch, self._pos = divmod(num_batches_already_consumed, self.steps_per_epoch)
        self._iter = self.make_iter_for_epoch(self._epoch)
        for _ in range(self._pos):
            next(self._iter)
