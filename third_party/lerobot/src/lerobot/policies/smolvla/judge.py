# Copyright 2026 The ARD-VLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ARD-VLA 폐루프 구조의 Judge/Selector head.

SmolVLM2 backbone의 prefix(이미지+언어+state) hidden state 하나를 masked mean pooling한
벡터 위에, 서로 완전히 분리된(trunk를 공유하지 않는) 작은 MLP head들을 얹는다:
`success_logit`(1), `failure_logits`(K, 선택), `progress`(0~1, 선택), `selector_logits`
(N, 선택). pooling 다음에 두는 `LayerNorm`만 모든 head가 공유하는 전처리인데, 이건
`judge_feature_layer=-1`(prefix 마지막 레이어, post-norm)과 중간 레이어(pre-norm, Bridge
Attention과 같은 `collect_layer_indices` 경로로 뽑음)의 스케일 불일치를 흡수하기 위한
것이다(PROGRESS.md의 폐루프 1단계 추적 기록 참고) — "head 간에 공유하는 feature extractor
trunk"가 아니다.

이 모듈 자체는 `predict_action_chunk_with_decisions`/학습 forward가 prefix hidden state를
어디서 뽑아오는지(마지막 레이어 vs `last_collected_prefix_layers`)는 몰라도 된다 — 호출부가
`(batch, prefix_seq_len, vlm_hidden_size)` 텐서와 `prefix_pad_masks`만 넘기면 된다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Judge(nn.Module):
    """Judge(성공 판정)/Selector(다음 서브태스크 선택) head 묶음.

    `use_judge`/`use_selector`로 어느 head들을 실제로 만들지 독립적으로 결정한다 —
    `judge_num_failure_types`/`judge_use_progress`는 `use_judge=True`일 때만 의미가 있고,
    `selector_num_options`는 `use_selector=True`일 때만 의미가 있다(상위 설정 검증은
    `configuration_smolvla.py`에서 한다, 여기서는 단순히 `None`인 head는 안 만든다).
    """

    def __init__(
        self,
        vlm_hidden_size: int,
        hidden_dim: int = 256,
        use_judge: bool = True,
        num_failure_types: int = 0,
        use_progress: bool = False,
        use_selector: bool = False,
        selector_num_options: int = 0,
        detach_features: bool = True,
    ):
        super().__init__()
        self.detach_features = detach_features
        self.pool_norm = nn.LayerNorm(vlm_hidden_size)

        def _mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(vlm_hidden_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.success_head = _mlp(1) if use_judge else None
        self.failure_head = _mlp(num_failure_types) if use_judge and num_failure_types > 0 else None
        self.progress_head = _mlp(1) if use_judge and use_progress else None
        self.selector_head = _mlp(selector_num_options) if use_selector and selector_num_options > 0 else None

        # 캘리브레이션용 온도 — 학습되지 않는 버퍼(nn.Parameter가 아님), 기본 1.0이면
        # sigmoid(logit / 1.0) = sigmoid(logit)로 기존과 동일. 실제 값을 맞추는 건(온도 스케일링
        # fitting) 이번 작업 범위 밖 — 나중에 외부에서 `judge.temperature.fill_(T)`로 채울 슬롯만
        # 만들어둔다.
        self.register_buffer("temperature", torch.tensor(1.0))

    def pool(self, prefix_hidden: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        """`prefix_pad_masks`(batch, seq) 기준 masked mean pooling + LayerNorm."""
        mask = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_hidden.dtype)
        summed = (prefix_hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / count
        return self.pool_norm(pooled)

    def forward(
        self,
        prefix_hidden: Tensor,
        prefix_pad_masks: Tensor,
        allowed_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """allowed_mask: bool, shape (N,) 또는 (batch, N) — True인 옵션만 허용(나머지는
        -inf로 마스킹돼 softmax 확률이 0이 된다). `use_selector=False`면 무시된다."""
        pooled = self.pool(prefix_hidden, prefix_pad_masks)
        if self.detach_features:
            # Judge/Selector 손실이 백본/action expert로 역전파되지 않게 한다
            # (judge_detach_features=True, 1차 연구 범위에서는 백본을 고정한다는 전제).
            pooled = pooled.detach()

        out: dict[str, Tensor] = {}
        if self.success_head is not None:
            success_logit = self.success_head(pooled).squeeze(-1)
            out["success_logit"] = success_logit
            out["success_prob"] = torch.sigmoid(success_logit / self.temperature)
        if self.failure_head is not None:
            failure_logits = self.failure_head(pooled)
            out["failure_logits"] = failure_logits
            out["failure_probs"] = torch.softmax(failure_logits, dim=-1)
        if self.progress_head is not None:
            out["progress"] = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
        if self.selector_head is not None:
            selector_logits = self.selector_head(pooled)
            if allowed_mask is not None:
                selector_logits = selector_logits.masked_fill(~allowed_mask, float("-inf"))
            out["selector_logits"] = selector_logits
            out["selector_probs"] = torch.softmax(selector_logits, dim=-1)
        return out
