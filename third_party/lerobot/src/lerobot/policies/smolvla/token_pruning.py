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
"""EfficientVLA(Yang et al., 2025)식 Task-Relevance and Diversity-Driven Visual Token Pruning.

SmolVLA는 프레임 하나를 SigLIP 비전 인코더 + pixel shuffle(`transformers`의
`SmolVLMConnector.pixel_shuffle`, `scale_factor=2`로 패치 수를 1/4로 줄임) +
modality projection을 거쳐 고정된 개수의 비전 토큰으로 만든다
(`SmolVLMWithExpertModel.embed_image()`, `smolvlm_with_expert.py` 참고). 이 비전 토큰은
전부 언어 지시문 토큰과 나란히 VLM에 들어가는데, 실제로 태스크와 관련 있는 건 그중 일부뿐이다.

이 모듈은 그 비전 토큰 집합을 다음 순서로 줄인다:
  1. 언어 지시문 토큰을 쿼리, 비전 토큰을 키로 하는 (파라미터 없는) scaled dot-product
     cross-attention을 계산해서, 각 비전 토큰이 언어 쿼리들로부터 평균적으로 얼마나
     주의를 받는지를 "태스크 관련성 점수"로 쓴다 (`compute_task_relevance_scores`).
  2. 관련성 상위 `k_key`개(논문 기준 4~8개)는 무조건 남긴다 — "핵심 토큰 세트".
  3. 남은 예산의 절반은 관련성 순위대로, 절반은 이미 뽑힌 토큰들과 코사인 거리가 가장 먼
     (=가장 다른) 토큰을 그리디하게 골라서 채운다 — 다양성 확보 (`select_tokens`).

주의 — 이건 EfficientVLA 논문을 직접 읽고 옮긴 게 아니라, 사용자가 요청에서 설명한 알고리즘
(관련성 top-K_key 고정 + 절반 관련성/절반 다양성 채우기)을 그대로 구현한 것이다. SmolVLA에는
비전 토큰과 언어 토큰 사이에 사전학습된 cross-attention이 원래 없어서(둘 다 텍스트 임베딩과
같은 `text_config.hidden_size` 차원 공간에 있긴 하지만), 여기서는 추가 학습 파라미터 없이
두 임베딩의 내적만으로 관련성을 매기는 training-free 방식을 썼다 — 이는 FastV/VisionZip 등
다른 비전 토큰 프루닝 연구에서도 흔한 접근이지만, 사전학습된 attention이 아니므로 임베딩이
실제로 얼마나 의미 있는 관련성 신호를 담고 있는지는 사전학습된 SmolVLM2 가중치로 실험해봐야
확인할 수 있다 (이 레포 샌드박스는 그 가중치를 받을 수 없다 — README 참고).
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_task_relevance_scores(
    image_tokens: Tensor, language_tokens: Tensor, language_mask: Tensor | None = None
) -> Tensor:
    """언어 토큰을 쿼리로, 비전 토큰을 키로 하는 scaled dot-product cross-attention을 계산해서
    각 비전 토큰의 태스크 관련성 점수를 매긴다.

    image_tokens: (batch, num_img_tokens, hidden)
    language_tokens: (batch, num_lang_tokens, hidden) — image_tokens와 같은 hidden 차원 공간
        (SmolVLA에서는 둘 다 modality_projection/embedding lookup을 거쳐 text_config.hidden_size).
    language_mask: (batch, num_lang_tokens) bool, True=유효 토큰(패딩 아님). 없으면 전부 유효로 본다.

    반환: (batch, num_img_tokens) — 각 비전 토큰이 (유효한) 언어 쿼리들로부터 평균적으로 받는
        attention 가중치. 값이 클수록 태스크(언어 지시문)와 관련 있는 토큰이라는 뜻이다.
    """
    if image_tokens.ndim != 3 or language_tokens.ndim != 3:
        raise ValueError(
            f"image_tokens/language_tokens는 (batch, seq, hidden) 3차원이어야 합니다. "
            f"받은 shape: {tuple(image_tokens.shape)}, {tuple(language_tokens.shape)}"
        )
    hidden = image_tokens.shape[-1]
    # (batch, num_lang, num_img): 언어 토큰 하나하나가 비전 토큰 전체에 갖는 raw 유사도.
    scores = torch.einsum("bld,bnd->bln", language_tokens.float(), image_tokens.float()) / (hidden**0.5)

    # 여기서는 language_mask로 행 전체를 -inf 마스킹하지 않는다 — 패딩 쿼리 행이 전부 -inf가
    # 되면 그 행의 softmax가 0/0 = NaN이 되어버린다(분배할 곳이 없으므로). 어차피 무효한 언어
    # 위치의 기여는 아래에서 `valid` 마스크를 곱해 평균에서 제외하므로, 여기서는 그냥 정상적으로
    # softmax를 계산한다 (무효 행의 값 자체는 버려지니 의미가 없어도 상관없다).
    attn = F.softmax(scores, dim=-1)  # (batch, num_lang, num_img), 비전 토큰(N) 축으로 정규화

    if language_mask is not None:
        valid = language_mask.float().unsqueeze(-1)  # (batch, num_lang, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)  # (batch, 1)
        relevance = (attn * valid).sum(dim=1) / denom  # 유효 언어 토큰에 대해서만 평균
    else:
        relevance = attn.mean(dim=1)

    return relevance.to(image_tokens.dtype)  # (batch, num_img_tokens)


def select_tokens(
    image_tokens: Tensor,
    relevance_scores: Tensor,
    k_final: int,
    k_key: int = 6,
) -> tuple[Tensor, Tensor]:
    """관련성 상위 `k_key`개(핵심 세트) + 나머지 예산의 절반은 관련성 순, 절반은 다양성
    (기존 선택 세트와 코사인 거리가 최대인 토큰을 그리디하게 선택)으로 채워 `k_final`개를 고른다.

    image_tokens: (batch, N, hidden). relevance_scores: (batch, N) — compute_task_relevance_scores 참고.
    k_key: 4~8 권장(논문 기준) — 이보다 벗어나도 에러는 아니고 경고만 남긴다(실험 유연성을 위해).

    반환: (selected_tokens (batch, k_final, hidden), selected_indices (batch, k_final) long).
    선택된 인덱스는 각 샘플 안에서 원본 위치 기준 오름차순으로 정렬된다 — 위치 인코딩(RoPE)이
    선택 순서가 아니라 시퀀스 내 위치에 의존하므로, 원래의 공간적 순서를 보존하기 위함이다.
    """
    if not (4 <= k_key <= 8):
        logging.warning("k_key=%d는 논문이 권장하는 4~8 범위를 벗어났습니다 (그대로 진행합니다).", k_key)

    batch, n_tokens, hidden = image_tokens.shape
    device = image_tokens.device

    if k_final >= n_tokens:
        if k_final > n_tokens:
            logging.warning(
                "k_final=%d이 비전 토큰 개수(%d)보다 커서 전부 남깁니다.", k_final, n_tokens
            )
        indices = torch.arange(n_tokens, device=device).unsqueeze(0).expand(batch, -1)
        return image_tokens, indices

    k_key = max(min(k_key, k_final), 1)
    order = torch.argsort(relevance_scores, dim=-1, descending=True)  # (batch, N), 관련성 내림차순

    remaining_budget = k_final - k_key
    n_relevance_fill = remaining_budget // 2
    n_diversity_fill = remaining_budget - n_relevance_fill

    selected_per_sample = []
    for b in range(batch):
        order_b = order[b].tolist()
        core = order_b[:k_key]
        pool = order_b[k_key:]

        relevance_fill = pool[:n_relevance_fill]
        diversity_pool = pool[n_relevance_fill:]

        chosen = list(core) + list(relevance_fill)
        diversity_pool_idx = torch.tensor(diversity_pool, device=device, dtype=torch.long)

        for _ in range(n_diversity_fill):
            if diversity_pool_idx.numel() == 0:
                break
            chosen_tokens = F.normalize(image_tokens[b, chosen].float(), dim=-1)  # (len(chosen), hidden)
            pool_tokens = F.normalize(image_tokens[b, diversity_pool_idx].float(), dim=-1)  # (P, hidden)
            cos_sim_to_chosen = pool_tokens @ chosen_tokens.T  # (P, len(chosen))
            # "가장 다르다" = 이미 뽑힌 것들 중 제일 가까운(제일 비슷한) 것까지의 코사인 거리가 최대.
            dist_to_nearest_chosen = 1.0 - cos_sim_to_chosen.max(dim=-1).values  # (P,)
            best_local = int(torch.argmax(dist_to_nearest_chosen).item())
            chosen.append(int(diversity_pool_idx[best_local].item()))
            keep = torch.ones_like(diversity_pool_idx, dtype=torch.bool)
            keep[best_local] = False
            diversity_pool_idx = diversity_pool_idx[keep]

        # 다양성 후보가 모자라 목표 개수(k_final)를 못 채웠으면, 관련성 순으로 마저 채운다.
        if len(chosen) < k_final:
            leftover = [i for i in order_b if i not in chosen]
            chosen.extend(leftover[: k_final - len(chosen)])

        selected_per_sample.append(sorted(chosen[:k_final]))

    selected_indices = torch.tensor(selected_per_sample, device=device, dtype=torch.long)  # (batch, k_final)
    selected_tokens = torch.gather(image_tokens, 1, selected_indices.unsqueeze(-1).expand(-1, -1, hidden))
    return selected_tokens, selected_indices
