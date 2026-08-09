#!/usr/bin/env python3
"""EfficientVLA(Yang et al., 2025)식 비전 토큰 프루닝(`lerobot.policies.smolvla.token_pruning`)에
대한 오프라인 단위 테스트. GPU/Hub 접근 없이 합성 텐서로 검증한다 (test_ard.py와 같은 방식).
"""

import sys

import torch

from lerobot.policies.smolvla.token_pruning import compute_task_relevance_scores, select_tokens

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_relevance_scores():
    torch.manual_seed(0)
    batch, n_img, n_lang, hidden = 2, 64, 10, 16
    image_tokens = torch.randn(batch, n_img, hidden)
    language_tokens = torch.randn(batch, n_lang, hidden)
    language_mask = torch.ones(batch, n_lang, dtype=torch.bool)
    language_mask[:, 7:] = False  # 뒤 3개는 패딩

    scores = compute_task_relevance_scores(image_tokens, language_tokens, language_mask)
    check("relevance scores shape이 (batch, n_img)다", scores.shape == (batch, n_img))
    check("relevance scores에 NaN/Inf가 없다 (전부 -inf 마스킹된 행이 있어도)", torch.isfinite(scores).all().item())
    check(
        "샘플별 relevance 합이 1.0 근처다 (평균 attention 분포의 합=1)",
        abs(scores[0].sum().item() - 1.0) < 1e-4,
        detail=f"sum={scores[0].sum().item()}",
    )

    scores_no_mask = compute_task_relevance_scores(image_tokens, language_tokens, None)
    check("language_mask가 실제로 결과에 영향을 준다", not torch.allclose(scores, scores_no_mask))


def test_select_tokens_core_set_always_kept():
    n_tokens, hidden = 20, 16
    image_tokens = torch.randn(1, n_tokens, hidden)
    relevance = torch.zeros(1, n_tokens)
    relevance[0, :6] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    relevance[0, 6:] = torch.rand(n_tokens - 6)

    k_final, k_key = 10, 6
    sel_tokens, sel_idx = select_tokens(image_tokens, relevance, k_final=k_final, k_key=k_key)
    check("select_tokens 출력 shape이 (batch, k_final, hidden)이다", sel_tokens.shape == (1, k_final, hidden))
    check("selected_indices shape이 (batch, k_final)이다", sel_idx.shape == (1, k_final))

    idx_set = set(sel_idx[0].tolist())
    check("관련성 상위 k_key개(핵심 세트)가 항상 포함된다", set(range(6)).issubset(idx_set), detail=f"selected={sorted(idx_set)}")
    check("선택된 인덱스가 원본 위치 기준 오름차순 정렬된다", sel_idx[0].tolist() == sorted(sel_idx[0].tolist()))

    sel_tokens_all, sel_idx_all = select_tokens(image_tokens, relevance, k_final=n_tokens, k_key=6)
    check("k_final >= N이면 전부 그대로 반환한다", torch.equal(sel_idx_all[0], torch.arange(n_tokens)))

    sel_tokens_edge, _ = select_tokens(image_tokens, relevance, k_final=3, k_key=6)
    check("k_final < k_key 엣지케이스도 에러 없이 동작한다 (k_key가 k_final로 클램프됨)", sel_tokens_edge.shape == (1, 3, hidden))


def test_diversity_fill_picks_farthest_token():
    hidden = 4
    image_tokens = torch.zeros(1, 8, hidden)
    image_tokens[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    image_tokens[0, 1] = torch.tensor([0.99, 0.01, 0.0, 0.0])  # 토큰0과 매우 비슷
    image_tokens[0, 2] = torch.tensor([-1.0, 0.0, 0.0, 0.0])  # 토큰0과 정반대 (제일 다름)
    for i in range(3, 8):
        image_tokens[0, i] = torch.tensor([0.5, 0.5, 0.0, 0.0])

    relevance = torch.tensor([[10.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    # k_key=1(핵심=토큰0), k_final=2 -> 남은 예산 1개가 전부 다양성 채우기로 간다.
    _, sel_idx = select_tokens(image_tokens, relevance, k_final=2, k_key=1)
    check(
        "다양성 채우기가 코사인 거리가 가장 먼 토큰을 고른다 (비슷한 토큰1이 아니라 정반대인 토큰2)",
        set(sel_idx[0].tolist()) == {0, 2},
        detail=f"selected={sorted(sel_idx[0].tolist())}",
    )


def test_gradient_flow():
    torch.manual_seed(0)
    batch, n_img, hidden = 2, 64, 16
    image_tokens = torch.randn(batch, n_img, hidden, requires_grad=True)
    relevance = compute_task_relevance_scores(image_tokens, torch.randn(batch, 10, hidden), None)
    sel_tokens, _ = select_tokens(image_tokens, relevance, k_final=32, k_key=6)
    sel_tokens.sum().backward()

    n_with_grad = (image_tokens.grad.abs().sum(dim=-1) > 0).sum().item()
    check(
        "선택된 토큰 수만큼만 (batch*32개) gradient가 흐른다 — gather 기반이라 나머지는 정확히 0",
        n_with_grad == batch * 32,
        detail=f"n_with_grad={n_with_grad}, 기대값={batch * 32}",
    )


def main():
    test_relevance_scores()
    test_select_tokens_core_set_always_kept()
    test_diversity_fill_picks_farthest_token()
    test_gradient_flow()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)}개 항목 실패: {FAILURES}")
        sys.exit(1)
    print("[OK] 비전 토큰 프루닝 단위 테스트 전체 통과.")
    print(
        "참고: SmolVLAPolicy 전체(embed_prefix 안에서 실제로 호출되는 경로)를 생성해서 테스트하지는 "
        "않았습니다 — Hugging Face Hub에서 SmolVLM2 백본 config를 내려받아야 하는데 이 환경은 Hub "
        "접근이 막혀 있습니다. 대신 token_pruning.py 자체를 합성 텐서로 직접 검증했습니다."
    )


if __name__ == "__main__":
    main()
