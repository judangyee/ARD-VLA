#!/usr/bin/env python3
"""FreqPolicy(2025)식 주파수 일관성 손실(lerobot.policies.smolvla.freq_policy)에 대한
오프라인 단위 테스트. GPU/Hub 접근 없이 순수 텐서 연산만으로 전부 검증 가능하다.
"""

import math
import sys

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.freq_policy import (
    build_dct_matrix,
    compute_frequency_band_errors,
    compute_frequency_band_weights,
    compute_frequency_consistency_loss,
    get_dct_matrix,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_dct_matrix_orthonormal():
    torch.manual_seed(0)
    for n in (2, 8, 50, 97):  # 97은 홀수/소수 케이스
        D = build_dct_matrix(n, torch.device("cpu"), torch.float64)
        identity_err = (D @ D.T - torch.eye(n, dtype=torch.float64)).abs().max().item()
        check(f"DCT 행렬(n={n})이 직교정규다 (D @ D.T == I)", identity_err < 1e-8, detail=f"err={identity_err}")

        x = torch.randn(n, dtype=torch.float64)
        x_rec = D.T @ (D @ x)
        rec_err = (x_rec - x).abs().max().item()
        check(f"DCT 역변환(n={n})이 원신호를 정확히 복원한다", rec_err < 1e-8, detail=f"err={rec_err}")


def test_dct_matrix_cache():
    m1 = get_dct_matrix(20, torch.device("cpu"), torch.float32)
    m2 = get_dct_matrix(20, torch.device("cpu"), torch.float32)
    check("get_dct_matrix가 같은 (n, device, dtype)에 대해 캐싱된 동일 텐서를 반환한다", m1 is m2)


def test_band_weights():
    for n in (10, 50, 100):
        w0 = compute_frequency_band_weights(n, decay=0.0, device=torch.device("cpu"), dtype=torch.float64)
        check(f"decay=0(n={n})이면 모든 주파수 bin 가중치가 1이다", torch.allclose(w0, torch.ones(n, dtype=torch.float64)))

        w2 = compute_frequency_band_weights(n, decay=2.0, device=torch.device("cpu"), dtype=torch.float64)
        check(f"decay=2.0(n={n})이면 저주파(k=0) 가중치가 고주파(k=n-1)보다 크다", (w2[0] > w2[-1]).item())
        ratio = (w2[-1] / w2[0]).item()
        check(
            f"decay=2.0(n={n})의 최고/최저 주파수 가중치 비율이 chunk_size와 무관하게 exp(-2)와 같다",
            abs(ratio - math.exp(-2)) < 1e-6,
            detail=f"ratio={ratio}",
        )


def test_freq_loss_parseval_equivalence():
    torch.manual_seed(0)
    batch, chunk, dim = 4, 50, 14
    pred = torch.randn(batch, chunk, dim, dtype=torch.float64)
    target = torch.randn(batch, chunk, dim, dtype=torch.float64)

    freq_loss = compute_frequency_consistency_loss(pred, target, decay=0.0)
    time_mse = (pred - target).pow(2).mean()
    check(
        "decay=0이면 주파수 일관성 손실이 시간 영역 MSE와 수학적으로 동일하다 (Parseval 정리)",
        abs(freq_loss.item() - time_mse.item()) < 1e-8,
        detail=f"freq={freq_loss.item()} time={time_mse.item()}",
    )


def test_freq_loss_zero_when_equal():
    pred = torch.randn(2, 20, 8, dtype=torch.float64)
    loss = compute_frequency_consistency_loss(pred, pred.clone(), decay=1.5)
    check("pred == target이면 주파수 일관성 손실이 정확히 0이다", loss.item() == 0.0)


def test_freq_loss_prioritizes_low_frequency():
    """decay > 0이면, 저주파 성분에 생긴 오차가 고주파 성분의 같은 크기 오차보다 손실을 더 키워야 한다."""
    torch.manual_seed(0)
    batch, chunk, dim = 2, 32, 4
    target = torch.randn(batch, chunk, dim, dtype=torch.float64)

    D = build_dct_matrix(chunk, torch.device("cpu"), torch.float64)
    target_freq = torch.einsum("kt,btd->bkd", D, target)

    # 저주파(k=0) bin에만 교란을 준 예측 vs 고주파(k=chunk-1) bin에만 같은 크기 교란을 준 예측
    perturb = 1.0
    low_freq_perturbed = target_freq.clone()
    low_freq_perturbed[:, 0, :] += perturb
    high_freq_perturbed = target_freq.clone()
    high_freq_perturbed[:, -1, :] += perturb

    pred_low = torch.einsum("kt,bkd->btd", D.T, low_freq_perturbed)
    pred_high = torch.einsum("kt,bkd->btd", D.T, high_freq_perturbed)

    loss_low_perturbed = compute_frequency_consistency_loss(pred_low, target, decay=2.0)
    loss_high_perturbed = compute_frequency_consistency_loss(pred_high, target, decay=2.0)

    check(
        "decay>0이면 저주파 bin의 오차가 고주파 bin의 같은 크기 오차보다 손실에 더 크게 반영된다",
        loss_low_perturbed.item() > loss_high_perturbed.item(),
        detail=f"low={loss_low_perturbed.item()} high={loss_high_perturbed.item()}",
    )


def test_freq_loss_gradient_flow():
    torch.manual_seed(0)
    pred = torch.randn(3, 16, 6, requires_grad=True)
    target = torch.randn(3, 16, 6)
    loss = compute_frequency_consistency_loss(pred, target, decay=1.0)
    check("주파수 일관성 손실은 유한한 스칼라값이다", torch.isfinite(loss).item() and loss.ndim == 0)
    loss.backward()
    check("주파수 일관성 손실이 역전파 가능하다 (pred에 gradient가 흐른다)", pred.grad is not None and pred.grad.abs().sum().item() > 0)


def test_freq_loss_input_validation():
    pred = torch.randn(2, 10, 4)
    target_wrong_shape = torch.randn(2, 10, 5)
    try:
        compute_frequency_consistency_loss(pred, target_wrong_shape)
        check("pred/target shape이 다르면 거부한다", False)
    except ValueError:
        check("pred/target shape이 다르면 거부한다", True)

    target_wrong_ndim = torch.randn(2, 10)
    try:
        compute_frequency_consistency_loss(pred, target_wrong_ndim)
        check("2차원 텐서를 주면 거부한다 (3차원이어야 함)", False)
    except ValueError:
        check("2차원 텐서를 주면 거부한다 (3차원이어야 함)", True)


def test_freq_loss_short_chunk_no_crash():
    pred = torch.randn(2, 1, 4)
    target = torch.randn(2, 1, 4)
    loss = compute_frequency_consistency_loss(pred, target, decay=1.0)
    check("chunk_size=1이면 (DCT가 의미 없으므로) 에러 없이 0을 반환한다", loss.item() == 0.0)


def test_band_errors_diagnostic():
    torch.manual_seed(0)
    pred = torch.randn(2, 30, 4)
    target = torch.randn(2, 30, 4)
    errors = compute_frequency_band_errors(pred, target, n_bands=3)
    check("compute_frequency_band_errors가 n_bands개의 값을 반환한다", len(errors) == 3)
    check("compute_frequency_band_errors의 모든 값이 유한하고 음수가 아니다", all(e >= 0 and e == e for e in errors))


def test_config_validation():
    cfg = SmolVLAConfig(use_freq_policy=True, chunk_size=50)
    check("SmolVLAConfig(use_freq_policy=True)가 정상 생성된다", cfg.use_freq_policy is True)
    check("freq_lambda 기본값은 1.0이다", cfg.freq_lambda == 1.0)
    check("freq_decay 기본값은 2.0이다", cfg.freq_decay == 2.0)

    try:
        SmolVLAConfig(use_freq_policy=True, chunk_size=1)
        check("chunk_size < 2면 use_freq_policy를 거부한다", False)
    except ValueError:
        check("chunk_size < 2면 use_freq_policy를 거부한다", True)

    try:
        SmolVLAConfig(use_freq_policy=True, freq_lambda=-1.0)
        check("freq_lambda가 음수면 거부한다", False)
    except ValueError:
        check("freq_lambda가 음수면 거부한다", True)

    try:
        SmolVLAConfig(use_freq_policy=True, freq_decay=-1.0)
        check("freq_decay가 음수면 거부한다", False)
    except ValueError:
        check("freq_decay가 음수면 거부한다", True)

    # use_freq_policy=False면 위 제약이 전혀 적용되지 않는다 (하위호환 확인).
    cfg2 = SmolVLAConfig(use_freq_policy=False, chunk_size=1, n_action_steps=1)
    check("use_freq_policy=False면 chunk_size=1도 허용된다 (freq_policy와 무관)", cfg2.chunk_size == 1)


def main():
    test_dct_matrix_orthonormal()
    test_dct_matrix_cache()
    test_band_weights()
    test_freq_loss_parseval_equivalence()
    test_freq_loss_zero_when_equal()
    test_freq_loss_prioritizes_low_frequency()
    test_freq_loss_gradient_flow()
    test_freq_loss_input_validation()
    test_freq_loss_short_chunk_no_crash()
    test_band_errors_diagnostic()
    test_config_validation()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)}개 항목 실패: {FAILURES}")
        sys.exit(1)
    print("[OK] FreqPolicy 주파수 일관성 손실 단위 테스트 전체 통과.")
    print(
        "참고: SmolVLAPolicy 전체(VLAFlowMatching.forward/SmolVLAPolicy.forward에 실제로 연결된 "
        "경로)를 생성해서 테스트하지는 않았습니다 — Hugging Face Hub 접근이 막혀 있는 이 환경의 "
        "한계입니다. 대신 freq_policy.py 자체를 직접 검증했고, 실제 forward/backward 연결은 "
        "합성 백본으로 별도 확인했습니다(세션 기록 참고)."
    )


if __name__ == "__main__":
    main()
