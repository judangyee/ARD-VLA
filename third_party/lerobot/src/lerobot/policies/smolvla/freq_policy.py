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
"""FreqPolicy(2025)식 주파수 영역 일관성(consistency) 손실.

FreqPolicy 원 논문은 액션 청크를 DCT(Discrete Cosine Transform)로 주파수 성분으로 분해해서,
저주파(청크 전체의 큰 흐름/궤적 형태)부터 고주파(세부 디테일)까지 coarse-to-fine 순서로
자기회귀(autoregressive) 생성하는 새로운 액션 디코딩 방식이다. 이 레포는 SmolVLA의 기존
flow-matching 병렬 디코딩(고정 `num_steps`번의 반복 정제, 청크 전체를 한 번에 예측)을 그대로
유지하기로 했으므로(생성 메커니즘 자체를 바꾸지 않음), 여기서는 그 논문의 핵심 통찰 —
"저주파(궤적의 전반적인 형태)가 고주파(디테일)보다 더 근본적이고 먼저/더 신뢰성 있게 맞아야
한다" — 만 가져와서, 기존 flow-matching 회귀 손실에 얹는 **추가(additive) 주파수 일관성
손실**로 구현했다. 즉 이건 FreqPolicy 논문의 자기회귀 생성 아키텍처를 이식한 게 아니라, 그
논문이 formalize한 "저주파 우선" 원칙을 단일 미분 가능한 손실 항으로 옮긴 것이다.

참고로 ARD의 기존 `smooth_loss`(1차 차분)/`traj_loss`(2차 차분)도 사실 고주파 성분에 벌점을
주는 것과 같은 방향의 아이디어다(유한 차분은 일종의 고역통과 필터다) — 이 손실은 그걸 1차/2차
차분이라는 좁은 근사 대신, 청크 길이 전체에 대한 완전한 스펙트럼 분해로 일반화한 것이라고 볼
수 있다.

`decay=0`이면(주파수별 가중치가 전부 1) 이 손실은 (Parseval 정리에 의해, DCT가 직교
정규화되어 있으므로) 시간 영역에서의 표준 MSE와 정확히 같아진다 — `scripts/test_freq_policy.py`
에서 이 성질 자체를 회귀 테스트로 검증해뒀다.
"""

import torch
from torch import Tensor

_DCT_MATRIX_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}


def build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """길이 n의 실수 신호를 직교정규(orthonormal) DCT-II로 변환하는 (n, n) 행렬을 만든다.

    `D @ x`가 순변환(시간 -> 주파수), `D.T @ X`가 역변환(주파수 -> 시간)이다(D가 직교정규라
    D.T == D^-1). `scipy.fft.dct(x, norm="ortho")`와 동일한 정의를 쓴다:
        X[k] = scale(k) * sum_t x[t] * cos(pi/n * (t + 0.5) * k)
        scale(0) = sqrt(1/n), scale(k>0) = sqrt(2/n)
    """
    t = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # (1, n)
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n, 1)
    basis = torch.cos(torch.pi / n * (t + 0.5) * k)  # (n, n), basis[k, t]
    scale = torch.full((n, 1), (2.0 / n) ** 0.5, device=device, dtype=dtype)
    scale[0, 0] = (1.0 / n) ** 0.5
    return basis * scale


def get_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """`build_dct_matrix`를 (n, device, dtype) 기준으로 캐싱한다 — chunk_size는 학습 내내
    고정이라 사실상 최초 1회만 계산된다."""
    key = (n, device, dtype)
    if key not in _DCT_MATRIX_CACHE:
        _DCT_MATRIX_CACHE[key] = build_dct_matrix(n, device, dtype)
    return _DCT_MATRIX_CACHE[key]


def compute_frequency_band_weights(n: int, decay: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """주파수 bin k=0..n-1에 대해 exp(-decay * k/(n-1)) 가중치를 주고, 평균이 1이 되도록
    정규화한다(decay=0이면 전부 1). k를 (n-1)로 나눠 [0,1] 범위로 정규화해두었기 때문에,
    `decay`의 의미가 chunk_size(n)에 무관해진다 — decay=D면 항상 "최고 주파수 bin의 가중치가
    최저 주파수 bin의 exp(-D)배"가 되도록 스케일된다(예: decay=2 -> 최고 주파수 가중치가
    최저 주파수의 약 13.5%). decay가 클수록 저주파(k가 작은 쪽)에 더 크게 쏠린 가중치가 된다."""
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype) / (n - 1)
    weights = torch.exp(-decay * k)
    return weights * n / weights.sum()


def compute_frequency_consistency_loss(pred: Tensor, target: Tensor, decay: float = 1.0) -> Tensor:
    """pred/target: (batch, chunk_size, dim) — 보통 예측/목표 velocity(v_t/u_t). chunk_size(시간)
    축을 DCT-II로 주파수 영역으로 바꾼 뒤, 저주파 성분에 더 큰 가중치를 준 MSE를 계산한다.

    decay=0이면 이 함수는 (직교정규 DCT가 L2 노름을 보존하므로, Parseval 정리) 시간 영역
    `F.mse_loss(pred, target)`과 수학적으로 동일한 값을 낸다 — 저주파 가중을 얼마나 "추가로"
    주는지가 이 손실의 유일한 새로운 신호다.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape이 달라야 합니다: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"pred/target은 (batch, chunk_size, dim) 3차원이어야 합니다. 받은 shape: {tuple(pred.shape)}")

    chunk_size = pred.shape[1]
    if chunk_size < 2:
        return pred.new_zeros(())

    matrix = get_dct_matrix(chunk_size, pred.device, pred.dtype)
    pred_freq = torch.einsum("kt,btd->bkd", matrix, pred)
    target_freq = torch.einsum("kt,btd->bkd", matrix, target)

    weights = compute_frequency_band_weights(chunk_size, decay, pred.device, pred.dtype)
    per_bin_mse = (pred_freq - target_freq).pow(2).mean(dim=-1)  # (batch, chunk_size)
    return (per_bin_mse * weights[None, :]).mean()


def compute_frequency_band_errors(pred: Tensor, target: Tensor, n_bands: int = 3) -> list[float]:
    """진단용: 주파수 축을 n_bands개 구간(저->고)으로 나눠 구간별 평균 MSE를 반환한다
    (loss 계산에는 쓰이지 않음, 로깅/시각화 전용)."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred/target shape이 (batch, chunk_size, dim)으로 같아야 합니다.")
    chunk_size = pred.shape[1]
    matrix = get_dct_matrix(chunk_size, pred.device, pred.dtype)
    pred_freq = torch.einsum("kt,btd->bkd", matrix, pred)
    target_freq = torch.einsum("kt,btd->bkd", matrix, target)
    per_bin_mse = (pred_freq - target_freq).pow(2).mean(dim=(0, 2))  # (chunk_size,)

    band_edges = torch.linspace(0, chunk_size, n_bands + 1).round().long().tolist()
    errors = []
    for i in range(n_bands):
        lo, hi = band_edges[i], max(band_edges[i + 1], band_edges[i] + 1)
        errors.append(per_bin_mse[lo:hi].mean().item())
    return errors
