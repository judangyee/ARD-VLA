# FreqPolicy: 주파수 영역 일관성 손실 (`--use-freq-policy`, FreqPolicy(2025)식)

[← README로 돌아가기](../README.md)

FreqPolicy 원 논문은 액션 청크를 DCT(Discrete Cosine Transform)로 주파수 성분으로 분해해서,
저주파(궤적 전체의 형태)부터 고주파(세부 디테일)까지 coarse-to-fine 순서로 자기회귀
(autoregressive) 생성하는 새로운 액션 디코딩 방식입니다. 이 레포는 SmolVLA의 기존
flow-matching 병렬 디코딩(청크 전체를 한 번에 예측, 고정 `num_steps`번 반복 정제)을 그대로
유지하기로 했으므로(생성 메커니즘 자체는 바꾸지 않음), 그 논문의 핵심 통찰 — **저주파(궤적의
전반적인 형태)가 고주파(디테일)보다 더 근본적이고 먼저/더 신뢰성 있게 맞아야 한다** — 만
가져와서, 기존 flow-matching 회귀 손실에 얹는 **추가(additive) 주파수 일관성 손실**로
구현했습니다.

- `lerobot/policies/smolvla/freq_policy.py`(신규) — `build_dct_matrix`: 직교정규(orthonormal)
  DCT-II 변환 행렬을 만든다(`scipy.fft.dct(norm="ortho")`와 동일 정의, 학습 파라미터 없이
  고정 행렬이라 새 `nn.Module`이 필요 없다). `compute_frequency_consistency_loss`: 예측
  velocity(`v_t`)와 목표 velocity(`u_t`)를 청크(시간) 축으로 DCT 변환한 뒤, 저주파 bin에
  더 큰 가중치(`exp(-decay * k/(chunk_size-1))`)를 준 MSE를 계산한다. `decay=0`이면(전
  bin 가중치 1) 이 손실은 **Parseval 정리**에 의해 시간 영역 MSE와 수학적으로 정확히 같다 —
  `decay`를 얼마나 올리는지가 "저주파를 얼마나 더 우선할지"를 결정하는 유일한 새 신호다.
- 참고로 기존 ARD의 `smooth_loss`(1차 차분)/`traj_loss`(2차 차분)도 사실 고주파 성분에
  벌점을 주는 것과 같은 방향의 아이디어입니다(유한 차분은 일종의 고역통과 필터입니다) — 이
  손실은 그걸 1차/2차 차분이라는 좁은 근사 대신, 청크 길이 전체에 대한 완전한 스펙트럼
  분해로 일반화한 것으로 볼 수 있습니다.
- `use_ard`와 완전히 독립적입니다(둘 다 꺼도 켤 수 있음) — `SmolVLAConfig(use_freq_policy=True)`
  만으로 켤 수 있고, `VLAFlowMatching.forward()`에서 계산된 `freq_loss`가 `loss_dict["freq_consistency_loss"]`
  로 로깅되며 `freq_lambda`만큼 가중되어 total loss에 더해집니다.
- 새 학습 파라미터가 전혀 없습니다(순수 신호처리 손실 함수라 `nn.Module`을 새로 추가하지
  않음) — Bridge Attention과 달리 파라미터 수/메모리 증가가 사실상 0에 가까워야 정상입니다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --use-freq-policy
# 가중치/저주파 우선 정도 조정:
python scripts/train_ard.py --dataset-repo-id <...> --use-freq-policy --freq-lambda 0.5 --freq-decay 3.0
# 데이터셋 없이 빌드/메모리만 빠르게 확인:
python scripts/profile_memory.py --use-freq-policy
```

**검증.** `tests/test_freq_policy.py`(오프라인, GPU/Hub 접근 없이 순수 텐서 연산만으로
전부 검증 가능)에서: DCT 행렬의 직교정규성과 역변환 정확도(임의 n에 대해 복원 오차 <1e-8),
`decay=0`일 때 시간 영역 MSE와의 정확한 수학적 등가성(Parseval 정리 자체를 회귀 테스트로
검증), `decay>0`일 때 저주파 bin의 오차가 고주파 bin의 같은 크기 오차보다 손실에 더 크게
반영되는지, gradient 흐름, 잘못된 입력 거부, config 검증까지 확인했습니다. 별도로 소형 합성
SmolVLM 백본으로 `SmolVLAPolicy.forward()`를 실제로 만들어서 `use_ard`/`use_freq_policy`
4가지 조합(둘 다 꺼짐/`use_ard`만/`use_freq_policy`만/둘 다 켜짐) 전부와 `use_bridge_attention`
까지 셋을 동시에 켠 조합까지 forward+backward가 정상 동작하는지, `loss_dict`의
`freq_consistency_loss` 키 존재 여부가 정확히 `use_freq_policy`를 따르는지 확인했습니다.
실제 SmolVLM2 가중치로 이 손실이 실제로 궤적 품질(smoothness, task 성공률 등)을 개선하는지는
아직 확인하지 못했습니다 — 실제 로봇 데이터셋으로 학습해봐야 압니다.
