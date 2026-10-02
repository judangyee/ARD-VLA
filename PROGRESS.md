# ARD-VLA 5단계 리팩토링 진행 상황

사용량 한도로 세션이 중간에 끊길 수 있어서, 완료한 단계와 다음 단계를 여기 기록한다.
각 단계는 전체 pytest 통과를 확인한 뒤 커밋+푸시한다. 실험(probe/학습/메모리 측정)은
이 작업 범위에 포함되지 않는다 — 코드 수정 + 단위 테스트 + 문서만.

## 단계 목록

- [x] 1. smooth/traj 손실 대상 수정 — `x0_hat` 기반 전환은 이전 세션에서 이미 커밋됨
      (`167aaa7`/`78824ae`/`ae09269`). 이번 작업은 그 위에 `ard_reg_time_weighting`
      ("none" 기본 | "one_minus_t") 옵션을 추가했다.
- [x] 2. ForceHead — `ard_use_force_head`(기본 False), 별도 MLP, GradNorm/옵티마이저/PEFT
      unfreeze 연결.
- [ ] 3. 대칭 대조군 모드 — `ard_symmetric`(기본 False), train_ard.py/compare_smolvla_ard.py/
      compare_bridge_attention.py CLI 연결.
- [ ] 4. 테스트를 `tests/`로 이전 + pytest 설정 + CPU 전용 CI.
- [ ] 5. README 요약/가설 추가 + 새 옵션 문서화 + Layout 갱신 + docs/ 분리.

## 현재 상태

1~2단계 완료.

1단계: `configuration_smolvla.py`에 `ard_reg_time_weighting`("none" 기본 |
"one_minus_t") 필드 + 검증 추가. `ard.py`에 `_reduce_reg_loss` 헬퍼(가중치 없으면
기존 `.mean()`과 bit-for-bit 동일, 가중치 있으면 샘플별 평균 후 가중 평균) +
`compute_ard_losses`에 `reg_time_weights` 파라미터 추가. `modeling_smolvla.py`의
`VLAFlowMatching.forward`에서 `ard_reg_time_weighting="one_minus_t"`일 때
`1 - time`을 계산해서 넘기도록 연결(기본 "none"이면 `None`을 넘겨서 기존과 완전히
동일).

2단계: `ard.py`에 `ForceHead`(suffix_out -> 스칼라 힘, 일반 초기화 — zero-init을 쓰면
GradNorm이 초기 그래디언트를 못 봐서 일부러 안 씀) 추가, `configuration_smolvla.py`에
`ard_use_force_head`(기본 False, `use_ard=True` 필요) 추가. `compute_ard_losses`에
`force_pred` 파라미터 추가 — `force_target`만 있고 `force_pred`가 없으면(즉
`ard_use_force_head=False`인데 force_target이 들어온 경우) `warnings.warn`으로 한 번
경고 후 `force_loss=0`(과거처럼 `actuator_traj_pred`의 엉뚱한 채널을 쓰지 않음).
`modeling_smolvla.py`에 `self.ard_force_head` 추가(추론 경로에서는 호출 안 함 —
`sample_actions`/`denoise_step` 미변경). `get_optim_params()`는 `self.parameters()`
재귀 덕에 별도 수정 없이 자동 포함됨. `profile_memory.py`/`compare_smolvla_ard.py`/
`compare_bridge_attention.py`의 LoRA 적용 후 수동 unfreeze 루프에 `ard_force_head`
추가. `scripts/test_ard.py`에 config 검증 3개, `ForceHead` 단위 테스트, force_pred
유무에 따른 경고/0-처리 테스트, GradNorm-ForceHead 연결 테스트 추가.

`check_env.py`/`test_ard.py`/`test_freq_policy.py`/`test_token_pruning.py` 전부 통과
확인(실험/probe는 이번 작업 범위 밖이라 실행 안 함).

## 다음에 할 일

3단계(대칭 대조군 모드 `ard_symmetric`)부터 시작.
