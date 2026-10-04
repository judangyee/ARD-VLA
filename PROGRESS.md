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
- [x] 3. 대칭 대조군 모드 — `ard_symmetric`(기본 False), train_ard.py/compare_smolvla_ard.py/
      compare_bridge_attention.py CLI 연결.
- [x] 4. 테스트를 `tests/`로 이전 + pytest 설정 + CPU 전용 CI.
- [x] 5. README 요약/가설 추가 + 새 옵션 문서화 + Layout 갱신 + docs/ 분리.

## 현재 상태

1~5단계 전부 완료.

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

3단계: `ard.py`에 `compute_symmetric_losses`(ARD 모드 `compute_ard_losses`는 전혀 건드리지
않고, 새 함수로 분리) 추가 — 양팔 모두 `L_pos + lambda_smooth*L_smooth + lambda_traj*L_traj`
동일 적용, alpha=beta=0.5 고정, force는 ARD와 동일하게 actuator 쪽에만(물리적으로 Actuator
에만 의미 있는 신호라서 — docstring에 근거 명시). head 구조는 `AsymmetricResidualHeads`를
그대로 재사용해 ARD와 파라미터 수가 정확히 같음. `configuration_smolvla.py`에
`ard_symmetric`(기본 False, `use_ard=True` 필요) 추가 — 켜면 `ard_alpha`/`ard_beta`를 항상
0.5/0.5로 강제하고, 사용자가 다른 값을 줬으면 `logging.warning`. `modeling_smolvla.py`의
`SmolVLAPolicy.forward()`에서 `ard_symmetric` 여부로 `compute_ard_losses`/
`compute_symmetric_losses`를 분기(기본 False 경로는 리팩토링 전과 bit-for-bit 동일한 인자로
`compute_ard_losses` 호출). `loss_dict["ard_mode"]`("asymmetric"/"symmetric")를 추가해 로깅
키로 모드를 구분할 수 있게 함. `train_ard.py`/`compare_bridge_attention.py`에
`--ard-symmetric` 플래그 연결. `compare_smolvla_ard.py`는 `--variants`를 쉼표 구분 목록으로
바꿔 `base,ard,symmetric` 세 변형을 지원하도록 재작성(기본값 `"base,ard"`는 이전 동작과
동일). `scripts/test_ard.py`에 config 검증(대칭 모드 강제 alpha/beta 등), 양팔 손실 형태
동일성(입력을 맞바꾸면 stab/act_loss도 맞바꿔짐), 파라미터 수 동일성, force 처리 동일성
테스트를 추가했다.

`check_env.py`/`test_ard.py`/`test_freq_policy.py`/`test_token_pruning.py` 전부 통과
확인(실험/probe는 이번 작업 범위 밖이라 실행 안 함). 모든 CLI 스크립트 `--help` 정상 동작
확인.

4단계: `scripts/test_ard.py`/`test_freq_policy.py`/`test_token_pruning.py`를 `git mv`로
`tests/`로 옮겼다. 각 파일의 커스텀 `check(name, condition, detail)` 헬퍼 구현을
`assert condition, message`로 바꾸고(개별 `check(...)` 호출부는 전혀 안 건드림 — 그래서
기존 체크 개수/내용이 그대로 유지됨), `FAILURES`/`main()`/`if __name__` 하네스는 삭제했다
(pytest가 `test_*` 함수를 자동 수집). `scripts/check_env.py`는 `test_*.py` 이름이 아니라서
이동 대상은 아니지만(사람이 pip install 직후 독립적으로 돌리는 스모크 스크립트로 그대로
둠), 같은 점검을 pytest에도 포함시키려고 `tests/test_env.py`를 새로 추가했다.

`tests/conftest.py`에 `AutoConfig.from_pretrained`/`AutoProcessor.from_pretrained`를 작은
합성 SmolVLM config로 몽키패치하는 fixture(`tiny_smolvlm_monkeypatch`)와, 이를 바탕으로
`SmolVLAConfig` 오버라이드를 받아 `SmolVLAPolicy`를 만드는 팩토리(`build_tiny_policy`)/배치
팩토리(`build_tiny_batch`)를 추가했다 — pytest의 `monkeypatch` fixture를 써서 테스트마다
자동 원복된다. 이걸로 `tests/test_ard_integration.py`(신규)에 두 테스트를 추가해 남아있던
요구사항을 채웠다:
  - ForceHead가 추론 경로(predict_action_chunk/sample_actions/denoise_step)에서 전혀
    호출되지 않음을 `ForceHead.forward`를 "호출되면 실패"로 바꿔치기해서 직접 확인(요청 (d)의
    나머지 절반).
  - 오늘 추가한 세 플래그(`ard_reg_time_weighting`/`ard_use_force_head`/`ard_symmetric`)가
    전부 기본값(off)일 때, 고정 시드 전체 forward+backward 결과가 commit `ae09269`(1~3단계
    작업 시작 전)와 bit-for-bit 동일함을 확인(요청 (f)). 실제로 `ard.py`/`modeling_smolvla.py`/
    `configuration_smolvla.py`를 `git checkout ae09269 -- <paths>`로 그 커밋 버전으로 임시
    교체해서 같은 입력으로 다시 캡처하고, 모든 수치가 정확히 일치하는 것을 먼저 확인한 뒤
    `git checkout HEAD -- <paths>`로 복원했다(작업 트리가 깨끗한 상태로 돌아왔음을
    `git status`로 확인) — 테스트에 박아넣은 EXPECTED_* 상수는 그때 실측한 값이다.

레포 루트에 `pytest.ini`(testpaths=tests, `requires_hub` 마커 등록 — 현재 이 마커를 쓰는
테스트는 없지만 나중에 실제 Hub 접근이 필요한 테스트가 생기면 붙이도록 선반영) 추가,
`requirements.txt`에 `pytest` 추가. `.github/workflows/tests.yml` 추가 — `scripts/install.sh
--no-venv`로 CPU 전용(실제로 GH Actions 러너엔 nvidia-smi가 없어서 install.sh가 자동으로
CPU wheel을 고름) 환경을 만들고 `pytest -m "not requires_hub"`를 돌린다. README의
`scripts/test_*.py` 경로 참조를 전부 `tests/test_*.py`로 갱신하고, "테스트 (pytest)" 절을
새로 추가했다(Layout 섹션 전체 개편은 5단계로 미룸).

bitsandbytes 같은 선택 의존성이 필요한 기존 테스트는 없었다(전부 `scripts/`의 GPU 프로파일링
도구에만 있고 `tests/`로 옮긴 파일들엔 없음) — 그래서 skipif를 쓸 데가 실제로는 없었고,
이 사실과 `requires_hub` 마커로 미리 깔아둔 패턴을 최종 보고에 남긴다.

`pytest`(루트에서 인자 없이), `pytest -m "not requires_hub"`, `scripts/check_env.py` 전부
통과 확인(35개 테스트 전부 통과).

5단계: `docs/` 신설 — `docs/ard.md`(ARD 핵심 구조 + GradNorm + 이번 작업에서 새로 추가된
`ard_reg_time_weighting`/`ard_use_force_head`/`ard_symmetric` 세 옵션 문서화, 1~3단계 전부
처음으로 README 밖에 문서화됨), `docs/bridge_attention.md`, `docs/freq_policy.md`,
`docs/token_pruning.md`, `docs/profiling_tools.md`(count_params/gradient checkpointing/
profile_memory/compare_smolvla_ard/layer_importance — compare_smolvla_ard의 세 변형 비교
사용법은 중복을 피해 docs/ard.md로 링크만 건다)로 README의 기능별 상세 검증 기록을 전부
옮겼다. "검증된 것 vs 아닌 것"(또는 "측정됨") 표기 관례는 그대로 유지했다.

README.md는 맨 위에 3~5줄 가설 요약("Stabilizer/Actuator 역할 비대칭을 구조/손실에
반영하면 대칭 처리보다 낫다")을 추가하고, 각 기능 절을 몇 줄 요약 + docs/ 링크로 압축했다.
Layout 섹션을 `tests/`/`docs/`/`.github/workflows/`/`pytest.ini`/`PROGRESS.md`/
새로 추가된 scripts(`profile_memory.py`/`compare_smolvla_ard.py`/
`compare_bridge_attention.py`/`layer_importance.py`)까지 포함하도록 갱신했다.

`pytest`(35개 전부 통과), `python scripts/check_env.py` 재확인(문서/README만 바꾼
변경이라 코드 동작에 영향 없음을 재확인하는 목적).

## 다음에 할 일 (5단계 기준)

없음 — 5단계까지 전부 완료. 사용자가 요청한 5단계 작업(코드 수정 + 단위 테스트 + 문서)은
전부 끝났고, 실제 GPU/Hub 환경에서의 실험(probe/학습/메모리 측정)은 이번 작업 범위 밖이라
그대로 남겨뒀다 — 각 docs/*.md의 "측정됨"/"아직 확인되지 않은 것" 절 참고.

---

# OpenArm 8DoF 확정 반영 (2026-10, 후속 작업)

로봇이 OpenArm + 공식 그리퍼로 확정되어 액션 차원이 팔당 7→8DoF(관절7+그리퍼1)로 바뀌었다.
요청 범위: (1) arm_dim 7→8, (2) 그리퍼 손실 분리 검토+구현, (3) joint_torque 입력 경로,
(4) wrist 카메라 config 확장, (5) 테스트/스크립트를 새 차원으로 갱신. GPU 실측은 미검증.

## 단계 목록

- [x] 1. `ard_arm_dim` 기본값 7→8 + 모든 스크립트(`train_ard.py`/`profile_memory.py`/
      `count_params.py`/`compare_smolvla_ard.py`/`compare_bridge_attention.py`/
      `verify_with_real_weights.py`) CLI 기본값/라벨 갱신.
- [x] 2. 그리퍼 손실 분리 — `ard_gripper_dim`(기본 1) 추가, `ard.py`에 `_exclude_gripper`.
- [x] 3. joint_torque 입력 — `ARD_JOINT_TORQUE` 배치 키 + `ard_use_joint_torque` config,
      `SmolVLAPolicy.prepare_state()`에서 concat.
- [x] 4. wrist 카메라 — 기존 `image_features` 메커니즘이 이미 범용이라 코드 변경 불필요임을
      확인, 합성 스크립트들의 카메라 이름을 `top`/`wrist_left`/`wrist_right`로 명시.
- [x] 5. 테스트 갱신 + `docs/ard.md` 문서화 + Ponytail 스킬(별도 요청, 별도 커밋)까지 pytest
      통과 확인.

## 현재 상태

**의견 먼저(사용자가 요청한 부분) — 둘 다 사용자가 제안한 방향을 그대로 채택했다:**
- 그리퍼: 그리퍼는 bang-bang성 신호라 smooth/traj(매끄러움 벌점)를 걸면 정상적인 빠른
  개폐를 방해한다 — `L_pos`만 받고 smooth/traj는 제외하는 게 맞다고 판단. `ard_gripper_dim`
  기본값을 1로 둬서(과거의 "모든 새 기능 기본 off" 관례와 달리) 새 하드웨어 스펙에 맞는
  올바른 동작이 기본이 되게 했다 — `0`으로 주면 과거 동작으로 되돌릴 수 있다.
- joint_torque: 별도 인코더 브랜치 대신 `observation.state`에 concat하는 방식을 채택 — 이미
  범용 패딩(`max_state_dim`)을 쓰는 `state_proj`가 그대로 처리해서 새 파라미터가 전혀
  생기지 않고, 정규화(MEAN_STD)도 추가 작업 없이 자동 적용된다. 트레이드오프(토크가 묻힐
  수 있음)와 대안(별도 브랜치로 승격)은 docs/ard.md에 명시. 기본값은 False(끄면 완전히
  기존과 동일) — 이건 학습 파이프라인에 새 데이터 키가 필요한 기능이라 관례대로 opt-in.

**1단계**: `ard_arm_dim`은 애초에 모든 코드에서 순수 파라미터로만 쓰여서(하드코딩 없음)
기본값만 7→8로 바꾸면 됐다. `AsymmetricResidualHeads`/`compute_ard_losses` 등 구조 변경 없음.
각 스크립트의 `--action-dim`/`--state-dim`/`--ard-arm-dim` 기본값과 `compare_smolvla_ard.py`의
`VARIANTS` 딕셔너리(base: 7→8, ard/symmetric: 14→16)도 갱신.

**2단계**: `ard.py`에 `_exclude_gripper(traj_pred, gripper_dim)` 헬퍼 추가 — `gripper_dim<=0`이면
그대로 반환(과거와 bit-for-bit 동일), 아니면 마지막 `gripper_dim`개 채널을 잘라낸다.
`compute_ard_losses`/`compute_symmetric_losses`에 `gripper_dim: int = 0` 파라미터 추가(기본값은
0 — 함수 자체의 하위호환). `configuration_smolvla.py`의 `ard_gripper_dim`(기본 1) + 검증
(`0 <= ard_gripper_dim < ard_arm_dim`). `modeling_smolvla.py`의 `common_ard_kwargs`에
`gripper_dim=self.config.ard_gripper_dim` 연결.

**3단계**: `ard.py`에 `ARD_JOINT_TORQUE = "observation.joint_torque"` 상수 추가.
`configuration_smolvla.py`에 `ard_use_joint_torque`(기본 False) + `validate_features()`에서
state+torque 합산 차원이 `max_state_dim`을 넘는지 사전 검증. `modeling_smolvla.py`의
`SmolVLAPolicy.prepare_state()`에서 concat(키가 없으면 `ValueError`, 조용히 무시하지 않음) —
학습(`forward`)과 추론(`predict_action_chunk`) 양쪽 다 이 메서드를 거치므로 자동으로 적용됨.

**4단계**: SmolVLA의 이미지 입력(`embed_prefix`/`prepare_images`)이 `config.image_features`
dict를 그냥 순회해서 카메라 이름/개수에 코드 변경이 필요 없음을 확인(실제 학습은
`LeRobotDataset`의 피처를 그대로 씀). `profile_memory.py`/`compare_smolvla_ard.py`/
`verify_with_real_weights.py`의 합성 카메라 이름을 `cam{i}` → `top`/`wrist_left`/
`wrist_right`로 바꿔서 이 사실을 명시적으로 드러냄.

**5단계**: `tests/conftest.py`의 `build_tiny_policy`/`build_tiny_batch` 기본 `arm_dim`을
7→8로 갱신. `tests/test_ard_integration.py`의 bit-exact 회귀 테스트
(`test_all_new_flags_off_matches_pre_change_baseline`)는 `ard_gripper_dim=0`을 명시적으로
넘겨서 ae09269 비교 취지를 그대로 유지(값은 그대로 유효). 신규 테스트: 그리퍼 제외
동작(`test_gripper_dim_excludes_last_channel_from_smooth_traj`,
`test_symmetric_losses_gripper_dim_excludes_last_channel`), joint_torque concat/에러/전체
forward 경로(`test_joint_torque_*`, 3개), config validation(`test_joint_torque_config_validation`,
`test_config_validation`에 신규 assert 추가). `docs/ard.md`에 "그리퍼 손실 분리"/"관절 토크
입력"/"Wrist 카메라 추가" 세 섹션 신설, README 요약 갱신.

사용자가 중간에 별도 요청한 Ponytail 스킬 설치(DietrichGebert/ponytail, MIT)는
`.claude/skills/`+`CLAUDE.md`로 완전히 분리된 커밋(`039726d`)으로 처리했다 — 이 작업의
변경사항과 섞이지 않음.

`pytest`(41개 전부 통과), `python scripts/check_env.py`, 모든 수정된 스크립트의 `--help`
정상 동작 확인. GPU/Hub 실측(실제 학습, 메모리 프로파일)은 요청대로 범위 밖 — 미검증으로
남김.

## 다음에 할 일

없음 — 요청한 5개 항목 전부 완료, 커밋 예정(기능별로 분리: arm_dim/gripper/joint_torque+camera
문서/테스트). 실제 GPU/OpenArm 하드웨어로의 검증은 범위 밖으로 남겨뒀다.
