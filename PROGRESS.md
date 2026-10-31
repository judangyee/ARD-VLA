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

---

# 폐루프(closed-loop) 구조 구현 (branch: closed-loop-structure)

SmolVLM2 백본 prefix 계산을 한 번만 공유해 Judge(성공 판정)/Selector(다음 서브태스크
선택)/Action expert(행동) 세 갈래로 쓰는 구조. 이번 작업은 **구조 구현만** — 데이터 생성/실제
학습/캘리브레이션 실측은 범위 밖.

## 단계 목록

- [x] 1. 코드 흐름 추적 (수정 없음)
- [x] 2. Judge/Selector 헤드 + 단일 prefix 추론 API
- [x] 3. 학습 손실 연결
- [x] 4. decision_logic.py
- [x] 5. closed_loop.py 러너
- [x] 6. 테스트 + 문서 + 최종 보고

## 현재 상태

브랜치 `closed-loop-structure`를 `claude/smolvla-research-setup-x29mi7`(OpenArm 8DoF 작업
커밋 `b15191b`)에서 분기. 1단계(코드 수정 없이 추적만) 완료, 5개 항목 결과:

1. **`sample_actions()`의 prefix forward 버려지는 첫 반환값**:
   `modeling_smolvla.py:1065` `_, past_key_values = self.vlm_with_expert.forward(...)`.
   `smolvlm_with_expert.py`의 `forward()`를 직접 읽어서 확인 — `outputs_embeds`는
   `[final_norm(vlm_hidden), None]`이다(118-591행, "final norm" 블록이
   `inputs_embeds[0]`에 `models[0].norm()`을 적용). 즉 버려지는 값은 prefix의 **마지막
   레이어, post-norm** hidden state, shape `(batch, prefix_seq_len, vlm_hidden_size)` —
   `judge_feature_layer=-1`(기본값) 케이스에 추가 연산 없이 바로 쓸 수 있다.
2. **Bridge Attention의 `collect_layer_indices`/`last_collected_prefix_layers` 재사용
   가능 여부**: 가능. 같은 `forward()`의 레이어 인터리브 루프 안에서
   `collect_set`/`collected[layer_idx] = inputs_embeds[0]`로 캡처(580행 부근, **final
   norm 적용 전** 시점)하고 루프가 끝나면 `self.last_collected_prefix_layers`에 저장한다.
   단, 이 캡처값은 **pre-norm**이라 (1)의 post-norm 마지막 레이어 값과 스케일이
   다르다 — `judge_feature_layer=-1`과 `judge_feature_layer=<중간 레이어>`를 같은
   파이프라인으로 넣으려면 `judge.py`의 pooling 다음에 자체 `LayerNorm`을 둬서 이
   불일치를 흡수하는 쪽으로 설계(2단계에서 반영). 또한 Bridge Attention과 Judge가 같은
   `forward()` 호출에서 각자 다른 레이어를 요청할 수 있으므로, 호출부(`sample_actions`,
   학습 경로의 동일 호출, 신규 `predict_action_chunk_with_decisions`)에서
   `collect_layer_indices`를 **두 요청의 합집합**으로 넘기고 각자 자기 인덱스만 꺼내 쓰도록
   2단계에서 구현한다.
3. **prefix 시퀀스 배치(이미지→언어→state)와 `prefix_pad_masks`로 masked mean pooling
   가능 여부**: 가능, 확인됨. `embed_prefix()`(750-856행)가 정확히 이미지 임베딩들 →
   언어 임베딩 → state 임베딩 순서로 `torch.cat`하고, 동일한 순서로 `pad_masks`를
   이어붙여 `self.prefix_length`까지 우측 제로패딩한다. 반환값은
   `return embs, pad_masks, att_masks`(856행) — `pad_masks`가 그대로
   `sample_actions()`의 `prefix_pad_masks`가 되어 `judge.py`의 masked mean pooling
   (`(hidden * mask).sum(1) / mask.sum(1)`)에 바로 쓸 수 있다.
4. **지시문 문자열이 토큰화되는 지점**: `processor_smolvla.py`의
   `make_smolvla_pre_post_processors()` 전처리 파이프라인. 흐름은
   `complementary_data["task"]`(평문 문자열 또는 문자열 리스트) →
   `SmolVLANewLineProcessor`(끝에 `\n` 보장) → `TokenizerProcessorStep`(HF 토크나이저,
   `config.vlm_model_name`/`config.pad_language_to`/`config.tokenizer_max_length`)이
   `lang_tokens`/`lang_masks`로 변환 → 이후 `OBS_LANGUAGE_TOKENS`/
   `OBS_LANGUAGE_ATTENTION_MASK` 배치 키로 `_get_action_chunk()`에 들어간다. 이 전처리는
   **정책 바깥(호출자 쪽)에서** 돈다 — `SmolVLAPolicy.predict_action_chunk`/
   `select_action`은 이미 토큰화된 배치를 받는다. 따라서 decision_logic이 만든 지시문
   문자열(목표+현재 서브태스크+완료 단계)은 5단계 `closed_loop.py` 러너가
   `observe()`로 받은 관측치의 `complementary_data["task"]`에 꽂아 넣고 같은 전처리
   파이프라인(`make_smolvla_pre_post_processors`)을 통과시키는 방식으로 연결한다 —
   `predict_action_chunk_with_decisions` 자체는 토큰화를 신경 쓰지 않는다(기존
   `predict_action_chunk`와 동일한 전제).
5. **`select_action()`의 `self._queues[ACTION]` 드레인이 RTC/freq-policy 포함해서 안전하게
   비워지는지**: 안전함, 확인됨. `self._queues[ACTION]`은 `SmolVLAPolicy.reset()`에서
   `deque(maxlen=self.config.n_action_steps)`로 만드는 **ACTION 전용** 큐이고(265-269행),
   `_check_get_actions_condition()`이 비었을 때만 새로 채운다(372-373행). RTC는
   `select_action()` 자체에서 `assert not self._rtc_enabled()`로 막혀 있어 RTC 경로와는
   아예 안 겹치고, RTC가 쓰이는 `predict_action_chunk`(+`prev_chunk_left_over` 같은 RTC
   상태)는 전부 **호출자가 매 호출 인자로 넘기는 값**이라 `RTCProcessor`
   (`modeling_rtc.py`) 자신은 청크 간에 상태를 들고 있지 않다(`self.tracker`는 디버그
   기록용일 뿐 큐와 무관). `freq_policy`는 학습 손실 항(`freq_loss`)일 뿐 추론 큐와 전혀
   상호작용하지 않는다. 결론: `cancel_pending_actions()`는 단순히
   `self._queues[ACTION].clear()`만 하면 되고, 호출자(`closed_loop.py`)가 재시도 시
   `prev_chunk_left_over`를 넘기지 않으면 RTC 쪽도 자동으로 깨끗하게 끊긴다.

**설계에 미치는 영향(2단계 반영 예정)**:
- `judge.py`의 pooling 출력에 `LayerNorm`을 추가해 pre-norm(중간 레이어)/post-norm(-1)
  불일치를 흡수.
- `predict_action_chunk_with_decisions`/`sample_actions`/학습 경로 forward 호출부 모두
  `collect_layer_indices=sorted(set(bridge_indices or []) | ({judge_layer} if judge_layer
  != -1 else set()))` 형태로 합집합 처리.
- `cancel_pending_actions()`는 `self._queues[ACTION].clear()` 한 줄.

## 2단계: Judge/Selector 헤드 + 단일 prefix 추론 API

`configuration_smolvla.py`에 플래그 추가(전부 기본 off): `use_judge`, `judge_hidden_dim`(256),
`judge_feature_layer`(-1), `judge_num_failure_types`(0), `judge_use_progress`(False),
`judge_detach_features`(True, 1차 연구는 백본 고정 전제), `use_selector`, `selector_num_options`
(0). `judge_loss_weight`/`selector_loss_weight`(둘 다 1.0, 3단계에서 실제로 손실에 쓰임)도
미리 같이 추가했다 — 필드 추가 자체는 비활성이라 2단계 bit-exact에 영향 없음. `__post_init__`
검증: `judge_feature_layer`는 -1 또는 0 이상(상한은 트림된 `num_vlm_layers`를 알아야 해서
`VLAFlowMatching.__init__`에서 확인, `ard_bridge_layer_indices`와 동일한 이유), `use_judge=False`
인데 `judge_num_failure_types>0`/`judge_use_progress=True`면 에러, `use_selector=True`면
`selector_num_options>=1` 필수(반대로 `use_selector=False`인데 0이 아니면도 에러).

신규 `policies/smolvla/judge.py`의 `Judge(nn.Module)`: `prefix_pad_masks` 기준 masked mean
pooling + `LayerNorm`(1단계 추적에서 결정한 pre/post-norm 불일치 흡수용) 다음에, 서로 완전히
분리된(trunk 공유 없음) 작은 MLP head 4개 — `success_head`(1, `use_judge`일 때만),
`failure_head`(K, `use_judge` and `judge_num_failure_types>0`), `progress_head`(1, `use_judge`
and `judge_use_progress`), `selector_head`(N, `use_selector` and `selector_num_options>0`).
`forward(prefix_hidden, prefix_pad_masks, allowed_mask=None)`가 `success_logit`/`success_prob`
(=`sigmoid(logit/T)`, `T`=학습되지 않는 buffer `temperature`, 기본 1.0 — 실제 캘리브레이션
fitting은 범위 밖, 슬롯만), `failure_logits`/`failure_probs`, `progress`(0~1 sigmoid 단일값),
`selector_logits`/`selector_probs`(allowed_mask로 비허용 옵션을 `-inf` 마스킹 후 softmax)를
있는 head만 채워서 dict로 반환. `detach_features=True`면 pooled feature를 `.detach()`해서
Judge/Selector 손실이 백본으로 역전파되지 않게 한다.

`modeling_smolvla.py`: `VLAFlowMatching.__init__`에 `self.judge`(둘 다 off면 `None`) 추가,
`judge_feature_layer` 상한을 `num_vlm_layers`로 런타임 검증. 기존 `sample_actions()`의 denoise
루프(`for step in range(num_steps): ...` 블록)를 동작을 전혀 바꾸지 않고 `_denoise_loop()`
헬퍼로 추출(순수 리팩터링 — pytest 41개가 수치까지 그대로 통과하는 것으로 확인). 신규
`sample_actions_with_decisions()`가 prefix forward를 **정확히 한 번** 호출해서(Bridge
Attention의 `collect_layer_indices`와 Judge의 `judge_feature_layer`를 `_merge_collect_layer_indices()`
로 합집합 처리) 그 결과(prefix 마지막 레이어 hidden state 또는 `last_collected_prefix_layers`의
중간 레이어, `past_key_values`, `bridge_kv_layers`)를 Judge와 `_denoise_loop()` 양쪽에
공유한다. `SmolVLAPolicy`에 `predict_action_chunk_with_decisions(batch, selector_mask=None) ->
(actions, decisions)`(`predict_action_chunk`과 동일한 배치 전처리 + 큐 갱신 후
`sample_actions_with_decisions` 호출)과 `cancel_pending_actions()`(1단계 추적 결론대로
`self._queues[ACTION].clear()` 한 줄) 추가.

테스트: `tests/test_judge.py`(신규, `test_ard.py`와 같은 순수 단위 테스트 스타일) —
`Judge`가 `use_judge`/`use_selector`에 따라 필요한 head만 만드는지, 출력 shape/확률 범위,
masked mean pooling이 패딩 구간을 무시하는지, temperature 스케일링, selector
`allowed_mask`가 비허용 옵션 확률을 0으로 만드는지, `detach_features`에 따른 그래디언트
차단/통과, config 검증 6개. `tests/test_ard_integration.py`에 통합 테스트 6개 추가 — 플래그
모두 off면 `predict_action_chunk_with_decisions`의 액션이 `predict_action_chunk`과
bit-for-bit 동일하고 `decisions=={}`인지, prefix forward(`fill_kv_cache=True`)가 정확히
한 번만 호출되는지(래핑해서 호출 횟수 직접 확인), `judge_feature_layer=-1`이 Bridge
Attention 없이도 동작하는지, selector mask가 실제 정책 경로에서도 비허용 옵션을 0으로
만드는지, `cancel_pending_actions()`가 `select_action()`이 채운 큐를 비우는지.

`pytest`(58개 전부 통과 — 기존 41 + 신규 17). 커밋 예정, 이어서 3단계(학습 손실 연결) 진행.

## 3단계: 학습 손실 연결

`VLAFlowMatching.forward()`(학습 경로)의 prefix forward 호출(`(_, suffix_out), _ = ...`)도
추론 경로와 같은 이유로 손을 봤다 — `_`로 버리던 prefix 반환값을 `prefix_out`으로 받고,
`collect_layer_indices`를 Bridge Attention과 `judge_feature_layer`(2단계의
`_merge_collect_layer_indices` 재사용)의 합집합으로 넘긴다. `self.judge`가 있으면
`judge_feature_layer`에 따라 `prefix_out`(마지막 레이어) 또는
`last_collected_prefix_layers[judge_layer]`(중간 레이어)를 뽑아 `self.judge(...)`를 호출해서
raw logit을 포함한 `judge_out`을 `ard_extras["judge_out"]`에 담는다 — **추가 prefix forward
없음**(학습 경로도 2단계 추론 경로와 동일한 "한 번만" 보장). `use_judge=use_selector=False`면
`judge_layer=None`이라 `_merge_collect_layer_indices`가 기존과 똑같은 `self.bridge_layer_indices`
만 돌려주고 `judge_out=None`이라 `ard_extras`도 기존과 동일하게 구성된다 — 이 경로는 bit-exact
회귀 테스트(`test_all_new_flags_off_matches_pre_change_baseline`)가 여전히 통과하는 것으로
확인했다.

`SmolVLAPolicy`에 `_compute_judge_losses(batch, ard_extras)` 추가 — `ard_force_target`과 같은
"배치에 키가 있을 때만 계산, 없으면 0" 패턴: `judge_success`(0/1)→BCE(`success_logit`),
`judge_failure_type`(int, -100 무시)→CE(`failure_logits`), `judge_progress`(0~1)→MSE
(`progress`), `selector_target`(int)+`selector_mask`(선택, 있으면 비허용 옵션을 -inf로 마스킹
후)→CE(`selector_logits`). 각각 `judge_loss_weight`/`selector_loss_weight`로 가중합되고,
계산된 항목만 `loss_dict`에 로그(`judge_success_loss`/`judge_failure_type_loss`/
`judge_progress_loss`/`selector_loss`/`judge_total_loss`) — 전부 없으면 로그 키 자체가 생기지
않고 total loss에도 기여하지 않는다. `forward()`의 `use_ard` 분기/기본 분기 양쪽 모두
`loss = loss + judge_loss`로 합산.

테스트(`tests/test_ard_integration.py`에 3개 추가): judge_*/selector_* 배치 키가 전부
없으면 관련 loss_dict 키가 하나도 안 생기는지, 전부 있으면(실패 유형 -100 포함, selector_mask
포함) 전부 유한한 값으로 채워지는지, `judge_detach_features=True`일 때 **실제 backbone forward를
통과한** judge_out으로 `judge_loss`만 단독 backward해도 `vlm_with_expert`/`action_out_proj`
파라미터에 그래디언트가 전혀 남지 않는지(Judge 모듈 자체의 detach 단위 테스트는
tests/test_judge.py에 이미 있어서, 여기서는 "진짜 prefix forward 경로에 꽂았을 때도" 끊기는지만
확인).

`scripts/profile_memory.py`/`compare_smolvla_ard.py`/`compare_bridge_attention.py`의 LoRA
적용 후 수동 unfreeze 루프(ForceHead 추가 때 이미 패치됐던 자리)에 `policy.model.judge is not
None`이면 그 파라미터도 풀어주는 블록을 추가 — 이 스크립트들엔 아직 `--use-judge` 같은 CLI
플래그가 없어서(요청 범위 밖) 지금은 항상 `judge is None`이라 실행 동작에 변화는 없고, 나중에
Judge/Selector로 실제 파인튜닝할 때 바로 쓸 수 있게만 선반영했다.

`pytest`(61개 전부 통과 — 2단계 종료 시점 58개 + 신규 3개). 세 스크립트 `--help` 정상 동작
확인. 커밋 예정, 이어서 4단계(decision_logic.py) 진행.

## 4단계: decision_logic.py

신규 `policies/smolvla/decision_logic.py` — `import torch`가 전혀 없다(확인:
`grep "^import\|^from"` 결과에 torch 없음, `tests/test_decision_logic.py`도 torch를 쓰지 않고
돈다). 구성:
- `TaskGraphNode`(`id`/`next`/`recoveries`/`max_retries`), `load_task_graph(dict) ->
  dict[str, TaskGraphNode]` — plain dict(YAML에서 읽은 dict여도 그대로 통과)를 받는다.
- `EpisodeState`(`current_node`/`completed`/`retry_counts`/`recheck_counts`).
- `DecisionLogic(task_graph, option_vocab, tau_success=0.7, tau_fail=0.3, max_recheck=3,
  failure_type_to_recovery=None)` — `tau_success`/`tau_fail`은 생성자 인자이고, docstring/주석에
  "placeholder, 실제 분리 지점은 실험(Judge 캘리브레이션 곡선)으로 정해야 한다"고 명시했다.
  `tau_fail > tau_success`면 `__init__`에서 즉시 에러.
- `decide(decisions, state) -> Decision`(`DecisionKind` = ADVANCE/RETRY/RECOVER/RECHECK/STOP):
  `success_prob >= tau_success` → ADVANCE(완료 목록에 추가, 다음 노드로 `state.current_node`
  갱신, `next=None`이면 target=None으로 "종료" 신호), `< tau_fail` → 실패 처리, 그 사이 →
  RECHECK. 실패 처리는 먼저 재시도 한도(`node.max_retries`) 초과 여부를 보고(초과 시 STOP),
  아니면 `_choose_recovery`로 복구 동작을 고른다: `selector_probs`가 있으면 그 argmax를
  `option_vocab`으로 역매핑해서 "안전 규칙"(해당 노드의 `recoveries`에 실제로 속하는지) 통과
  여부를 **항상** 검사하고, 통과 못 하면(또는 Selector 자체가 없으면) `failure_probs`의
  argmax를 `failure_type_to_recovery` rule table로 매핑 — 그 결과도 없거나 허용 안 되면
  RETRY(같은 노드 재시도). RECHECK도 `max_recheck` 한도를 넘으면 STOP.
- `allowed_mask(state) -> list[bool]` — `option_vocab` 중 현재 노드의 `recoveries`에 속하는
  것만 True(2단계 `predict_action_chunk_with_decisions(selector_mask=...)`에 넘기기 전에
  호출부가 `torch.tensor(mask, dtype=torch.bool)`로 감싸야 함 — 이 변환은 torch를 쓰는
  `closed_loop.py`(5단계) 쪽 책임).
- `instruction_text(goal, state) -> str` — "goal | current subtask: <node> | completed: <list>"
  형태 문자열(1단계 추적 기록의 항목 4대로, 이 문자열은 `closed_loop.py`가
  `complementary_data["task"]`에 넣어 기존 전처리 파이프라인에 통과시키는 용도).

`decisions`/`selector_probs`/`failure_probs` 값은 torch 텐서든 plain list든 모두 받을 수
있게 `_argmax`/`_scalar` 헬퍼가 `.item()`/`.argmax()`/`.dim()` duck-typing(있으면 쓰고
없으면 plain list/float로 처리)으로 다룬다 — 이 파일 자체는 torch를 import하지 않는다.

테스트: 신규 `tests/test_decision_logic.py`(12개, torch 미사용) — 3개 임계값 구간(ADVANCE/
RECHECK/RETRY-or-RECOVER), 종료 노드 도달 시 target=None, 재시도/재관측 한도 초과 시 STOP,
Selector 없이 rule table만으로 복구 동작을 고르는지, failure_probs 없으면 RETRY로 떨어지는지,
Selector가 **허용되지 않은** 옵션을 고르면 안전 규칙이 거부하고 rule table로 대체하는지,
허용된 옵션은 그대로 쓰는지, `allowed_mask`가 노드별 `recoveries`와 일치하는지,
`instruction_text`에 목표/현재 서브태스크/완료 목록이 다 들어가는지, `success_prob` 없이
`decide()`를 부르면 에러인지.

`pytest`(73개 전부 통과 — 61 + 신규 12). 커밋 예정, 이어서 5단계(closed_loop.py 러너) 진행.

## 5단계: closed_loop.py 러너

신규 `policies/smolvla/closed_loop.py` — `ClosedLoopEnv`(`typing.Protocol`, duck typing만
문서화 — 상속 강제 없음)로 환경을 받는 `run_closed_loop(policy, env, decision_logic, goal,
start_node, preprocessor=None, max_iterations=50, pre_chunk=None, per_step=None,
post_decision=None) -> dict` 하나로 구현했다.

**루프 설계(spec의 모호한 지점에 대한 내 설계 결정, PROGRESS.md에 명시)**: "실행 → 종료 시
재관측 → judge" 순서를 글자 그대로 "매 iteration마다 두 번 모델을 호출"하는 것으로 읽지
않았다 — 대신 `predict_action_chunk_with_decisions`를 **iteration마다 한 번만** 불러서, 이번
iteration의 관측치에 대한 "액션 청크"와 "Judge 판정"을 동시에 받는다(아키텍처 자체가 "하나의
prefix forward를 세 갈래가 공유"이므로, 이게 실제 구조와 일치). 그래서 각 iteration의
`decisions`는 사실상 "이전 iteration이 실행+재관측한 결과"를 판정한 값이 된다(첫 iteration은
초기 관측치를 그대로 판정) — 여러 control cycle에 걸쳐 진행되는 서브태스크를 자연스럽게
표현한다. 매 iteration: ① `decision_logic.instruction_text(goal, state)`로 지시문 생성 ②
`env.observe()` + 지시문을 `preprocessor`(기존 `make_smolvla_pre_post_processors` 역할,
1단계 추적 기록의 항목 4대로 "task" 키에 지시문을 꽂아 넘긴다)로 배치 변환 ③
`predict_action_chunk_with_decisions(selector_mask=...)` 한 번 호출 ④ 반환된 액션 청크를
`env.execute()`로 한 스텝씩 실행하다 `env.subtask_finished()`가 True가 되면 멈춤(`_execute_chunk`)
⑤ `env.settle()` ⑥ 이미 ③에서 받은 `decisions`로 `decision_logic.decide()` 호출 — ADVANCE가
아니면 `policy.cancel_pending_actions()`로 "남은 청크"를 버린다(ponytail: 이 구현은 청크를
큐에 쌓지 않고 한 번에 다 실행하므로 이 시점에 실제로 버려질 큐 항목은 없지만, 요구사항을
문자 그대로 지키고 — 나중에 select_action류의 큐 기반 실시간 실행으로 바꿔도 안전하도록 —
항상 호출한다. 업그레이드 경로: `_execute_chunk`를 `_queues[ACTION]`을 직접 채우고 매 스텝
popleft하는 방식으로 바꾸면 이 호출이 그 시점부터는 실제로 의미 있는 drain이 된다). 종료
노드 도달(ADVANCE, target=None) 또는 STOP이면 종료, `max_iterations` 초과도 안전하게
`safe_stop()`+종료 처리. `pre_chunk(actions, decisions, state)`/`per_step(step, action,
state)`/`post_decision(decision, decisions, state)` 콜백은 기본 `None`이고 실제 호출은
있지만(인자까지 전달) 그 안에 어떤 로직도 구현하지 않았다(요청대로 "자리만").

반환하는 에피소드 로그 dict: `decision_history`(각 iteration의 노드/Decision kind/target/
reason), `judge_probabilities`(각 iteration의 `decisions`를 `.tolist()`로 python 값화한 것),
`retry_counts`(최종 `EpisodeState.retry_counts`), `termination_reason`
("task_graph_end"/"stop"/"max_iterations_exceeded").

테스트: 신규 `tests/test_closed_loop.py`(2개) — 가짜 환경(`FakeEnv`, 매 청크 첫 스텝에서 바로
서브태스크 종료)과 `build_tiny_policy(use_judge=True)`로 만든 tiny policy를 쓰되,
`predict_action_chunk_with_decisions`만 몽키패치해서 Judge의 랜덤 초기화 출력에 의존하지 않고
success_prob 시퀀스를 스크립트로 고정했다(1차 실패→2차부터 성공) — stage 6이 요구하는 "실패
→ 남은 청크 버리고 재시도 → 성공 → 다음 노드로 → 종료" 패스를 `cancel_pending_actions` 호출
횟수(실패 1번에만)·decision_history(`retry, advance, advance`)·`termination_reason`까지 전부
확인한다. 두 번째 테스트는 계속 실패하면 재시도 한도를 넘겨 STOP + `env.safe_stop()`으로
끝나는지 확인.

`pytest`(75개 전부 통과 — 73 + 신규 2). 커밋 예정, 이어서 6단계(테스트+문서+최종 보고) 진행.

## 6단계: 테스트 재검토 + 문서 + 최종 보고

스펙의 stage-6 체크리스트 6개 항목을 2~5단계에서 이미 단계별로 테스트를 추가해왔던 것과
대조 확인(전부 커버됨, 추가 테스트 없이 통과): 플래그 off → bit-exact, prefix forward
정확히 한 번, 출력 shape/확률 범위/temperature/selector masking, 손실 키 유무에 따른
0/유한값/detach-gradient, 결정 로직 3구간/재시도 한도/전이 거부/rule table, 폐루프 러너
실패→재시도→성공→종료 패스 — 전부 `tests/test_judge.py`/`tests/test_ard_integration.py`/
`tests/test_decision_logic.py`/`tests/test_closed_loop.py`에 있음.

신규 `docs/closed_loop.md` — 구조도(ASCII), 각 블록(Judge/Selector/결정 로직/action
expert/러너) 역할, 설정 플래그 표(10개), 학습 배치 키 표(5개), 새/수정 파일 목록, "검증한
것 / 못 한 것" 절(기존 docs/*.md와 같은 관례). README.md에 "폐루프 구조" 절 추가(플래그
요약 + 문서 링크 + pytest 커맨드) + Layout 절에 새 파일 3개(`judge.py`/`decision_logic.py`/
`closed_loop.py`) + 새 테스트 3개 + `docs/closed_loop.md` 반영.

`pytest`(75개, 변화 없음 — 이번 단계는 문서화만), `python scripts/check_env.py` 재확인(코드
변경 없음 재확인 목적).

## 다음에 할 일

없음 — 요청한 6단계(코드 구조 + 단위 테스트 + 문서) 전부 완료. 최종 보고는 세션 응답에 남김.
실제 Judge/Selector 학습 데이터 생성·학습·임계값/온도 캘리브레이션 실측, 실제 로봇/시뮬레이터
환경에 `ClosedLoopEnv` 프로토콜 구현, 실제 SmolVLM2 가중치로의 검증은 전부 범위 밖으로
남겨뒀다 — `docs/closed_loop.md`의 "검증한 것 / 못 한 것" 참고.

---

# closed_loop.py 러너 수정 (리뷰 피드백 반영)

6단계 완료 후 리뷰에서 4개 문제가 확인됐다: (1) 판정이 한 박자 늦음(실행 전 관측으로
판정), (2) 청크=서브태스크로 혼동(여러 청크짜리 서브태스크를 청크마다 판정·실패 오판),
(3) RECOVER 결정이 실제로 실행되지 않음, (4) 행동 역정규화(postprocessor) 없음. `judge.py`/
`decision_logic.py`의 판정·결정 규칙 자체는 바꾸지 않았다(요청대로) — `closed_loop.py`만
재작성.

## 설계

`need_judgment` 플래그로 "이번에 받은 decisions를 판정에 쓸지"를 명시적으로 관리한다. 매
iteration은 여전히 `predict_action_chunk_with_decisions`를 정확히 한 번 부르지만:
- `need_judgment=False`(서브태스크 진행 중)면 이번 decisions는 버리고 actions로 다음 청크만
  실행한다 — 여러 청크짜리 서브태스크가 중간에 실패로 오판되지 않는다.
- 서브태스크가 끝나면(`env.subtask_finished()` 또는 `max_subtask_steps` 초과) `env.settle()`
  후 `need_judgment=True`로 표시한다.
- **다음** iteration이 새로 관측(=방금 끝난 서브태스크의 결과 반영)+forward해서 그
  decisions로 `decide()`를 부른다 — 판정이 항상 실행 *후* 관측을 본다.

결정별 처리:
- `STOP`/`ADVANCE(target=None)` — 즉시 종료.
- `RECHECK` — 행동 실행 없이 `env.settle()` 후 `continue`(need_judgment는 True 유지, 다음
  iteration에서 또 판정).
- `RECOVER` — `env.run_recovery(target)`(선택 메서드)를 실제로 호출 + `env.settle()` 후
  `continue`(지시문/환경이 바뀌었으니 다시 forward). 환경에 `run_recovery`가 없으면
  (`hasattr` 체크) `warnings.warn()` 후 `RETRY`와 동일하게 처리하고(추가 forward 없음)
  `decision_history`에 `recovery_unsupported=True`를 남긴다 — NotImplementedError로 죽이지
  않는다(사용자 지시대로, 실제 복구 구현은 환경 쪽 스크립트 몫).
- `RETRY` — 지시문이 같으므로 이번 forward의 actions를 그대로 실행(`continue` 없음, 추가
  forward 없음).
- `RETRY`/`RECOVER`/`RECHECK` 공통: `policy.cancel_pending_actions()` 호출.

실행 단계는 `pre_chunk` 훅(역정규화 전 raw actions) → `postprocessor(actions)`(신규 인자,
역정규화) → `_execute_chunk`(신규: `max_subtask_steps`의 남은 예산을 받아 그만큼만 실행하고,
`(실행한 스텝 수, env.subtask_finished()로 끝났는지)`를 돌려줌) 순서. 서브태스크 스텝 예산
초과(`budget_exceeded`)도 `env.subtask_finished()`와 동일하게 "서브태스크 종료"로 취급한다.

`max_subtask_steps`는 **task graph 노드별 필드가 아니라 러너 레벨 단일 인자**로 뒀다
(ponytail: YAGNI — 노드별 예산이 실제로 필요해지면 그때 `TaskGraphNode`에 필드를 추가하면
된다). `decision_logic.py`는 이 변경 때문에 전혀 손대지 않았다 — `decide()`는 여전히
"판정이 실제로 일어날 때 한 번 호출된다"는 전제만 필요하고, 그 호출 빈도가 iteration 대비
어떻게 분포하는지는 `decision_logic.py`가 알 필요가 없다.

로그에 `forward_calls`(iteration마다 항상 1 — 매 iteration이 정확히 한 번만 forward하므로)와
`judged`(그 iteration에서 실제로 판정이 일어났는지)를 추가했다. "경계당 forward 몇 번"은
이 리스트를 합산해서 알 수 있다: 서브태스크 진행 중(판정 없음)은 청크당 1, `RETRY` 경계는
1(판정+실행이 같은 iteration), `ADVANCE`/`RECOVER` 경계는 2(판정 iteration + 재forward
iteration), `RECHECK`는 호출마다 1씩 쌓인다.

## 테스트

`tests/test_closed_loop.py`를 전면 재작성(10개): 기존 2개(전체 패스/재시도한도)를 새 루프
흐름에 맞게 갱신하고(한쪽은 `forward_calls`/`judged` 로그까지 정확히 검증), 8개 신규 —
①판정이 실행 후 관측(obs_id로 직접 확인)을 쓰는지, ②청크 3개짜리 서브태스크에서 앞 두
청크 뒤에는 `decide()`/`settle()`이 불리지 않는지, ③`RECOVER`가 `env.run_recovery(id)`를
실제로 부르고 같은 노드를 재시도하는지, ④`run_recovery`가 없는 환경에서 경고+`RETRY`
대체+`recovery_unsupported` 로그, ⑤`RECHECK`이 행동 실행 없이 재판정만 하는지(+재관측
한도 초과 시 STOP), ⑥`postprocessor`가 실행 직전에 적용되고 `pre_chunk`는 그 전 값을
받는지, ⑦`max_subtask_steps` 초과 시 판정으로 넘어가는지. 전부 `FakeEnv`(obs_id 카운터,
`finish_after`로 "청크 1개=서브태스크"와 "여러 청크짜리 서브태스크" 둘 다 표현,
`support_recovery` 플래그로 `run_recovery` 유무 전환) + `predict_action_chunk_with_decisions`
몽키패치 스텁(호출 순번별로 success_prob/failure_probs를 스크립트로 고정)으로 Judge의 실제
랜덤 초기화 출력에 의존하지 않는다.

`pytest`(83개 전부 통과 — 75 + 신규 8, 기존 2개는 갱신). `docs/closed_loop.md`/README의
"prefix forward 1회" 설명을 "청크당 1회, 경계에 따라 1~2회"로 정정하고, `closed_loop.py`
모듈 docstring도 전면 재작성.

## 다음에 할 일

없음 — 요청한 4개 문제(판정 시점/청크-서브태스크 구분/RECOVER 실행/역정규화) 전부 해결,
pytest 전체 통과, 문서 정정 완료. 실제 환경에 `run_recovery()`를 구현하는 것, 실제
SmolVLM2 가중치로 판정 시점 수정의 효과를 확인하는 것은 여전히 범위 밖.

---

# LAC(Latent-Aware Controller) 추론 필터 추가 (branch: lac-inference)

Co-VLA(arXiv 2606.20285)의 LAC를 ARD-VLA에 이식 — 학습 없는 추론 시점 EMA 필터로, ARD의
"공유(action_out_proj)+잔차(ARD 헤드)" 가산 구조를 보고 매 실행 스텝마다 필터 강도 α를
조절한다. Co-VLA 원본 재현(`lac_mode="covla"`) + ARD 역할 인식 확장(`lac_mode="role_aware"`)
둘 다 구현.

## 0단계 확인 결과 (코드 수정 전 보고)

- **가산 구조 확인(A안)**: `denoise_step()`/`sample_actions()` 둘 다 `v_t =
  action_out_proj(suffix_out)`(공유 베이스) 뒤에 ARD 헤드 잔차를 양팔 채널에 **더한다**
  (`ard.py`의 `AsymmetricResidualHeads`는 마지막 레이어 zero-init — 학습 전엔 잔차≡0).
  단, 이건 flow-matching **velocity** 레벨의 가산이라, 최종 액션에서 공유/잔차를 뽑으려면
  마지막 denoise 스텝 하나만 보면 안 되고 **루프 전체에서 베이스/잔차 velocity를 따로
  적분**해야 한다(선형 적분이라 정확히 분해됨) — 그래서 `sample_actions_with_role_split()` +
  `_denoise_loop()`의 누적기를 새로 추가했다.
- **16 DoF 배치**: `ard_arm_dim=8`(관절7+그리퍼1), 채널 `[0:8]`=왼팔, `[8:16]`=오른팔, 그리퍼는
  각 팔 블록의 마지막 1채널. `ard_default_actuator_arm="right"`(기본) → Actuator=오른팔.
- **chunk 실행 지점**: `select_action()`의 `_queues[ACTION]`(deque)이 유일한 "하나씩 꺼내는"
  경로. `predict_action_chunk`/`predict_action_chunk_with_decisions`(closed_loop.py 러너가
  씀)는 청크를 통째로 반환하는 별도 경로 — **LAC는 `select_action()`에만 연결**하기로 결정
  (사용자 확인 없이 "진행해"로 승인받은 뒤 이 범위로 진행, PROGRESS.md에 명시).
- **역할 할당**: 고정 config뿐(`resolve_actuator_is_first`) — role classifier 없음(폐루프의
  Judge/Selector는 성공 판정/서브태스크 선택용이라 팔 역할과 무관). role_aware의 "classifier
  출력이 있으면 그걸 따르게" 분기는 `stabilizer_is_left` 인자로 자리만 만들고 classifier
  자체는 구현하지 않음.
- **mirror_sign**: `/home/user/ard-gen/ard-gen/assets/openarm/openarm_bimanual.xml`에서 좌우
  joint의 `axis`/`range`를 직접 비교 — joint1/2/6과 그리퍼가 미러링(부호 반대), joint3/4/5/7은
  아님. 기본값 `[-1,-1,1,1,1,-1,1]`(joint1~7)을 제안했지만, **이건 MJCF 텍스트 비교일 뿐
  forward kinematics로 검증하지 않았다(미검증)**. 부가 발견: 현재 ARD-Gen의
  `pipeline/to_lerobot.py`는 peg_in_hole/insertion에서 `action = right_arm.action`만
  내보내서(왼팔은 고정 그립) 실제로는 16-DoF 양팔 제어 데이터셋이 아직 없음 — mirror_sign을
  실제 양팔 데이터로 검증할 방법이 현재는 없다.

## 구현

**`policies/smolvla/lac.py`(신규)** — `LACConfig`(dataclass), `LatentAwareController`:
- `step(a_shared_L, a_shared_R, a_res_L, a_res_R, q_t, stabilizer_is_left=True) -> (q̃_t,
  debug)`. 그리퍼 채널은 내부에서 분리해 ρ/ω 계산과 필터링 양쪽에서 제외하고 그대로
  pass-through.
- 첫 스텝(`reset()` 직후)은 `q̃_0 = q_1`을 **EMA 수식을 타지 않고 직접 반환** — 처음엔
  `(1-a)*q+a*q`로 구현했다가 부동소수점 라운딩 때문에 `q`와 비트 단위로 안 같아서 통합
  테스트(`test_select_action_with_lac_covla_passes_through_first_step_then_filters`)가
  실패, 특수 분기로 고쳤다.
- role_aware: `_role_alpha_hat()`이 공유 regime 추정치에서 Stabilizer는 상한만 낮추고
  (`alpha_stab_max`), Actuator는 협동(ω>τ_ω)일 때만 `alpha_act_cooperative`로 덮어쓴다(그 외엔
  공유 추정치 그대로 — "떨림일 때만 낮춤"이 자동 성립). mirror_sign은 항상 **오른팔**
  잔차에(역할과 무관하게) 적용한 뒤 역할 해석.
- `SmolVLAConfig`에 `lac_mode`("none"/"covla"/"role_aware", 기본 "none") + 관련 수치 플래그
  10개. `__post_init__` 검증: `use_ard=True` 필요, RTC(`rtc_config.enabled=True`)와 동시
  사용 금지(RTC의 velocity 블렌딩이 LAC의 선형 적분 전제를 깨서), `lac_mirror_sign` 길이
  검증.

**`modeling_smolvla.py` 연결** — 기존 `last_collected_prefix_layers` side-channel과 같은
패턴으로 설계:
- `denoise_step()`에 `role_split_sink: dict|None=None` 파라미터 추가 — `None`이면(기본)
  기존과 완전히 동일, 주어지면 ARD 잔차(`left_residual`/`right_residual`)만 써넣고 반환값은
  안 바꿈.
- `_denoise_loop()`가 `self._lac_role_split_accum`(기본 None, `VLAFlowMatching.__init__`에서
  초기화)이 설정돼 있을 때만 매 스텝 `dt * residual`을 누적 — 다른 모든 호출 경로는 이 값이
  항상 None이라 분기 자체가 실행 안 됨(bit-exact 유지, 기존 바이트 단위 회귀 테스트로 확인).
- 신규 `sample_actions_with_role_split()`(`sample_actions()`는 **전혀 수정하지 않음**) —
  누적기를 설정→`_denoise_loop()` 호출→`shared = x_t - residual_accum`으로 역산해서
  `(x_t, {shared_left, shared_right, residual_left, residual_right})` 반환.
- `SmolVLAPolicy.reset()`이 `lac_mode != "none"`이면 `self._lac`(LatentAwareController)를
  config에서 만들고, `lac_log=True`면 `self.lac_debug_log = []`.
- `select_action()`이 `lac_mode != "none"`이면 신규 `_get_action_chunk_with_lac()`(→
  `sample_actions_with_role_split()` 호출)로 분기, 아니면 기존 `_get_action_chunk()`
  (미변경) — `_apply_lac_filter()`가 반환된 청크 전체(`n_action_steps`개)를 시간 순서대로
  `self._lac.step()`에 하나씩 넣어 필터링한 뒤 큐에 넣는다. **"청크 생성 시점에 한 번에
  필터링"이지 "pop할 때마다 지연 필터링"이 아니다** — 수학적으로는 동일(생성↔실행 사이에
  새 정보가 없는 순수 순차 재귀)하지만 구현이 훨씬 단순해서 이 방식을 택했다(별도의
  shared/residual 큐가 필요 없음). `predict_action_chunk`/
  `predict_action_chunk_with_decisions`(closed_loop.py 러너)는 건드리지 않았다 — 0단계
  보고에서 올린 스코프 질문에 사용자가 "진행해"로 답해서, 과제 문구를 글자 그대로("chunk에서
  액션 하나씩 꺼내는 곳" = `select_action()`) 따르는 쪽으로 결정했다.

**`scripts/compare_lac_modes.py`(신규)** — `tests/conftest.py`와 같은 몽키패치로 합성
SmolVLM 백본을 만들어 none/covla/role_aware를 비교 실행. ARD 헤드가 zero-init이라 섭동 없이
돌리면 잔차≡0 → 항상 "macro" regime으로 결정론적(`--no-perturb-ard-heads`로 확인 가능,
covla는 alpha=0.8 고정, role_aware는 Stabilizer가 0.5로 캡되는 게 바로 보임) — 그래서
`--perturb-ard-heads`(기본 켬)로 ARD 헤드 마지막 레이어에 작은 가우시안 잡음을 주입해서
"약간 학습된" 것처럼 흉내 낸 잔차로도 비교한다(이것도 실제 학습된 가중치가 아니라는 걸
스크립트 docstring에 명시). jerk(3차 차분 크기 평균) 비교 결과 covla/role_aware가 none보다
낮게 나왔다(각 세션의 특정 시드에 대한 관찰일 뿐, 일반적 결론은 아님).

## 테스트

- `tests/test_lac.py`(18개, 순수 단위 테스트): Co-VLA 손계산 3케이스(precision/noise/macro
  regime이 수식과 정확히 일치), α clip, β 평활화, 첫 스텝 pass-through, `reset()`, 그리퍼
  pass-through, role_aware(Stabilizer 상한/Actuator 협동 승격/떨림 시에만 하향/
  stabilizer_is_left 반영), mirror_sign의 ω 부호 반전 + 길이 검증, `SmolVLAConfig`의
  `lac_mode`/`use_ard` 필요/RTC 금지/`lac_mirror_sign` 길이 검증 4개.
- `tests/test_ard_integration.py`에 LAC 통합 테스트 6개 추가: `sample_actions_with_role_split`
  의 (shared+residual) 분해가 `sample_actions()`(미변경)와 bit-for-bit 동일한 x_t를 내고
  정확히 재구성되는지, `lac_mode="none"`이 기존과 bit-for-bit 동일한지, `select_action()`
  에서 첫 스텝은 그대로 통과하고 두 번째 스텝부터 실제로 달라지는지(같은 noise로 unfiltered
  버전과 직접 비교), `lac_log`가 "호출마다"가 아니라 "청크 생성마다 n_action_steps개"로
  쌓이는지, `reset()`이 LAC 상태를 재초기화하는지.
- 전체 `pytest` 106개 통과(`test_ard.py`/`test_token_pruning.py`/`test_freq_policy.py`
  등 기존 테스트 전부 포함, 회귀 없음 확인).

## 검증한 것 / 못 한 것

`docs/lac.md`의 "검증한 것 / 못 한 것" 절에 상세 정리(README에서 링크). 요약: LAC 수식·
role_aware 분기·mirror_sign 메커니즘·select_action 연결·bit-exact(off)는 합성 데이터로
검증됨. 실제 SmolVLM2 가중치, 실제 OpenArm 시뮬/하드웨어, mirror_sign의 forward-kinematics
검증, `predict_action_chunk` 계열(closed_loop 러너) 연결, 하이퍼파라미터(τ/Δ/β) 실측은
전부 범위 밖/미검증.

## 다음에 할 일

없음 — 요청한 3개 목표(Co-VLA 원본 재현, 역할 인식 확장, 비교 코드/테스트) 전부 완료.
추후 작업 후보: mirror_sign을 실제 양팔 데이터/시뮬로 검증, 실제 가중치로 하이퍼파라미터
재조정. "predict_action_chunk_with_decisions에도 LAC 연결"은 바로 다음 작업(아래)으로
완료됐다.

---

# LAC을 ARD-VLA 2.0 폐루프 경로에도 연결 (branch: lac-closed-loop)

이전 작업은 LAC를 `select_action()` 경로에만 연결했다. 이번 작업은 `predict_action_chunk_
with_decisions()`/`closed_loop.py`(ARD-VLA 2.0 폐루프 러너)에도 **같은 `self._lac`
인스턴스·같은 필터 코드**로 연결한다 — chunk 단위가 아니라 실제 실행 스텝마다, 서브태스크
전환/재시도·복구 진입 시 리셋하는 규칙까지 포함.

## 0단계 확인 결과

- **predict_action_chunk_with_decisions() 반환 형태**: `(actions, decisions)` — `actions`는
  (batch, chunk_size, action_dim) 전체 청크, `decisions`는 Judge/Selector 판정 dict. prefix
  forward가 정확히 한 번만 일어나도록 `sample_actions_with_decisions()`를 부른다.
- **러너가 스텝을 꺼내 실행하는 곳**: 러너 안(`closed_loop.py`의 `_execute_chunk()`) — 러너는
  `select_action()`을 전혀 안 부른다(`predict_action_chunk_with_decisions()`만 직접 호출).
  → **select_action()과 폐루프 경로가 한 스텝에 겹쳐서 두 번 걸릴 위험은 원래 없다**(서로
  다른 호출 경로이고, 폐루프는 select_action을 안 거침). 코드 검토로 확인, 직접 테스트로도
  재확인(`test_closed_loop_lac_applied_exactly_once_per_step`).
- **서브태스크 전환/재시도·복구가 일어나는 곳**: 전부 `closed_loop.py`의 `run_closed_loop()`
  안 — `decide()`가 반환한 `DecisionKind`에 따라 `ADVANCE`(target≠None, 서브태스크 전환)/
  `RETRY`/`RECOVER`(실패→재시도·복구) 분기에서 처리.

## 설계/구현

1. **predict_action_chunk_with_decisions() 확장** — `lac_mode != "none"`이면 신규
   `VLAFlowMatching.sample_actions_with_decisions_and_role_split()`(기존
   `sample_actions_with_decisions()`와 완전히 같은 prefix-forward-once Judge/Selector
   계산 + `sample_actions_with_role_split()`의 공유/잔차 적분을 합친 자매 메서드 — 이
   코드베이스가 이미 `sample_actions`→`sample_actions_with_decisions`→
   `sample_actions_with_role_split` 세 자매로 풀던 것과 같은 패턴)을 불러서 `decisions
   ["_lac_role_split"]`에 분해를 담아 돌려준다. `lac_mode="none"`이면 기존
   `sample_actions_with_decisions()` 호출이 전혀 안 바뀐다 — bit-exact.
   - **반환 시그니처를 안 바꾼 이유**: `(actions, decisions)` 2-tuple을 그대로 유지하고
     `_lac_role_split`을 `decisions` dict 안에 추가 키로 넣는 쪽을 택했다(3-tuple로
     바꾸는 대안도 검토했지만, 이미 이 메서드를 2-tuple로 unpack하는 기존 호출부/테스트가
     `closed_loop.py` 1곳 + `tests/test_ard_integration.py`/`test_closed_loop.py` 9곳
     있어서, 전부 고쳐야 하는 불필요한 blast radius가 생긴다). `_lac_role_split`은 Judge/
     Selector의 진짜 판정이 아니라는 걸 이름의 밑줄 접두사로 표시했고, 러너가 `decide()`/
     로깅 전에 `pop()`해서 쓰므로 그 둘은 이 키를 절대 보지 않는다(요청 (4)).
2. **실행 스텝마다 적용** — `predict_action_chunk_with_decisions()`는 **필터링 전 raw
   액션**을 그대로 돌려준다(이것도 테스트로 직접 확인). 실제 필터링은
   `closed_loop.py`의 `_execute_chunk()`가 `env.execute()` 바로 직전에 그 스텝의 공유/
   잔차 조각으로 `policy.apply_lac_step()`을 불러서 한다 — chunk 전체를 한 번에 거는 게
   아니라 정말 스텝 하나씩.
3. **복붙 없이 코드 공유** — 신규 `SmolVLAPolicy.apply_lac_step(action_t, role_split_t) ->
   filtered`를 `select_action()` 쪽 `_apply_lac_filter()`(청크 전체를 이 메서드로 한 스텝씩
   돌리는 걸로 리팩터)와 `closed_loop.py`의 `_execute_chunk()`가 똑같이 부른다. LAC 컨트롤러
   인스턴스도 `self._lac` 하나뿐 — 두 경로가 같은 상태를 공유한다(단, 동시에 두 경로를 같은
   정책 인스턴스로 섞어 쓰면 상태가 꼬일 수 있다는 뜻이기도 하다 — 보통은 한 정책을 한
   경로로만 쓰므로 문제 없음, 문서에 명시는 안 했지만 기록만 남김).
4. **리셋 규칙**(가장 중요한 요구사항) — `run_closed_loop()`가:
   - 시작할 때 `policy.reset()`(큐까지 전체 리셋) → 로그 이유 `"episode"`.
   - `ADVANCE`(다음 노드로, target≠None) 때 신규 `_reset_lac()` 헬퍼로 `policy._lac`만
     리셋(큐/디버그 로그는 에피소드 전체에 걸쳐 누적되게 그대로 둠) → 이유 `"subtask"`.
   - `RETRY`/`RECOVER`(실패 → 재시도/복구, `run_recovery` 없어서 RETRY로 대체되는 경우
     포함) 때도 `_reset_lac()` → 이유 `"retry"`.
   - `RECHECK`(아직 불확실 — 같은 시도를 다시 관측만 함)은 리셋 안 함 — 리셋 규칙의 "이전
     명령을 천천히 따라가는 필터가 새 시도로 끌려가지 않게 한다"는 취지가 RECHECK에는
     적용 안 됨(새 시도가 아니라 같은 시도를 계속 보는 것).
   - `log_lac_events=True`면 반환 로그의 `lac_resets`(iteration/이유 목록)와
     `lac_debug_log`(`policy.lac_debug_log`)로 확인 가능.
5. Judge/Selector의 입력·판정 로직(`judge.py`/`decision_logic.py`)은 전혀 안 건드렸다 —
   `_lac_role_split`은 `decide()`에 도달하기 전에 빼낸다(요청 (4) 그대로).

## 테스트

`tests/test_lac_closed_loop.py`(7개): `lac_mode="none"`이면 `decisions`에
`_lac_role_split`이 없는지 + 같은 시드로 두 번 돌려 완전히 동일한지, Judge의
`success_prob`이 lac_mode와 무관하게 동일한지(같은 noise로 비교), `predict_action_chunk_
with_decisions()`가 raw 액션을 돌려주는지(필터링이 러너에서만 일어난다는 증거),
`apply_lac_step()`의 그리퍼 pass-through, 리셋이 episode/subtask/retry 세 시점에
정확히(스크립트로 고정한 시나리오: pick 1차 실패→RETRY→pick 2차 성공→ADVANCE to place) 일어
나는지, 그리고 독립적으로 재현한 단일 패스 LAC 참조값과 실제 실행된 두 스텝이 정확히
일치하는지(=한 스텝에 두 번 안 걸림).

`pytest`(113개 전부 통과 — 106 + 신규 7). 기존 `test_lac.py`/`test_ard_integration.py`/
`test_closed_loop.py`/`test_decision_logic.py` 등 전부 회귀 없음.

## 검증 실행

`scripts/compare_lac_closed_loop_modes.py`(신규) — 실제 `run_closed_loop()` + Judge(zero-
init이 아니라서 success_prob이 매 스텝 랜덤) + 결정 로직을 합성 백본으로 몇 에피소드 끝까지
돌려서 jerk/`lac_resets`/regime 분포를 비교. 결과: covla/role_aware가 none보다 jerk를
낮췄다(16.08 → 9.50, 섭동 적용 시). 다만 이번 특정 실행에서는 랜덤 Judge 가중치가
success_prob을 RECHECK 구간에만 떨어뜨려서 재관측 한도 초과(STOP)로 끝났고, 그래서
`lac_resets`에 `"subtask"`/`"retry"`가 안 찍혔다 — 그 둘이 실제로 불리는지는 랜덤성과
무관한 `tests/test_lac_closed_loop.py`의 스크립트 테스트가 보장한다. 이 사실과 "랜덤
초기화라 regime 분포 자체는 의미가 없을 수 있다"는 점을 스크립트 출력에 그대로 남겼다.

## 검증한 것 / 못 한 것

`docs/lac.md`에 "폐루프 리셋 규칙" 절을 추가하고 "검증한 것/못 한 것"을 갱신했다. 요약:
반환 시그니처 설계·리셋 3시점·raw-액션-반환(이중 적용 방지)·그리퍼 pass-through는 합성
데이터로 검증됨. 실제 SmolVLM2 가중치, 실제 OpenArm 시뮬/하드웨어, "실제 Judge 분포가 세
결정을 골고루 만드는지"는 범위 밖/미검증(기존 LAC 작업의 미검증 항목과 동일한 제약).

## 다음에 할 일

없음 — 요청한 작업(ARD-VLA 2.0 폐루프 경로 LAC 연결, 실제 실행 스텝 단위 적용, 3가지
리셋 규칙, 복붙 없는 코드 공유, 테스트, 검증 실행, 문서) 전부 완료.

---

# 설정 가능한 ARD(Configurable ARD) + RoCo Challenge 프리셋 (branch: configurable-ard)

RoCo Challenge(대회마다 과제가 바뀌는 양팔 조립 대회) 참가를 위해, OpenArm 16채널/왼팔=
Stabilizer/chunk 50/L2 손실로 고정돼 있던 ARD-VLA를 설정만으로 다른 과제에 맞출 수 있게
바꿔달라는 요청. 학습 실행(GPU)은 범위 밖, 구조 구현 + 작은 합성 모델 검증까지.

## 0단계: 하드코딩 위치 조사 (코드 수정 전 보고)

사용자 지시대로 코드를 먼저 바꾸지 않고 6가지 항목을 조사해서 보고한 뒤 진행 승인을
받았다. 핵심 발견: **"16 DoF가 하드코딩"이라는 걱정의 대부분은 이미 config화돼 있었다**
(`ard_arm_dim`/`ard_gripper_dim`/`ard_default_actuator_arm`에 리터럴 16/8 없음, grep
확인). 진짜 하드코딩은:
- `ard.py`의 `split_by_role`/`combine_by_role`/`AsymmetricResidualHeads`가 "연속·동일
  길이·왼팔먼저" 구조를 함수 시그니처(`arm_dim: int` 하나)로 강제하는 것(숫자가 아니라
  구조 가정).
- `modeling_smolvla.py:1236`의 `F.mse_loss(u_t, v_t, reduction="none")` — 메인 회귀
  손실이 진짜로 L2 하나로 고정.
- chunk_size/n_action_steps는 이미 config대로 정확히 동작(검증까지 있음) — 하드코딩
  아님, 수정 불필요.
- 액션 정규화(`processor_smolvla.py`의 `NormalizerProcessorStep`)는 차원별 평균/표준편차
  만 쓰고 시간축 awareness 전혀 없음 — `per_timestep_norm`은 순수 신규 기능.
- `freeze_vision_encoder`(기본 True)는 **이미 존재**. `_get_default_peft_targets()`도
  이미 차원 독립적.
- `quantization` 옵션은 레포 전체에 **존재하지 않음**(grep 0건) — 작업 지시의 전제
  ("기존 옵션은 그대로 두되")가 실제와 달랐다.
- RoCo 공식 데이터셋(`rocochallenge2025/rocochallenge2026_Industrial_Assembly`)은 HF
  Hub 차단으로 스키마를 못 열어봄 — 공개 스펙으로 진행, 미확인 표시(기존 모든 작업과
  동일한 샌드박스 제약).

보고 후 사용자에게 두 설계 결정을 확인받았다: (1) `ArmLayoutConfig`를 완전 일반화(임의
비연속/비대칭 슬라이스)하지 않고, 기존 `ard_arm_dim` 등을 그대로 재사용하는 **얇은
래퍼**로 간다(RoCo의 7+7 레이아웃도 기존 "연속·동일길이" 패턴으로 완전히 표현 가능해서
`modeling_smolvla.py`의 7곳 슬라이스 조립 코드를 리팩터링할 필요가 없음). (2)
`quantization` 필드는 추가하지 않는다(없는 기능의 placeholder를 만드는 건 YAGNI).

## 구현

1. **`role_map`**(`configuration_smolvla.py`) — `{"stabilizer":.., "actuator":..}` →
   `__post_init__`에서 `ard_default_actuator_arm = role_map["actuator"]`로 파생(단일
   소스 유지, 실제 분리/결합 로직은 여전히 `ard_default_actuator_arm` 하나만 읽음).
2. **`action_space`**(`"joint"`\|`"ee"`) — `"ee"`면 `lac_mirror_sign`을 강제로 `None`
   으로 덮어쓰고 경고 로그(관절 축 미러링 전제가 엔드이펙터 델타에는 안 맞음). config
   레이어 변경만으로 충분 — 모델 코드는 전혀 안 건드림(`lac_mirror_sign`을 읽는 곳은
   이미 `lac.py` 한 곳뿐이라 config에서 미리 고쳐두면 그걸로 끝).
3. **`pos_loss_type`**(`"l2"`\|`"l1"`\|`"smooth_l1"`) — `modeling_smolvla.py`의
   `F.mse_loss` 호출 한 곳을 3분기로 교체. L2는 목표 근처에서 기울기가 사라지는 문제가
   있음(RoCo 1회 우승팀 보고, arXiv 2603.15469).
4. **`per_timestep_norm`**(+`_a`/`_s`) — σ(t)=a+s·√t(t=실행 스텝 위치, 0..chunk_size-1).
   `ard.py`에 순수 함수 3개(`compute_timestep_norm_sigma`/`apply_timestep_norm`/
   `invert_timestep_norm`, 패딩 채널은 identity로 채움) 추가. 모델 wiring은 단 두 지점:
   `VLAFlowMatching.forward()`가 ground-truth actions를 받는 즉시 스케일하고,
   `_denoise_loop()`(모든 `sample_actions*` 4변형이 공유하는 **단 하나의 반환 지점**)가
   최종 x_t를 역스케일 — 중간(denoise 루프 전체, LAC 포함)은 전부 스케일된 공간에서
   동작하고 표준 정규화는 전혀 안 건드림. 꺼지면(기본) bit-for-bit 동일. LAC와 동시에
   켜면 ρ/ω 임계값이 스케일된 공간 기준이 되는 상호작용이 있어 RoCo 프리셋은
   `lac_mode="none"`으로 피함.
5. **chunk_size/n_action_steps** — 조사 결과 이미 config대로 정확히 동작해서 코드 변경
   없음. 테스트로 RoCo값(chunk 20/실행 5)에서 LAC 없이/있이 둘 다 정상 동작하는지만
   확인.
6. **`freeze_vision_encoder`** — 이미 존재(기본 True), 변경 없음. RoCo 프리셋에 명시적
   으로만 박아서 학습 범위가 프리셋만 봐도 드러나게 함.
7. **`scripts/fit_timestep_norm.py`**(신규) — 데이터셋의 위치별 정규화된 액션 표준편차를
   `std(t) ≈ a + s·√t` 최소제곱 피팅. HF Hub 차단으로 실행 못 해봄(미검증).
8. **`configs/presets/roco_board.yaml`**(신규) — `ard_arm_dim=7`/`ard_gripper_dim=1`
   (공개 스펙 기반 가정, 미확인), `role_map={stabilizer:right, actuator:left}`,
   `action_space=ee`, `pos_loss_type=l1`, `chunk_size=20`/`n_action_steps=5`,
   `per_timestep_norm=true`(a/s는 미피팅 항등값), `freeze_vision_encoder=true`,
   `lac_mode=none`. quantization 키 없음(위 참고). 값마다 한 줄 이유 주석.

## 테스트

`tests/test_configurable_ard.py`(17개, 전부 통과): 새 필드 기본값 no-op(bit-exact),
7+7 레이아웃 forward/backward, role_map 반전(+잘못된 값 거부), pos_loss_type 3종(수학적
성질 + 실제 torch 함수 dispatch 가로채기 확인 + 잘못된 값 거부), per_timestep_norm
(스칼라/채널별+패딩 왕복오차, on/off 동작, 추론 경로 finite, a<=0 거부), action_space=ee
→ mirror_sign 강제 비활성+경고(+joint면 유지, 잘못된 값 거부), RoCo 프리셋 로드 검증 +
작은 합성 모델 학습/추론 스모크 테스트.

**테스트 작성 중 발견한 버그(테스트 자체의 버그, 코드 버그 아님)**: 처음 "새 필드
no-op" 비교 테스트가 두 정책의 loss가 다르게 나와 실패했다 — 원인은 `batch`를 만드는
데 쓴 `torch.manual_seed`가 forward() 직전 재설정 시점과 안 맞아서(한쪽은 batch 생성
직후 RNG 상태를 이어서 forward, 다른 쪽은 seed를 다시 리셋하고 forward) 두 forward()
호출의 noise/time 샘플링이 서로 다른 지점에서 RNG를 소비한 것 — **모델 코드와는 무관한
테스트 작성 실수**였다(같은 정책을 여러 번 forward하면 완벽히 결정적임을 별도로 확인해
입증). batch 생성과 forward() 사이의 seed 리셋 위치를 맞춰서 고쳤다.

`pytest tests/`(130개 전부 통과 — 기존 113 + 신규 17). 기존 `test_ard*`/`test_lac*`/
`test_closed_loop*`/`test_judge*` 전부 회귀 없음.

## 문서

`docs/configurable_ard.md`(신규) — 조사 결과 표, 각 필드의 수식적 근거(arXiv
2603.15469/2606.27163 인용), RoCo 프리셋 값 표 + 선택 이유, quantization을 추가하지
않은 이유, "검증한 것/못 한 것"(README 관례). README.md에 새 섹션 + 링크 추가.

## 검증한 것 / 못 한 것

**검증함**: 위 테스트 전부(합성 모델, Hub 접근 없음). **못 함**(범위 밖/미확인): RoCo
공식 데이터셋 스키마(HF Hub 차단, 공개 스펙 기반 가정으로 대체), `per_timestep_norm_a`/
`_s` 실데이터 피팅(스크립트는 작성, 실행 못 함), 실제 SmolVLM2 가중치/실제 로봇·시뮬,
chunk_size=20/n_action_steps=5가 실제 RoCo 과제에 적절한 값인지, `per_timestep_norm`+
`lac_mode`를 동시에 켠 실제 효과(분석만 함, 실측 안 함).

## 다음에 할 일

- RoCo 데이터셋 접근이 가능한 환경에서: `scripts/fit_timestep_norm.py` 실행, 실제
  `ard_arm_dim`/`ard_gripper_dim`/`action_space` 가정 재확인, `scripts/train_ard.py`로
  작은 규모 실제 학습.
- `per_timestep_norm` + `lac_mode`를 함께 쓰고 싶다면 ρ/ω 임계값 재캘리브레이션 필요.

---

# RoCo 스키마 확인 반영 + 각도 래핑(ard_angle_dims) (branch: configurable-ard, 후속)

사용자가 RoCo 공식 데이터셋의 `meta/info.json`을 직접 열어서 action 14차원 순서를 확인해
전달했다: `left_ee_x/y/z/rx/ry/rz/gripper, right_ee_x/y/z/rx/ry/rz/gripper` — 팔별로
연속된 [ee pose 6 + 그리퍼 1] = 7채널 블록. 4가지를 확인/수정 요청받았다.

## 1. `ard_arm_dim`이 그리퍼 포함/제외 중 어느 쪽인지

코드 확인 결과(`configuration_smolvla.py:126` 주석 "팔 하나당 자유도 (관절 7 + 그리퍼
1)"): **`ard_arm_dim`은 그리퍼를 포함한 블록 전체 크기**이고, `ard_gripper_dim`은 그 안의
마지막 N채널(그리퍼로 간주해 smooth/traj 손실에서만 제외)이다. 따라서 RoCo(팔당 7=ee
pose 6+그리퍼 1)는 **이미 설정된 `ard_arm_dim=7`/`ard_gripper_dim=1`이 정확히 맞다** —
사용자가 예상한 "그리퍼 제외 의미면 arm_dim=6" 가설은 틀렸고, 변경 불필요. 대신 액션
차원 일치(`2*ard_arm_dim == 14`)를 테스트로 명시적으로 고정했다.

## 2. rx/ry/rz 각도 래핑(`ard_angle_dims`, 신규)

±π 경계 불연속(3.14 vs -3.14 — 거의 같은 각도인데 raw 차이는 ~2π) 문제를 막기 위해:
- `ard.py`: `wrap_to_pi`(순수 함수), `build_angle_channel_mask`(각 팔 블록 내 상대
  인덱스를 양팔 블록 둘 다에 적용, 패딩 채널은 제외), `apply_angle_aware_target`
  (`u_t - wrap_to_pi(u_t-v_t)`로 "각도-인식" v_t를 만들어 **기존 F.mse_loss/F.l1_loss/
  F.smooth_l1_loss 호출부를 바꾸지 않고** 그 자리에 넣는 트릭).
- `configuration_smolvla.py`: `ard_angle_dims: list[int] | None = None` 필드 + 검증
  (빈 리스트/중복/범위 밖 거부, `use_ard=True` 필요).
- `modeling_smolvla.py`: pos_loss 계산 직전에 `ard_angle_dims`가 있으면
  `apply_angle_aware_target` 호출, 없으면(기본) `v_t_for_loss = v_t`로 **함수 호출조차
  없음** — bit-exact 보장.
- **중요한 미검증 사항**: 이 wrap은 입력이 실제 라디안일 때만 정확하다. 하지만 이 지점의
  u_t/v_t는 이미 `NormalizerProcessorStep`의 표준 차원별(z-score) 정규화를 거친 값이라서
  (코드 확인: `policy.forward()` 호출 **전에** `preprocessor(batch)`가 적용됨,
  `scripts/train_ard.py:375`), 그 채널의 std가 1이 아니면 실제 경계는 `2π`가 아니라
  `2π/std`다. 모델에 std가 전달되지 않아 이 보정을 못 한다 — RoCo 프리셋의
  `ard_angle_dims=[3,4,5]`는 **근사치**로 명시했다.
- RoCo 프리셋에 `ard_angle_dims=[3, 4, 5]` 추가(위 확인된 스키마의 rx/ry/rz 인덱스).

## 3. 정규화가 차원별로 되는지(위치 m vs 회전 rad)

`normalize_processor.py` 코드 직접 확인: `(tensor - mean) / (std + eps)` — mean/std가
피처(채널)별 텐서라 위치/회전처럼 단위가 달라도 각 채널이 자기 자신의 통계로만
정규화된다. **문제없음, 수정 불필요** — 확인만 하고 docs에 기록.

## 4. LAC의 ρ/ω가 정규화된 값 기준으로 계산되는지

`lac.py` 코드 확인: `E_s`/`E_r`/`rho`/`omega`는 전부 `apply_lac_step()`에 들어오는
shared/residual 벡터에 `.norm(dim=-1)`/코사인 유사도를 직접 적용한다. 이 벡터들은
`_denoise_loop()`/`combine_by_role()`을 거쳐 온 것이라 **항상 v_t가 사는 공간**(표준
정규화 + `per_timestep_norm` 스케일이 적용된 공간)에서 계산된다 — `action_space`
("joint"/"ee")와 무관하게 일관된 동작이고, ee 전용 특이사항은 없다. **조사만 하고
수정은 보류**(RoCo 프리셋은 `lac_mode="none"`이라 당장 영향 없음).

## 테스트/문서

`tests/test_configurable_ard.py`에 10개 추가(총 24개): `wrap_to_pi`/
`build_angle_channel_mask`/`apply_angle_aware_target` 순수 함수, `ard_angle_dims` 검증
(빈 리스트/중복/범위 밖/use_ard=False 거부), 기본값(None)일 때 `apply_angle_aware_target`
호출 없음을 몽키패치로 직접 확인, 설정 시 호출됨 + finite 유지, RoCo 프리셋 로드에
`ard_angle_dims`/차원 일치 어서션 추가. `docs/configurable_ard.md`에 확인된 스키마·각도
래핑 설명·단위 미검증 사항 반영, README.md 요약 갱신.

`pytest tests/`(137개 전부 통과 — 기존 130 + 신규 7 테스트 함수 중 일부는 추가 assert만
포함돼 개수가 10개 늘었지만 신규 함수는 7개). 회귀 없음.

## 다음에 할 일

- `ard_angle_dims`의 `2π/std` 정확한 경계 처리가 필요해지면, 모델에 정규화 std를 전달하는
  경로(현재는 전혀 없음)를 추가해야 함 — 지금은 범위 밖.
- "대부분 왼팔만 사용"(role_map의 근거)과 10fps/시연 200개는 여전히 공개 설명 기반 가정 —
  스키마처럼 사용자가 직접 재확인하면 갱신 필요.

---

# ard_angle_dims 되돌림 + angle_unwrap_dims 전처리로 재설계 + 2가지 보고 (branch: configurable-ard, 후속)

사용자가 직전 `ard_angle_dims`(pos_loss에서 회전 채널을 `[-π,π)`로 감싸는 설계)를
되돌리라고 지시했다. 이유: (1) `u_t`/`v_t`는 액션이 아니라 flow-matching **속도장**
(`noise - actions`)이라 각도 주기성이 적용되지 않고, 속도 오차를 접으면 정상적인 큰
오차까지 왜곡된다. (2) `ard_angle_dims=[3,4,5]`가 왼팔 회전만 포함하고 오른팔[10,11,12]이
빠진 것처럼 보임. 대신 데이터 전처리 단계(학습 전 unwrap/추론 후 rewrap)로 다시
설계하라는 요청과, 회전 표현(오일러각/회전벡터) 근거 조사, 데이터 점검 스크립트, 그리고
이전부터 밀려 있던 두 가지 보고(기존 기능 브랜치 조사, freeze_vision_encoder/LoRA 현황)
요청.

## 1. `ard_angle_dims` 되돌림

`ard.py`의 `wrap_to_pi`/`build_angle_channel_mask`/`apply_angle_aware_target`,
`configuration_smolvla.py`의 `ard_angle_dims` 필드+검증, `modeling_smolvla.py`의
pos_loss 분기 앞 각도-인식 바꿔치기를 전부 제거했다. pos_loss는 다시 순수하게
`F.*_loss(u_t, v_t)`만 계산한다(기본 동작 bit-exact 재확인). RoCo 프리셋에서도
`ard_angle_dims: [3, 4, 5]` 제거.

**참고**: `build_angle_channel_mask`는 실제로는 왼팔[3,4,5]과 오른팔[10,11,12] 둘 다에
자동으로 마스크를 적용하도록 설계돼 있었다(설정값 자체가 "팔 블록 내 상대 인덱스"라는
의도였음) — 그래서 사용자가 말한 "오른팔이 빠짐" 문제는 실제 라우팅 버그는 아니었다.
다만 (1)번 이유(속도장에 각도 래핑을 거는 것 자체가 틀린 설계)가 더 근본적이라 되돌리는
게 맞다.

## 2. `angle_unwrap_dims` — 데이터 전처리로 재설계

- `processor_smolvla.py`(신규): `AngleUnwrapProcessorStep`(학습 전,
  `NormalizerProcessorStep`보다 먼저 — raw 액션의 회전 채널을 **액션 청크의 시간축을
  따라** `np.unwrap`과 동일한 원리로 연속으게 편다)과 `AngleRewrapProcessorStep`(추론 후,
  파이프라인 마지막 단계 — 로봇에 보내기 직전 `[-π,π)`로 다시 감싼다, Unwrap의 정확한
  역연산). 둘 다 `NormalizerProcessorStep`과 같은 패턴으로 배치에 액션이 없으면(실제
  추론 입력) 에러 없이 통과시킨다.
- **"에피소드 경계를 넘어 펴지 않는다"는 요구는 설계상 자동으로 만족된다** —
  `LeRobotDataset`의 `delta_timestamps` 윈도우(학습 배치의 액션 청크)는 항상 한
  에피소드 내부에서만 뽑히므로, "청크 안에서만" unwrap하는 것으로 충분하다(전체 에피소드를
  한 번에 펼 필요 없음 — 모델은 청크 내부의 상대적 연속성만 필요).
- `configuration_smolvla.py`: `angle_unwrap_dims: list[int] | None`(절대 채널 인덱스,
  `use_ard`와 무관하게 동작) 필드 + 검증. 기본 None이면 두 ProcessorStep이
  `make_smolvla_pre_post_processors()`의 파이프라인에 들어가지도 않아 bit-exact.
- RoCo 프리셋은 이 옵션을 **꺼둔 채로** 둔다(아래 3번 참고).

## 3. 오일러각 vs 회전벡터(axis-angle) 조사

이 레포/RoCo 스키마 자체에는 직접적인 증거가 없다. 간접 근거 두 가지를 찾았다:
1. 이 레포가 vendor로 들고 있는 LeRobot 자체가 `env_processor.py`의
   `LiberoProcessorStep`에서 엔드이펙터 회전을 쿼터니언 → **axis-angle(회전벡터)**로
   변환해서 쓰는 선례가 있다(`utils/rotation.py`에 `as_rotvec`/`from_rotvec`도 있음).
2. 산업용 로봇(Universal Robots 등)의 통상적인 TCP pose 표기 "x,y,z,rx,ry,rz"에서
   rx,ry,rz는 관례적으로 회전벡터를 가리키는 경우가 많다(오일러각이면 보통
   roll/pitch/yaw로 부름).

**결론: 확정적이지 않지만 회전벡터일 가능성을 시사하는 간접 근거가 있다.** 회전벡터면
`angle_unwrap_dims`(축별 unwrap)는 수학적으로 부적절하다(세 채널이 하나의 회전을 이루는
벡터라 축별 주기성 가정이 성립하지 않음) — 그래서 RoCo 프리셋은 이 옵션을 켜지 않았다.
`docs/configurable_ard.md`에 이 근거와 불확실성을 명시했다.

## 4. `scripts/check_rotation_range.py`(신규)

에피소드별 회전 채널 min/max, ±π 경계 도약 횟수를 출력. 채널 수가 3의 배수면(기본
[3,4,5,10,11,12]) 3개씩 묶어 "삼중값" 노름(||[rx,ry,rz]||) 통계도 출력 — 회전벡터는
보통 노름이 회전각 자체라 `≤ π` 안에 들어오는 경향이 있어, 오일러각/회전벡터 판별에
참고용 힌트가 된다(합성 데이터로 두 경우를 시뮬레이션해서 이 힌트가 실제로 구분되는지
확인함: 회전벡터형 0% 초과 vs 오일러형 55% 초과). HF Hub 차단으로 실제 데이터로는
실행 못 해봄.

## 보고 1: 기존 기능 브랜치 조사 (합치지 않음, 조사만)

`git branch -a` + `git log configurable-ard..<branch>`로 전체 브랜치(`claude/smolvla-
research-setup-x29mi7`, `closed-loop-structure`, `lac-inference`, `lac-closed-loop`,
`main`)를 확인 — **전부 0개**(configurable-ard에 없는 커밋 없음), 즉 quantization/
GradNorm/token_pruning/gradient_checkpointing/profile_memory.py/layer_importance.py
전부 이미 configurable-ard에 머지돼 있다(quantization은 애초에 어디에도 없음). 합칠
별도 브랜치가 없었다. 충돌 가능성도 전부 낮음(GradNorm만 per_timestep_norm과 약한
상호작용 있음 — x0_hat 스케일이 바뀌면 GradNorm이 균형 잡는 그래디언트 norm 크기도
같이 바뀔 수 있음, 버그는 아니고 참고 사항). 상세 표는 `docs/configurable_ard.md` 참고.

## 보고 2: `freeze_vision_encoder`/액션 expert/VLM LoRA 현황 (요구사항 6, 미수정)

- `freeze_vision_encoder`(기본 True) 존재하지만, 다른 기본값 `train_expert_only=True`가
  이미 VLM 전체(비전 인코더 포함)를 얼려서 기본 조합에서는 중복.
- "액션 expert 전체 학습"은 기본적으로 이미 그렇다(`lm_expert`는 PEFT 없이는 얼지 않음).
- **"VLM은 LoRA"는 구현돼 있지 않다** — `_get_default_peft_targets()`가 잡는 "lm_expert"는
  VLM이 아니라 **액션 expert**(완전히 별개 모델 인스턴스)다. 실제 VLM(`self.vlm`)은 PEFT
  타겟에 전혀 없고, 게다가 PEFT 진입점(`wrap_with_peft()`) 자체가 `scripts/train_ard.py`
  에서 호출되지 않는다(CLI 플래그 없음). VLM에 LoRA를 적용하려면 타겟 정규식 추가 +
  train_ard.py에 PEFT CLI 경로 추가가 둘 다 필요 — 이번엔 보고만, 수정 안 함.

## 테스트/문서

`tests/test_configurable_ard.py`: 옛 `ard_angle_dims` 테스트 7개 제거, `angle_unwrap_dims`
테스트 7개 추가(unwrap 연속화, rewrap 역연산+round-trip, 액션 없음/옵션 꺼짐 통과,
`make_smolvla_pre_post_processors()`가 실제로 만드는 파이프라인에 설정 시에만 두
ProcessorStep이 들어가는지 — 토크나이저 로드만 몽키패치, 나머지는 실제 함수 호출로
직접 확인). RoCo 프리셋 테스트의 `ard_angle_dims` 어서션을 `angle_unwrap_dims is None`
으로 교체. `docs/configurable_ard.md`에 되돌림 이유, 새 설계, 오일러/회전벡터 근거, 두
보고 섹션 추가. README.md 요약 갱신.

`pytest tests/`(136개 전부 통과). 회귀 없음.

## 다음에 할 일

- Hub 접근이 되는 환경에서 `scripts/check_rotation_range.py` 실행 → rx/ry/rz가 어느
  표현인지 확정 → 회전벡터면 `angle_unwrap_dims` 자체를 쓰지 말고 SO(3)-인식 처리를
  고려, 오일러각이면 그때 `angle_unwrap_dims` 켜고 A/B 비교.
- "VLM은 LoRA"를 실제로 구현하려면: `_get_default_peft_targets()`에 `model.vlm_with_
  expert.vlm.*` 타겟 추가 + `train_ard.py`에 `--use-peft` 경로 추가(이번 범위 밖, 보고만).

---

# RoCo 전체 통계(200 에피소드/121,454프레임) 반영: state_select + action_min_std (branch: configurable-ard, 후속)

사용자가 RoCo 공식 시뮬 데이터셋 **전체**(200 에피소드, 121,454프레임)를 집계한 결과를
전달했다: state 44차원 = EE pos+quat 14(0-13) + 관절 위치 14(14-27) + 관절 속도 14(28-41)
+ 그리퍼 2(42-43). action 14차원은 EE **절대** 목표(Euler XYZ, rad) — 각도는 이미 연속
(±π 경계 점프 0건)이고 값 자체는 ±π 밖으로도 나간다(왼팔 rx 최대 4.10, rz 최대 7.00).
오른팔 action은 사실상 상수(pos_x 범위 1.6e-5m 등). 양자화는 건드리지 말라는 명시적 지시.

## 1. 확인(가정 없이 코드로)

`max_state_dim`/`max_action_dim` 기본값 **둘 다 32**(`configuration_smolvla.py`). `pad_vector
(vector, new_dim)`는 입력이 `new_dim`보다 **크면 자르지 않고 `RuntimeError`**를 던짐 — 직접
재현 확인(`pad_vector(torch.randn(2,44), 32)` → `RuntimeError: ... non-singleton dimension 1`).
RoCo raw state(44) > 기본 `max_state_dim`(32)이라 아무 조치 없이 쓰면 첫 배치에서 크래시.

## 2. `state_select`(신규)

`configuration_smolvla.py`에 `state_select: list[int] | None` 필드 + 검증(빈 리스트/중복/
음수/`len > max_state_dim` 거부는 `__post_init__`, 실제 `observation.state` 차원을 넘는
인덱스 거부는 `validate_features()`). `modeling_smolvla.py`의 `prepare_state()`가
`ard_use_joint_torque` concat/`pad_vector`보다 먼저 `state.index_select(-1, idx)`를 적용 —
학습(`forward()`)/추론(`predict_action_chunk` 등) 둘 다 이 메서드를 공유하므로 자동으로
동일 적용된다. `None`(기본)이면 분기 자체가 없어 기존과 완전히 동일.

`state_proj = nn.Linear(max_state_dim, hidden_size)`로 입력 폭이 항상 `max_state_dim`
고정이라(기존 `ard_use_joint_torque`와 같은 패턴), 30채널(관절 속도 28-41 제외) <= 32이면
사전학습 가중치를 그대로 쓸 수 있음을 확인. RoCo 프리셋: `state_select = list(range(28))
+ [42, 43]`(30차원).

## 3. `action_min_std`(신규)

`configuration_smolvla.py`에 `action_min_std: float | list[float] | None` 필드 + 검증(0
이하 거부, 리스트 길이는 `validate_features()`에서 실제 액션 차원과 대조). 실제 적용은
`processor_smolvla.py`의 `_apply_action_min_std(dataset_stats, min_std)` — `dataset_stats
[ACTION]["std"]`를 `torch.maximum(std, min_std)`로 교체한 **새** dict를 한 번만 만들어서
`NormalizerProcessorStep`과 `UnnormalizerProcessorStep` 둘 다에 **같은 결과**를 넘긴다(원본
`dataset_stats`는 안 바꿈) — 이 구조 자체가 "정규화·역정규화가 항상 같은 std를 쓴다"를
보장한다. `None`이면 원본을 그대로 양쪽에 넘기므로 bit-exact.

RoCo 프리셋: 위치 0.01m/회전·그리퍼 0.05rad(둘 다 임시값), 왼팔(0-6)/오른팔(7-13) 구분 없이
14채널 전체에 같은 패턴을 반복 — 테스트로 "극단적으로 std=1e-6인 14채널 전체가 일관되게
보호되는지"(오른팔만 아니라) 직접 확인.

## 4. 각도: 실측으로 더 강한 근거 확보

이전 세션에서 "오일러각 vs 회전벡터 미확인"을 근거로 `angle_unwrap_dims`를 꺼뒀는데, 이번
전체 통계로 **그 질문과 무관한 더 직접적인 이유**가 나왔다: 각도 점프가 0건이면서 값 자체는
±π 밖(rz 최대 7.00)이라는 건 "데이터가 이미 unwrap된 채로 기록돼 있다"는 뜻이다. 그래서
`AngleUnwrapProcessorStep`은 펼 게 없어 거의 항등 변환이고, `AngleRewrapProcessorStep`은
오히려 데모에 없던 불연속을 만든다(예: 7.00을 보낼 자리에 0.717을 보냄). 오일러/회전벡터
여부와 무관하게 결론 확정 — RoCo 프리셋은 `angle_unwrap_dims`를 아예 명시하지 않는다.

## 테스트/문서

`tests/test_configurable_ard.py`에 11개 추가: `state_select` 검증 거부(빈 리스트/중복/음수/
개수초과/실제 차원 밖), `prepare_state()`가 선택한 채널을 정확한 순서로 가져오고 나머지는
0 패딩하는지, `None` 기본값이 기존과 동일한지, `action_min_std` 검증 거부(0/음수/길이
불일치), 정규화→역정규화 왕복이 min_std 켜도 원래 값으로 돌아오는지, std=1e-6인 14채널
**전체**가 일관되게 보호되는지(왼팔/오른팔 구분 없이), 이미 큰 std는 안 낮추는지, RoCo
프리셋이 `angle_unwrap_dims`를 아예 설정 안 하는지 + 실제 파이프라인에 Angle 스텝이 없는지.
기존 `test_roco_preset_tiny_model_trains_and_infers`/`test_roco_preset_loads_into_valid_config`
는 raw state 44차원(2*arm_dim 심메트릭 가정이 더 안 맞음)에 맞게 `input_features`/
`max_state_dim`을 명시적으로 덮어쓰도록 수정.

`docs/configurable_ard.md`: "RoCo 시뮬 실측" 절 추가(각도 연속성 근거), `state_select`/
`action_min_std` 설명 절 추가(`pad_vector` 크래시 재현 코드 포함), action/state 스키마를
전체 통계로 갱신. README.md 요약 갱신. `configs/presets/roco_board.yaml` 상단 주석을
전체 통계로 갱신 + `state_select`/`action_min_std` 추가.

`pytest tests/`(147개 전부 통과 — 기존 136 + 신규 11). 양자화는 건드리지 않음(그대로
존재하지 않는 상태 유지).

## 검증한 것 / 못 한 것

**검증함**: 위 테스트 전부(합성 모델, Hub 접근 없음) + `pad_vector` 크래시 실제 재현.
**못 함**(범위 밖/미확인): `action_min_std` 값(0.01/0.05)이 실제 RoCo 통계 기준으로
적절한지 재튜닝 안 함, rx/ry/rz 오일러/회전벡터 여부는 여전히 미확인(다만 영향 없음 —
위 4번 참고), `state_select`로 관절 속도를 빼는 게 실제 학습 성능에 미치는 영향은 실측
안 함.

## 다음에 할 일

- `action_min_std` 값을 실제 오른팔 채널 표준편차(pos_x 1.6e-5m 등)에 맞춰 재튜닝할지
  검토 — 지금 값(0.01/0.05)은 임시.
- 관절 속도(28-41)를 뺀 30차원 state가 실제로 충분한지(관절 속도 정보가 성능에 중요하면
  다시 넣고 `max_state_dim`을 44 이상으로 올려야 함) GPU 실험으로 확인 필요.

# `state_min_std` 추가: STATE도 action처럼 std 하한 (branch: configurable-ard, 후속)

`action_min_std`와 같은 구조를 STATE 정규화에도 적용해달라는 요청. 핵심 질문은 정규화와
`state_select`의 순서였다.

## 1. 확인: STATE 정규화 방식과 순서

`SmolVLAConfig.normalization_mapping` 기본값(`{"VISUAL": IDENTITY, "STATE": MEAN_STD,
"ACTION": MEAN_STD}`)을 확인해 STATE도 ACTION과 동일한 MEAN_STD임을 확인. 순서는
`scripts/train_ard.py`의 `batch = preprocessor(batch); loss, loss_dict = policy.forward(batch)`
와 `make_smolvla_pre_post_processors()`가 raw `config.input_features`로 `Normalizer`를
만든다는 점, `state_select`는 `SmolVLAPolicy.prepare_state()`(즉 `policy.forward()` 안쪽)
에서만 적용된다는 점을 코드로 추적해서, **정규화가 `state_select`보다 먼저, raw
observation.state 전체를 대상으로 일어난다**는 걸 확정했다 — `state_select`는 이미
정규화된 채널을 고르는 것이다.

## 2. `state_min_std`(신규)

이 순서 때문에, 사용자가 선택 후(post-`state_select`) 30채널 순서로 값을 주면 내부적으로
raw 44채널 위치에 scatter해야 한다. `configuration_smolvla.py`에 `state_min_std: float |
list[float] | None` 필드 + 검증(0 이하/빈 리스트 거부는 `__post_init__`, `state_select`가
있으면 그 길이와 대조, 없으면 `validate_features()`에서 실제 `observation.state` 차원과
대조) 추가. `processor_smolvla.py`에 `_apply_state_min_std(dataset_stats, min_std,
state_select)` 추가 — `state_select`가 있으면 그걸 인덱스 맵으로 써서 raw std 배열의 해당
위치만 `max(std, min_std)`로 바꾸고, 없으면 raw 전체에 직접 적용. `Normalizer`에만 넘기고
`Unnormalizer`에는 안 넘긴다 — `UnnormalizerProcessorStep(features=config.output_features,
...)`가 ACTION만 보고 STATE를 전혀 안 읽으므로 STATE는 역정규화되는 적이 없다(action과
달리 round-trip 보장이 필요 없는 이유).

**구현 중 발견한 위험(선제 수정)**: `torch.as_tensor(std, dtype=torch.float32)`는 이미
같은 dtype인 텐서를 **복사 없이** 그대로 돌려준다 — 그 뒤 `std[idx] = ...`로 in-place
수정하면 호출자의 원본 `dataset_stats` 텐서가 같이 바뀐다. `.clone()`을 명시적으로 추가해
원본이 안 바뀌는 걸 직접 재검증했다(action_min_std는 `torch.maximum()`이 새 텐서를
반환해서 이 문제가 없었음 — 다른 구현 패턴이라 별도로 확인이 필요했던 지점).

RoCo 프리셋(`configs/presets/roco_board.yaml`): `state_select` 적용 후 30채널 순서로
EE 위치 0.01/쿼터니언 0.05/관절 위치 0.05/그리퍼 0.05, 왼팔/오른팔 구분 없이 반복(모두
임시값). EE 위치+쿼터니언 14채널(0-13) 안의 "팔별 pos+quat" 서브순서는 action 14차원의
확인된 구조를 가져온 **미확인 가정**이라고 YAML 주석과 docs에 명시.

## 테스트/문서

`tests/test_configurable_ard.py`에 8개 추가: 잘못된 값 거부(빈 리스트/0/음수/길이 불일치
2종), `state_select`를 통한 scatter 정확성(선택 안 된 위치는 그대로, 원본 텐서 불변
직접 검증), `state_select` 없을 때 직접 적용, RoCo 실제 30채널 패턴으로 양팔 전체가
일관되게 보호되는지, `min_std=None`이면 객체 동일성까지 bit-exact, RoCo 프리셋의
`state_min_std` 길이가 `state_select`와 일치하는지.

`docs/configurable_ard.md`: `### state_min_std` 절 추가(정규화 순서 확인 내용 포함),
필드 테이블/프리셋 요약 테이블/"검증한 것"/"못 한 것" 절 갱신(서브순서 미확인 가정을
"못 한 것"에도 명시). README.md 요약 갱신.

`pytest tests/`(**155개 전부 통과** — 기존 147 + 신규 8). 양자화는 건드리지 않음.

## 검증한 것 / 못 한 것

**검증함**: STATE가 MEAN_STD이고 정규화가 `state_select`보다 먼저라는 것(코드 추적),
scatter 정확성과 원본 불변(위 테스트), `min_std=None` bit-exact, RoCo 프리셋 길이 일치.
**못 함**(범위 밖/미확인): `state_min_std` 값(0.01/0.05)과 EE 위치+쿼터니언 14채널 안의
팔별 서브순서 둘 다 미확인 가정 — 실제 RoCo state 스키마를 채널 단위로 재확인하기 전까지
남아 있는 위험(docs의 "못 함" 절 참고).

## 다음에 할 일

- `state_min_std`/`action_min_std` 값 전체를 실제 RoCo 통계로 재튜닝.
- EE 위치+쿼터니언 14채널(0-13)의 정확한 서브순서(왼팔/오른팔 각각 pos→quat인지, 아니면
  타입별로 묶여 있는지)를 실제 데이터셋 메타데이터나 사용자 재확인으로 검증.
