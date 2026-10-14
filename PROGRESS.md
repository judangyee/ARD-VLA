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
