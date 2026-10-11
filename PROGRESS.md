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
- [ ] 2. Judge/Selector 헤드 + 단일 prefix 추론 API
- [ ] 3. 학습 손실 연결
- [ ] 4. decision_logic.py
- [ ] 5. closed_loop.py 러너
- [ ] 6. 테스트 + 문서 + 최종 보고

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

## 다음에 할 일

2단계(Judge/Selector 헤드 + `predict_action_chunk_with_decisions` + `cancel_pending_actions`)
진행 — `configuration_smolvla.py` 플래그 추가 → `policies/smolvla/judge.py` 신규 → 위 설계로
`modeling_smolvla.py` 연결 → pytest 전체 통과 → 커밋/푸시 → 이 파일 갱신.
