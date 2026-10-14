# 폐루프(closed-loop) 구조: Judge/Selector + 결정 로직 + 러너

[← README로 돌아가기](../README.md)

SmolVLA(+ARD)를 "한 번 추론하고 끝"이 아니라, 매 서브태스크마다 성공/실패를 스스로 판정하고
재시도·복구·정지를 결정하는 폐루프 시스템으로 확장한 구조입니다. **이 문서가 다루는 범위는
구조 구현뿐입니다** — 실제 Judge/Selector 학습 데이터 생성, 학습, 임계값/온도 캘리브레이션
실측은 전부 범위 밖이며, 아래 "검증한 것 / 못 한 것"에 명확히 표시했습니다.

## 구조도

```
[camera image] [final goal] [completed-steps list]
          └──────────┼──────────┘
              SmolVLM2 backbone (prefix, 한 번만 계산)
      ┌──────────────┼───────────────┐
  Judge head     Selector head     Action expert
 (성공 여부)      (다음 서브태스크)    (액션 청크)
      └──────┬───────┘                 │
          결정 로직(decision_logic) ───→ ARD head (Stabilizer/Actuator)
 (성공→다음 단계 / 실패→재시도·복구·정지)   │
                                     로봇 실행 → 새 관측치 → 루프
```

하나의 SmolVLM2 prefix(이미지+언어+state) forward를 Judge/Selector/Action expert 세 갈래가
공유합니다 — Bridge Attention이 이미 쓰고 있던 `collect_layer_indices`/
`last_collected_prefix_layers` side-channel을 그대로 재사용해서, Judge가 중간 레이어를
보더라도 추가 forward pass가 생기지 않습니다(`judge_feature_layer=-1`이면 `sample_actions()`가
예전부터 버리던 prefix 마지막 레이어 반환값을 그냥 재사용합니다).

## 각 블록의 역할

- **Judge** (`policies/smolvla/judge.py`의 `Judge`) — 비생성(non-generative) 분류 head.
  prefix hidden state를 `prefix_pad_masks` 기준 masked mean pooling + `LayerNorm`한 벡터
  위에, 서로 완전히 분리된(trunk 공유 없음) 작은 MLP head들을 둡니다: `success_logit`(성공
  확률, 필수), `failure_logits`(실패 유형 분류, 선택), `progress`(0~1 진행도 회귀, 선택).
  `selector_logits`도 같은 pooled feature에서 나오지만 역할상 Selector입니다. 추론 확률은
  `sigmoid(logit / temperature)` — `temperature`는 학습되지 않는 buffer(기본 1.0, 캘리브레이션
  fitting은 범위 밖, 슬롯만 있음).
- **Selector** — 같은 `Judge` 모듈의 `selector_logits` head. 미리 정의된 옵션(서브태스크
  ID/복구 동작 ID) 중 하나를 고르며, 결정 로직이 주는 `allowed_mask`로 현재 상태에서 허용되지
  않는 옵션은 `-inf`로 마스킹됩니다.
- **결정 로직** (`policies/smolvla/decision_logic.py`의 `DecisionLogic`) — torch 의존성이
  전혀 없는 순수 Python 규칙 레이어. task graph(완료 단계/현재 노드/재시도·재관측 횟수)를
  관리하고, Judge/Selector가 뭘 내놓든 **항상** 안전 규칙(허용되지 않은 전이 거부 → rule
  table로 대체, 재시도/재관측 한도 초과 → STOP)을 강제합니다. Selector가 꺼져 있으면
  (`use_selector=False`) rule table(실패 유형 → 복구 동작 매핑)만으로 똑같은 자리에서 똑같은
  모양의 결정을 만들어내므로, Selector의 "기본값" 역할도 합니다.
- **Action expert** — 기존 SmolVLA(+ARD) 그대로. Judge/Selector 때문에 구조나 손실이 바뀌지
  않습니다.
- **폐루프 러너** (`policies/smolvla/closed_loop.py`의 `run_closed_loop`) — 환경을 duck
  typing 프로토콜(`observe`/`execute`/`subtask_finished`/`settle`/`safe_stop`, 선택적
  `run_recovery`)로만 받는 최소 루프. 매 iteration마다 `predict_action_chunk_with_decisions`를
  **정확히 한 번** 호출해서 "지금 실행할 액션 청크"와 "지금 관측치에 대한 Judge/Selector
  판정"을 동시에 얻지만, **판정에 실제로 쓰는지는 `need_judgment` 플래그로 따로 관리**합니다 —
  아래 "판정 시점과 prefix forward 횟수" 참고.

### 판정 시점과 prefix forward 횟수

서브태스크가 아직 끝나지 않았으면(`need_judgment=False`) 이번 forward의 `decisions`는 판정에
쓰지 않고 `actions`로 다음 청크를 그냥 실행합니다 — 여러 청크짜리 서브태스크를 청크 경계마다
실패로 오판하지 않습니다. 서브태스크가 끝나면(`env.subtask_finished()` 또는
`max_subtask_steps` 초과) `env.settle()`을 부르고 `need_judgment=True`로 표시합니다. **다음**
iteration은 새로 `env.observe()`(=방금 끝난 서브태스크의 결과를 반영한 관측)로 forward를
돌리고, 그 `decisions`로 `decide()`를 부릅니다 — 즉 판정은 항상 **실행 후** 관측을 본다
(이전 버전은 실행 *전* 관측으로 판정해서 한 박자 늦었던 버그가 있었습니다).

매 iteration은 여전히 forward를 정확히 한 번만 부르므로(로그의 `forward_calls`가 매
iteration 1), 결정이 "경계"를 넘는 데 forward가 몇 번 쓰이는지는 iteration 수로 환산됩니다:
- 서브태스크 진행 중(판정 없음, 청크 경계): 청크당 1회.
- `RETRY`(같은 지시문으로 재시도): 이번 forward의 `actions`를 그대로 재사용 — 경계당 1회
  (판정한 iteration과 재시도를 실행하는 iteration이 같음, 추가 forward 없음).
- `ADVANCE`(다음 노드) 또는 `RECOVER`(복구 실행): 지시문/환경이 바뀌므로 다음 iteration에서
  다시 관측+forward — 경계당 2회(판정 iteration + 재forward iteration).
- `RECHECK`: 행동 실행 없이 `env.settle()` 후 다시 관측+forward+판정 — 매 RECHECK마다 1회씩
  쌓임.

`RECOVER`는 `env.run_recovery(recovery_id)`(선택 메서드)를 실제로 호출합니다. 환경이
`run_recovery`를 구현하지 않았으면 경고를 내고 `RETRY`와 똑같이 처리하며(추가 forward
없음), `decision_history`의 해당 항목에 `recovery_unsupported=True`를 남깁니다 — 실제 복구
동작 구현은 환경 쪽 스크립트의 몫으로 남겨둔 것입니다.

액션 실행 직전에는 `postprocessor`(있으면)를 적용해서 역정규화합니다 — `pre_chunk` 훅은
역정규화 **전** 값을 받습니다(모델 공간 그대로).

## 설정 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `use_judge` | `False` | Judge head(성공 판정) 활성화 |
| `judge_hidden_dim` | `256` | Judge/Selector MLP head들의 은닉 차원 |
| `judge_feature_layer` | `-1` | `-1`=prefix 마지막 레이어(post-norm, 추가 연산 없음). 그 외=Bridge Attention과 같은 `collect_layer_indices` 경로로 뽑는 중간 레이어(pre-norm) |
| `judge_num_failure_types` | `0` | `>0`이면 실패 유형 분류 head 추가 (`use_judge=True` 필요) |
| `judge_use_progress` | `False` | 진행도(0~1) 회귀 head 추가 (`use_judge=True` 필요) |
| `judge_detach_features` | `True` | Judge/Selector 손실이 백본/action expert로 역전파되지 않게 pooled feature를 detach |
| `use_selector` | `False` | Selector head(다음 서브태스크/복구 동작 선택) 활성화 |
| `selector_num_options` | `0` | Selector가 고를 수 있는 옵션 수 (`use_selector=True`면 1 이상 필수) |
| `judge_loss_weight` | `1.0` | `judge_success`/`judge_failure_type`/`judge_progress` 손실 가중치 |
| `selector_loss_weight` | `1.0` | `selector_target` 손실 가중치 |

전부 기본값이 꺼짐(off)이고, **전부 off면 기존 SmolVLA(+ARD)와 bit-for-bit 동일합니다**
(`tests/test_ard_integration.py::test_judge_flags_off_matches_predict_action_chunk`,
`test_all_new_flags_off_matches_pre_change_baseline`로 확인).

## 학습 배치 키 (전부 선택적, 없으면 해당 손실 0 — `ard_force_target`과 같은 패턴)

| 배치 키 | 모양 | 손실 |
|---|---|---|
| `judge_success` | `(batch,)`, 0/1 | BCE |
| `judge_failure_type` | `(batch,)`, int, `-100`은 무시 | CE |
| `judge_progress` | `(batch,)`, 0~1 | MSE |
| `selector_target` | `(batch,)`, int | CE (`selector_mask`가 있으면 마스킹 후) |
| `selector_mask` | `(batch, N)` 또는 `(N,)`, bool | (selector_target과 함께 사용) |

## 새/수정 파일

- `policies/smolvla/judge.py` (신규) — `Judge` nn.Module.
- `policies/smolvla/decision_logic.py` (신규, torch 의존성 없음) — `DecisionLogic`,
  `EpisodeState`, `Decision`/`DecisionKind`.
- `policies/smolvla/closed_loop.py` (신규) — `run_closed_loop`, `ClosedLoopEnv` 프로토콜
  (+ 선택적 `run_recovery`), `postprocessor`/`max_subtask_steps` 인자.
- `configuration_smolvla.py` — 위 플래그 전부 추가.
- `modeling_smolvla.py` — `VLAFlowMatching.judge`, `sample_actions_with_decisions()`(추론),
  학습 `forward()`의 prefix forward에 Judge 연결, `SmolVLAPolicy.predict_action_chunk_with_decisions()`/
  `cancel_pending_actions()`/`_compute_judge_losses()` 추가. `sample_actions()`의 denoise
  루프는 `_denoise_loop()`로 추출(순수 리팩터링, 수치 변화 없음 — bit-exact 테스트로 확인).
- `scripts/profile_memory.py`/`compare_smolvla_ard.py`/`compare_bridge_attention.py` — LoRA
  적용 후 Judge/Selector 파라미터도 다시 학습 가능하게 푸는 블록 추가(아직 이 스크립트들에
  Judge/Selector CLI 플래그는 없어서 지금은 항상 no-op).

## 검증한 것 / 못 한 것

**검증함** (`tests/test_judge.py`, `tests/test_ard_integration.py`,
`tests/test_decision_logic.py`, `tests/test_closed_loop.py` — 전부 `tests/conftest.py`의
합성 SmolVLM fixture로 Hub 접근 없이 오프라인 CPU에서 돈다):
- 플래그 전부 off → `predict_action_chunk_with_decisions`/학습 forward가 기존과 bit-for-bit
  동일.
- `predict_action_chunk_with_decisions` 내부에서 prefix forward(`fill_kv_cache=True`)가
  정확히 한 번만 호출됨(래핑해서 호출 횟수 직접 확인).
- Judge 출력 shape/확률 범위, masked mean pooling이 패딩을 무시함, temperature 스케일링,
  Selector `allowed_mask`가 비허용 옵션 확률을 0으로 만듦.
- 학습 손실: 배치 키 없으면 0(로그 키 자체가 안 생김), 있으면 유한한 값, `judge_detach_features=True`면
  실제 backbone forward를 통과한 judge_out으로도 역전파가 백본/action expert에 전혀 닿지 않음.
- 결정 로직: 3개 임계값 구간, 재시도/재관측 한도 초과 → STOP, 허용되지 않은 Selector 선택
  거부 후 rule table로 대체, Selector 없이 rule table만으로도 동작.
- 폐루프 러너: 가짜 환경 + tiny policy로 "실패 → `cancel_pending_actions()`로 청크 버림 →
  재시도 → 성공 → 다음 노드로 → 종료" 한 번의 전체 패스. 판정이 실행 *후* 관측을 쓰는지
  (obs_id로 직접 확인), 서브태스크가 청크 여러 개에 걸칠 때 중간 청크 뒤에는 판정/settle이
  안 일어나는지, `RECOVER`가 `env.run_recovery()`를 실제로 부르는지(없으면 경고+RETRY로
  대체하고 `recovery_unsupported=True`를 남기는지), `RECHECK`이 행동 실행 없이 재관측만
  하는지, `postprocessor`가 실행 직전에 적용되는지(`pre_chunk`는 그 전 값을 받는지),
  `max_subtask_steps` 초과 시 판정으로 넘어가는지, `forward_calls`/`judged` 로그가 iteration별
  실제 횟수와 일치하는지까지 전부 확인.
- `cancel_pending_actions()`가 `select_action()`이 채운 `_queues[ACTION]`을 실제로 비움.

**못 함 (이번 작업 범위 밖, 구조만 있고 실측 없음)**:
- `tau_success`/`tau_fail`(결정 로직의 성공/실패/재관측 분리 임계값)과 `temperature`(Judge
  확률 캘리브레이션)는 전부 placeholder 기본값입니다 — 실제 분리 지점은 검증 데이터셋에서
  Judge 캘리브레이션 곡선을 보고 정해야 합니다.
- Judge/Selector 학습 데이터(성공/실패 라벨, 실패 유형, 진행도, 복구 동작 타겟) 생성 자체가
  범위 밖이라, 실제 가중치로 Judge/Selector를 학습·평가해본 적이 없습니다.
- `pre_chunk`(사전 청크 검증)/`per_step`(상시 Stabilizer 모니터링) 콜백은 호출 지점만 있고
  실제 로직은 구현하지 않았습니다.
- `closed_loop.py`의 러너는 액션 청크를 큐에 쌓지 않고 한 번에 다 실행합니다 — 실시간
  스트리밍 실행(select_action류의 큐 기반, 또는 RTC와의 조합)으로 바꾸는 건 범위 밖입니다.
- `RECOVER`의 실제 복구 동작(`env.run_recovery()`)은 환경 쪽에서 구현해야 합니다 — 이 러너는
  호출만 하고, 복구 동작 자체(로봇을 어떻게 움직여 재정렬/재파지할지)는 만들지 않았습니다.
- `max_subtask_steps`는 task graph 노드별이 아니라 러너 레벨 단일 값입니다(YAGNI — 노드별
  예산이 실제로 필요해지면 그때 `TaskGraphNode`에 필드를 추가하면 됩니다).
- 실제 로봇/시뮬레이터 환경에 `ClosedLoopEnv` 프로토콜을 구현해서 end-to-end로 돌려본 적이
  없습니다(지금까지는 전부 가짜 환경).
- 실제 SmolVLM2 가중치(Hub 접근)로 Judge의 `judge_feature_layer`별 특징 품질을 비교해본 적이
  없습니다 — pre-norm/post-norm 불일치를 `LayerNorm`으로 흡수하는 설계가 실제로 충분한지는
  미검증입니다.

```bash
pytest tests/test_judge.py tests/test_ard_integration.py tests/test_decision_logic.py tests/test_closed_loop.py
```
