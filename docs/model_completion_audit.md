# ARD-VLA 모델 완성도 감사 (학습 제외)

작성일: 2026-11-05. 대상 브랜치: `configurable-ard`(이 문서를 커밋하는 시점의 HEAD — 아래 3절에서
확인하듯 `main`/`closed-loop-structure`/`lac-inference`/`lac-closed-loop`를 전부 포함하는 가장 앞선
지점). 범위: **학습(실제 데이터로 돌린 학습 실행)은 제외** — 코드/설정/테스트 상태만 다룬다.

규칙: 모든 서술에 `파일:줄번호`를 붙인다. 코드를 실행해서 직접 확인한 것과, grep/읽기로만 확인한
것, 확인하지 못한 것을 구분해서 표시한다. **파일/함수 이름만 보고 동작을 추측하지 않는다** —
추측이 필요한 지점은 전부 "미확인"으로 명시한다.

---

## 1. 구현된 그대로의 아키텍처 (forward 경로)

### 1.1 `suffix_out`은 어디서 나오는가

`suffix_out`은 `VLAFlowMatching.vlm_with_expert.forward()`가 반환하는 `(prefix_out, suffix_out)`
튜플의 두 번째 원소다 — 학습 경로(`forward()`)는
`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:1211-1219`:

```python
(prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
    attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
    inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, fill_kv_cache=False,
    collect_layer_indices=collect_layer_indices,
)
```

추론 경로(`denoise_step()`)는 매 denoise 스텝마다 같은 모듈을 다시 호출해서 얻는다
(`modeling_smolvla.py:1748-1756`):

```python
outputs_embeds, _ = self.vlm_with_expert.forward(
    attention_mask=full_att_2d_masks, position_ids=position_ids, past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs], use_cache=self.config.use_cache, fill_kv_cache=False,
)
suffix_out = outputs_embeds[1]
suffix_out = suffix_out[:, -self.config.chunk_size :]
suffix_out = suffix_out.to(dtype=torch.float32)
```

양쪽 다 끝에서 `chunk_size`만큼 자르고(`modeling_smolvla.py:1237`/`:1757`) fp32로 올린다
(`:1239`/`:1758`, 주석: "Original openpi code, upcast attention output").

### 1.2 마지막 차원(720)은 어떤 설정에서 나오는가

`action_out_proj`는 `nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)`
(`modeling_smolvla.py:900`)로 생성된다. `expert_hidden_size`는
`third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:113-116`에서:

```python
lm_expert_config = copy.deepcopy(config.text_config)
hidden_size = lm_expert_config.hidden_size
lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
```

`expert_hidden_size = lm_expert_config.hidden_size`(`smolvlm_with_expert.py:151`). `expert_width_multiplier`
기본값은 `0.75`(`configuration_smolvla.py:140`, 호출부는 `modeling_smolvla.py` 893행 부근에서
`self.config.expert_width_multiplier`를 그대로 넘김 — `VLAFlowMatching.__init__`). `vlm_model_name`
기본값은 `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`(`configuration_smolvla.py:122`).

**"720"은 리터럴 상수가 아니라 `text_config.hidden_size * 0.75`다.** `text_config.hidden_size`가
960이라는 값은 `docs/bridge_attention.md:45`("SmolLM2-360M의 공개적으로 알려진 config
값(hidden_size=960)")에 적혀 있지만, **이 레포 안에서 실제로 로드해 확인한 값이 아니다** — 이
샌드박스에 HF 모델 캐시가 전혀 없어서(`find / -iname "config.json" | xargs grep -l SmolVLM` 결과
0건) 직접 검증 불가. 960×0.75=720은 산수로는 맞지만, 960 자체는 **미확인**(외부 HF 모델 설정).

### 1.3 ARD head 출력이 최종 action에 반영되는 방식

학습(`modeling_smolvla.py:1242-1259`)과 추론(`:1761-1777`) 양쪽에서 **정확히 같은 로직**이 두 번
따로 구현돼 있다(공유 헬퍼 없음 — 아래 7절 참고):

```python
v_t = self.action_out_proj(suffix_out)
if self.config.use_ard:
    stabilizer_residual, actuator_residual = self.ard_heads(suffix_out, bridge_kv_layers=bridge_kv_layers)
    left_residual, right_residual = combine_by_role(stabilizer_residual, actuator_residual, actuator_is_first)
    v_t = torch.cat([
        v_t[..., :arm_dim] + left_residual,
        v_t[..., arm_dim : 2 * arm_dim] + right_residual,
        v_t[..., 2 * arm_dim :],
    ], dim=-1)
```

**결론: ARD head의 출력은 `action_out_proj` 출력에 "더해진다"(additive residual) — 대체하지
않고, 보조 손실 전용도 아니다.** 앞쪽 `2*arm_dim` 채널만 수정되고, 그 뒤 채널(`max_action_dim`까지의
패딩)은 그대로 통과한다. 이 더하기 로직 자체가 `v_t`(최종 flow-matching velocity field, 곧 액션)에
직접 영향을 준다 — 학습 때는 이 `v_t`가 `u_t`와 비교되어 `pos_loss`가 되고(`:1271-1276`), 추론
때는 이 `v_t`가 `_denoise_loop`의 적분 대상이 된다(`:1443` `x_t = x_t + dt * v_t`).

대조로 `ForceHead`는 **액션 채널에 전혀 관여하지 않는다** — `ard.py:161-164`(docstring) 및
`modeling_smolvla.py:1319-1320`(`force_pred = self.ard_force_head(suffix_out)`, 오직 학습 쪽
`VLAFlowMatching.forward`에만 존재, `denoise_step`에는 `ard_force_head` 호출이 전혀 없음 —
`grep -n "ard_force_head" modeling_smolvla.py` 결과 923/928/953-954/1319-1320행 뿐, 전부
`__init__`과 학습 forward 안)에서 확인된다. `force_pred`는 `compute_ard_losses`의 `force_loss`
계산에만 쓰인다(`ard.py:623-627`).

### 1.4 학습 경로와 추론 경로가 다른가

**같은 부분**: `v_t = action_out_proj(suffix_out)` + ARD residual 더하기 로직이 bit 단위로 동일한
코드다(위 1.3). `BridgeAttention` 조건부 입력(`bridge_kv_layers`)도 둘 다 같은 방식으로 전달된다.

**다른 부분**:
- 학습은 **한 번의** forward로 (샘플된 `time`에서의) `v_t` 하나만 계산하고 `u_t`(목표 velocity)와
  비교한다(`modeling_smolvla.py:1196-1198`, `1271-1276`) — flow-matching 학습은 teacher-forcing이라
  반복 적분이 필요 없다.
- 추론은 `_denoise_loop`(`modeling_smolvla.py:1388-1463`)가 `config.num_steps`번 `denoise_step`을
  반복 호출해서 노이즈를 실제로 적분해 없앤다(`x_t = x_t + dt * v_t`, `:1443`). 매 스텝 `vlm_with_expert.forward`를
  다시 호출한다(prefix는 KV 캐시 재사용, suffix만 재계산) — 즉 **같은 "v_t 계산+ARD 더하기" 코드가
  추론에서는 `num_steps`번 반복 실행된다.**
- `ForceHead`/`GradNormLambdas`/`smooth_loss`/`traj_loss`는 **학습 forward에만 존재**한다
  (`ard_extras` dict, `modeling_smolvla.py:1281-1331`) — `denoise_step`/`_denoise_loop`에는 이에
  대응하는 코드가 전혀 없다(grep 확인: `smooth_loss`/`traj_loss`/`ard_gradnorm`은 `modeling_smolvla.py`에서
  1-640행대(학습 관련) 밖에 전혀 등장하지 않음).
- Judge/Selector는 **둘 다** 호출 가능하다 — 학습 forward(`:1228-1235`)와
  `sample_actions_with_decisions`(추론, `:1528-1538`)가 거의 동일한 로직으로 각자 호출한다(중복,
  공유 안 됨).

---

## 2. 역할 배정 (role assignment)

### 2.1 configurable_ard 이전의 기본 매핑

`ard_default_actuator_arm: str = "right"`(`configuration_smolvla.py:180`, 주석: "이 팔이 항상
Actuator 역할"). `docs/configurable_ard.md:18`의 조사 결과 표에 따르면 이 필드는 **configurable_ard
작업 이전부터 이미 config 필드였다** — "역할-팔 매핑(왼팔=Stabilizer) 하드코딩 | 조사 전 예상 |
**이미 config화됨** — `ard_default_actuator_arm`. `role_map`은 그 위에 얹은 이름만 더 직관적인
래퍼"(인용). 즉 "하드코딩에서 config로 바뀐" 게 아니라, 처음부터 config 필드였고 `role_map`은
그 위에 올라간 별칭일 뿐이다.

### 2.2 `role_map` 설정이 읽히는 경로

`SmolVLAConfig.__post_init__`(`configuration_smolvla.py:338` 시작) 안, 409-420행:

```python
if self.role_map is not None:
    if set(self.role_map.keys()) != {"stabilizer", "actuator"}:
        raise ValueError(...)
    stabilizer_arm, actuator_arm = self.role_map["stabilizer"], self.role_map["actuator"]
    if ...not in ("left", "right"): raise ValueError(...)
    if stabilizer_arm == actuator_arm: raise ValueError(...)
    self.ard_default_actuator_arm = actuator_arm
```

`role_map`이 `None`(기본값, `configuration_smolvla.py:319`)이면 이 블록 전체가 스킵되고
`ard_default_actuator_arm`은 그대로 `"right"`다. `role_map`이 주어지면 **단일 소스**를 유지하려고
`ard_default_actuator_arm`을 덮어쓸 뿐 — 실제 분리/결합 로직(`resolve_actuator_is_first`,
`ard.py:264-269`)은 여전히 `ard_default_actuator_arm` 하나만 읽는다.

### 2.3 언어 지시 기반 role classifier

**없다.** `third_party/lerobot/src/lerobot/policies/smolvla/` 디렉터리 전체와 레포 전체에
`role_classif`/`RoleClassifier`/`infer_role`/`classify_role`을 대소문자 무관으로 검색했을 때
**0건**이다(직접 실행한 grep 결과). `resolve_actuator_is_first`(`ard.py:264-269`)는 입력으로
언어/관측을 전혀 받지 않고 `config.ard_default_actuator_arm`(문자열)과 `batch_size`만 받아서,
**배치 내 모든 샘플에 동일한 상수 bool 텐서**를 만든다:

```python
def resolve_actuator_is_first(default_actuator_arm, batch_size, device):
    default_is_left = default_actuator_arm == "left"
    return torch.full((batch_size,), default_is_left, dtype=torch.bool, device=device)
```

즉 역할 배정은 **100% 설정값으로 고정**되며, 샘플마다/언어 지시마다 동적으로 바뀌지 않는다. 같은
배치 안에서도 모든 샘플이 같은 역할 매핑을 쓴다 — RoCo처럼 과제마다 우세팔이 바뀌는 경우
`role_map`을 **과제별로 다른 config 인스턴스**로 전환해야 하며, 모델이 스스로 판단하지 않는다.

---

## 3. 브랜치 지도

(아래는 별도 서브에이전트가 `git log`/`git diff`/`git merge-base`/`git grep`을 직접 실행해서 얻은
결과를 옮긴 것 — 명령 자체는 이 문서 작성자가 재실행하지 않고 서브에이전트의 실행 결과를 그대로
인용한다는 점을 밝힌다.)

**결론**: 5개 브랜치는 서로 갈라진 게 아니라 **하나의 선형 체인**이다. `git merge-base
--is-ancestor`로 확인:

```
main ⊂ closed-loop-structure ⊂ lac-inference ⊂ lac-closed-loop ⊂ configurable-ard ⊂ paper-figures
```

(각 화살표 전부 `git merge-base --is-ancestor origin/A origin/B` → `YES` 확인됨.)

| 브랜치 | `main` 대비 변경 | 다음 브랜치의 조상인가 | 고유 커밋 |
|---|---|---|---|
| `main` | 기준. `main`에만 있는 커밋은 PR 머지 wrapper 커밋 `968cb9f` 하나(실질 내용 없음) | 아님 | — |
| `closed-loop-structure` | `main...origin/closed-loop-structure`: 45 files, +9127/-418. `ard.py`(+652), `configuration_smolvla.py`(+256), 신규 `closed_loop.py`/`decision_logic.py`/`judge.py`/`freq_policy.py`/`token_pruning.py`, `tests/test_ard.py`(967줄) 등 | `lac-inference`의 조상 | `main..origin/closed-loop-structure` 10개("closed-loop stage 1~6" 등) |
| `lac-inference` | `closed-loop-structure` + 신규 `lac.py`(254줄) + `select_action()`에 LAC 추론 경로 연결 | `lac-closed-loop`의 조상 | 3개: `0ee5d8d`(LACConfig/LatentAwareController 추가), `48d8761`(select_action 연결), `8aa232b`(docs/lac.md) |
| `lac-closed-loop` | `lac-inference` + ARD-VLA 2.0 폐루프 경로에 LAC을 실제 스텝 단위로 연결 | `configurable-ard`의 조상 | 3개: `73eefda`/`df855d6`/`37a6f96` |
| `configurable-ard` | `lac-closed-loop` + role_map/action_space/pos_loss_type/per_timestep_norm/angle_unwrap_dims/state_select/action_min_std/state_min_std + `roco_board.yaml` | `paper-figures`의 조상(이번 세션 figure 작업 베이스) | 20개(`6068717`..`045cfd0`) |

**교차 검증**(이름만 보고 추측하지 않고 실제 diff 실행):
- `closed_loop.py`: `closed-loop-structure`→`configurable-ard` 사이 **같은 파일이 계속
  수정된 것**(79 insertions, 9 deletions, 별개 구현 두 개가 아님). 이번 세션 Fig.4 로깅 훅
  작업에 쓴 `closed_loop.py`가 이 체인의 최신 버전과 같은 파일이다.
- `lac.py`: `closed-loop-structure`에는 **존재하지 않음**(`git show origin/closed-loop-structure:.../lac.py` →
  "exists on disk, but not in..." 에러) — `lac-inference`의 `0ee5d8d`에서 신규 추가. 그 이후
  `lac-closed-loop`→`configurable-ard` 사이 diff는 **빈 출력**(완전히 동일).

**충돌/병합 위험**: 없음. `configurable-ard`(와 하위 `paper-figures`)가 이미 나머지 4개 브랜치의
내용을 전부 포함한 가장 앞선 지점이라, "하나로 모으는 작업" 자체가 불필요하다 — `main`을
`configurable-ard`로 fast-forward(또는 PR 머지)하면 끝난다. `closed-loop-structure`/`lac-inference`/
`lac-closed-loop`는 중간 스테이징 커밋일 뿐, 별도로 살려서 합쳐야 할 고유 코드가 없다(위 조상
관계로 직접 확인).

---

## 4. 기능 표

(아래는 별도 서브에이전트가 `configurable-ard` 브랜치에서 `configuration_smolvla.py`/`ard.py`/
`modeling_smolvla.py`/`tests/`를 직접 읽고 `pytest`를 실행해서 만든 표를 옮긴 것이다. 표의
file:line은 서브에이전트가 실행 시점에 읽은 결과이고, 이 문서 작성자가 1절에서 독립적으로 재확인한
`configuration_smolvla.py`의 줄 번호(아래 각주)와 1~2줄 오차가 있을 수 있다 — 두 번 읽은 결과가
갈리면 더 나중에(이 문서 작성 시점에) 직접 읽은 번호를 신뢰한다.)

**전역 사실**: 이 레포의 모든 테스트는 `tests/conftest.py:22-48`의
`_build_tiny_smolvlm_config()`(vision/text hidden_size=32, 레이어 2개, 64×64 이미지)를
`transformers.AutoConfig.from_pretrained`/`AutoProcessor.from_pretrained`에 몽키패치해서 쓰는
합성 모델이다. `grep -rn "from_pretrained" tests/` 결과 실제 `AutoModelForImageTextToText.from_pretrained`
(진짜 SmolVLM2 가중치 로드) 호출은 테스트 전체에 **0건**이다(`load_vlm_weights=False`라 애초에
호출 안 됨, `conftest.py:6` 주석). 전체 테스트 스위트(`configurable-ard` 브랜치, 이 문서 작성자가 직접 재실행):
`.venv/bin/python -m pytest tests/ -q` → **236 passed**, 실패/스킵 0. (이전 세션에서 `paper-figures`
브랜치 — `configurable-ard`보다 앞선 지점, 3절 참고 — 에서는 Fig.4 작업으로 추가한
`tests/test_eval_logging.py`(5개)까지 포함돼 241 passed였다. 두 수치가 다른 건 브랜치가 달라서지
불일치가 아니다.)

| 기능 | config 플래그(기본값, 위치) | 구현 | 테스트 | 테스트 수준 | 비고 |
|---|---|---|---|---|---|
| ARD 비대칭 헤드 | `use_ard: bool = False` (configuration_smolvla.py:170) | Yes — ard.py:196-261, modeling_smolvla.py:941-950/1242-1259/1761-1777 | Yes — test_ard.py, test_configurable_ard.py:81 | **full tiny policy** forward+backward | |
| 대칭 대조군 | `ard_symmetric: bool = False` (:222) | Yes — ard.py:688-814, alpha/beta 0.5/0.5 강제 | Yes — test_ard.py, test_ard_integration.py | full tiny policy | |
| Bridge Attention | `use_bridge_attention: bool = False` (:226) | Yes — ard.py:43-122, modeling_smolvla.py:931-940 | Yes — test_ard.py만 | full tiny policy | |
| GradNorm | `use_gradnorm: bool = False` (:237) | Yes — ard.py:296-439, modeling_smolvla.py:951-952,605-606,642-647 | **Partial** — test_ard.py가 `GradNormLambdas`/`compute_ard_losses(gradnorm=...)`를 직접 호출 | **단위 테스트만** — `build_tiny_policy(use_gradnorm=True)`로 전체 정책 forward를 돌리는 테스트 **없음**(`grep -rn "use_gradnorm" tests/` 0건) | **구멍** |
| ForceHead | `ard_use_force_head: bool = False` (:200) | Yes — ard.py:151-193, modeling_smolvla.py:953-954,1319-1320 | Yes — test_ard.py, test_ard_integration.py | full tiny policy | |
| 관절 토크 입력 | `ard_use_joint_torque: bool = False` (:210) | Yes — modeling_smolvla.py:790-798 | Yes — test_ard.py, test_ard_integration.py | full tiny policy | |
| role_map | `role_map: dict\|None = None` (:319) | Yes — configuration_smolvla.py:409-420 | Yes — test_configurable_ard.py:99-117 | **config 생성 + 단위 테스트만** — role_map이 실제 forward에서 올바른 팔에 잔차를 라우팅하는지 end-to-end 테스트는 **없음** | **구멍** |
| action_space | `action_space: str = "joint"` (:324) | Yes — configuration_smolvla.py:427-433(lac_mirror_sign 무효화) | Yes — test_configurable_ard.py:241-269 | config 단위 테스트만 | |
| pos_loss_type | `pos_loss_type: str = "l2"` (:329) | Yes — modeling_smolvla.py:1271-1276 | Yes — test_configurable_ard.py:123-171 | full tiny policy | |
| per_timestep_norm | `per_timestep_norm: bool = False` (:334) | Yes — ard.py:490-526, modeling_smolvla.py:1177-1188,1453-1461 | Yes — test_configurable_ard.py:173-239 | full tiny policy + 단위 | |
| angle_unwrap_dims | `angle_unwrap_dims: list\|None = None` (:162) | Yes — processor_smolvla.py의 AngleUnwrapProcessorStep/AngleRewrapProcessorStep | Yes — test_configurable_ard.py:272-386 | **프로세서 스텝 단위 테스트만**(정책 forward와 분리) | 아래 5절에서 이 경로를 실제로 호출까지 해봄(처음으로) |
| state_select | `state_select: list\|None = None` (:57) | Yes — modeling_smolvla.py:787-789 | Yes — test_configurable_ard.py:415-446 | full tiny policy + 단위 | |
| action_min_std | `action_min_std: float\|list\|None = None` (:67) | Yes — processor_smolvla.py:49-66 | Yes — test_configurable_ard.py:448-531 | **프로세서 단위 테스트만**(normalize/unnormalize round-trip) | |
| state_min_std | `state_min_std: float\|list\|None = None` (:80) | Yes — processor_smolvla.py:69-101 | Yes — test_configurable_ard.py:533-700 | 프로세서 단위 테스트만 | |
| Judge | `use_judge: bool = False` (:268) | Yes — modeling_smolvla.py 다수(1209,1228-1235,1528-1538 등) | Yes — test_judge.py, test_ard_integration.py:207-247, test_closed_loop.py | full tiny policy | |
| Selector | `use_selector: bool = False` (:280) | Yes — Judge와 동일 경로 | Yes — test_ard_integration.py:249-258, test_decision_logic.py, test_judge.py | full tiny policy | |
| judge_feature_layer | `judge_feature_layer: int = -1` (:274) | Yes — modeling_smolvla.py:1229-1235,1528-1538 | Yes — test_ard_integration.py:233-247, test_judge.py | full tiny policy | |
| LAC | `lac_mode: str = "none"` (:292) | Yes — lac.py(`LatentAwareController`) | Yes — test_lac.py, test_lac_closed_loop.py, test_ard_integration.py:391-480 | full tiny policy | `lac_mirror_sign`은 **미검증**(docstring이 "미검증" 명시, :309 부근) |
| RTC | `rtc_config: RTCConfig\|None = None` (:146 부근) | Yes — lac_mode≠none과 동시 사용 시 에러(:541-546) | **Partial** — RTC 자체를 켠 end-to-end 테스트는 서브에이전트가 못 찾음 | 미확인 | **구멍 가능성** |
| 토큰 프루닝 | `use_token_pruning: bool = False` (:246) | Yes — modeling_smolvla.py:1043-1049 | **Partial** — test_token_pruning.py는 `compute_task_relevance_scores`/`select_tokens` 함수만 단위 테스트 | `grep -rln "use_token_pruning" tests/` **0건** — config로 켜서 정책 forward를 돌리는 테스트 **없음** | **구멍** |
| FreqPolicy | `use_freq_policy: bool = False` (:257) | Yes — modeling_smolvla.py:1291-1292 | Yes — test_freq_policy.py | 미확인(함수 단위/full policy 분리 확인 안 함) | |
| RoCo 프리셋 전체 | configs/presets/roco_board.yaml | — (프리셋, 코드 아님) | Yes — test_configurable_ard.py:776-813(`test_roco_preset_tiny_model_trains_and_infers`) | full tiny policy, train+backward+eval+predict_action_chunk까지 — `run_closed_loop`는 **거치지 않음**, 실제 RoCo 데이터셋도 안 씀 | 5절에서 이 테스트 이후 구간(폐루프 러너)까지 직접 확장해서 실행함 |

**발견한 추가 플래그**(깊이 조사 안 함, 미확인으로 남김): `adapt_to_pi_aloha`,
`use_delta_joint_actions_aloha`(구현 안 됨 — `__post_init__`에서 `NotImplementedError` 추정,
미확인), `compile_model`, `judge_use_progress`, `judge_detach_features`, `lac_log`.

---

## 5. 연결 점검 (실제로 실행함)

아래는 이 문서를 쓰는 동안 **직접 실행**한 결과다(스크립트:
`/tmp/claude-0/.../scratchpad/audit_e2e.py`, 레포에는 포함하지 않음 — 일회성 점검 스크립트). 목표:
타이니 합성 백본 + RoCo 프리셋으로 "관측(카메라 3개·44차원 state·task 문자열) → 전처리
(state_select·min_std) → forward → 손실 → select_action → 폐루프 러너"가 한 번에 되는지, 막히는
지점을 전부 찾을 때까지 실행.

### 1차 시도 — 전체 전처리 파이프라인(`make_smolvla_pre_post_processors`) 구성에서 막힘

```
=== BROKE AT: build pre/post processor pipeline (make_smolvla_pre_post_processors) ===
ProxyError: ... HTTPSConnectionPool(host='huggingface.co', port=443) ... 403 Forbidden
  File ".../processor_smolvla.py", line 138, in make_smolvla_pre_post_processors
    TokenizerProcessorStep(...)
  File ".../tokenizer_processor.py", line 111, in __post_init__
    self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
```

원인: `make_smolvla_pre_post_processors`(`processor_smolvla.py:134-143`)가 항상
`TokenizerProcessorStep(tokenizer_name=config.vlm_model_name, ...)`로 호출한다 —
`tokenizer_name` 대신 사전 생성된 `tokenizer` 객체를 주입하는 경로가 `TokenizerProcessorStep`
클래스 자체에는 있지만(`tokenizer_processor.py:76-77` `tokenizer: Any | None = None`),
`make_smolvla_pre_post_processors`는 이 주입 경로를 쓰지 않는다. `tests/conftest.py`의
`tiny_smolvlm_monkeypatch`는 `AutoConfig`/`AutoProcessor`만 패치하고 **`AutoTokenizer`는
패치하지 않는다**(`conftest.py:66-75` 직접 확인). 기존 테스트 중 이 경로를 건드리는 유일한 것
(`tests/test_configurable_ard.py:352-383`,
`test_angle_unwrap_dims_inserted_into_pipeline_only_when_set`)도 `AutoTokenizer.from_pretrained`를
`object()`(호출하면 바로 깨지는 더미)로 몽키패치해서 **파이프라인 생성만** 확인하고, 실제
`preprocessor(obs)` 호출은 한 번도 하지 않는다(해당 테스트 함수 전체를 읽어 확인 — `_build()`
호출 뒤 `isinstance` 체크만 함).

**결론: 이 레포의 테스트 스위트 전체에서, `make_smolvla_pre_post_processors`가 만든 전처리
파이프라인에 실제 데이터를 통과시켜본 적이 단 한 번도 없다.** 토크나이저가 HF Hub 접근을
요구하기 때문이다.

### 2차 시도 — 작동하는 가짜 토크나이저를 직접 주입하고 재실행

`transformers.AutoTokenizer.from_pretrained`를 (input_ids/attention_mask를 실제로 반환하는) 최소
가짜 토크나이저로 몽키패치한 뒤 재실행:

```
OK: build config (RoCo preset, 3 cameras, 44-dim raw state)
OK: build policy
OK: build pre/post processor pipeline (make_smolvla_pre_post_processors)
OK: raw observation dict (3 cameras + 44-dim state + task string, no batch dim)
OK: preprocessor(raw_obs) -> keys: ['action', 'next.reward', ..., 'observation.images.cam_high',
    'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'observation.state',
    'observation.language.tokens', 'observation.language.attention_mask']
OK: policy.select_action(processed) -> action shape torch.Size([1, 14])
OK: postprocessor(action) -> final action shape torch.Size([1, 14])
OK: training forward() with RoCo preset + 3-camera/44-dim batch -> loss=6.7319 ard_mode=asymmetric
OK: closed_loop.run_closed_loop with a minimal stub env

=== ALL STEPS PASSED ===
```

(첫 시도에서 `run_closed_loop(...)`를 `preprocessor`/`postprocessor` 인자 없이 호출해서
`ValueError: (b,c,h,w) expected, but torch.Size([3, 64, 64])`로 깨진 적이 있는데, 이건 **이
점검 스크립트의 호출 실수**였다 — `closed_loop.py:119-126`의 `run_closed_loop` 시그니처는
`preprocessor`/`postprocessor`를 선택 인자로 받고, `closed_loop.py:164`
(`batch = preprocessor(observation) if preprocessor is not None else observation`)에서 안 주면
raw observation을 그대로 배치로 쓴다 — 프레임워크 버그가 아니라 호출 쪽이 batch 차원 추가를
직접 해줘야 한다는 뜻이었다. `preprocessor=preprocessor, postprocessor=postprocessor`를
넘기자 통과했다.)

### 결론

**"RoCo 프리셋 + 타이니 합성 백본"으로 관측→전처리→forward→손실→select_action→폐루프 러너
전체가 한 번에 돈다** — 단, 막히는 지점은 명확히 하나였다:

1. **(진짜 막힘, 코드 수정 또는 테스트 인프라 확장 필요)** `make_smolvla_pre_post_processors`가
   오프라인에서 토크나이저를 못 만든다 — `AutoTokenizer.from_pretrained`에 사전 생성 토크나이저를
   주입할 방법이 함수 레벨에 없다(클래스 레벨에는 있음). 이 경로를 실제 데이터로 테스트하려면
   매번 수동으로 `transformers.AutoTokenizer.from_pretrained`를 몽키패치해야 하고, 그런 패턴이
   `tests/conftest.py`에 없다.
2. (막힘 아님, 사용법 문제) `run_closed_loop`에 `preprocessor`/`postprocessor`를 명시적으로
   넘겨야 한다 — 문서화는 돼 있지만(파라미터 docstring), 누락하면 바로 깨진다.

---

## 6. 정책 실행 인터페이스 초안 (RoCo 평가용)

**주의**: RoCo 공식 평가 환경의 실제 API(관측을 주는 함수 시그니처, 액션을 받는 함수 시그니처,
초당 호출 빈도 등)는 이 레포 어디에도 없고 이 세션에서 열어볼 수도 없었다(HF Hub 차단) — 아래는
**이 레포가 이미 가진 `ClosedLoopEnv` 프로토콜과 `predict_action_chunk_with_decisions`의 실제
입출력을 그대로 베낀 초안**이지, 공식 스펙과 대조해서 맞춘 것이 아니다. 공식 스펙과 다른 부분은
전부 "미확인"으로 남긴다.

```python
from typing import Any, Protocol

class RoCoEvalEnv(Protocol):
    """closed_loop.py:88-100의 ClosedLoopEnv를 그대로 따름 — 이미 구현된 러너가 이 다섯
    메서드(+ 선택적 run_recovery)만 요구하므로, RoCo 평가 하네스를 이 모양으로 감싸면
    run_closed_loop()를 그대로 재사용할 수 있다."""

    def observe(self) -> dict[str, Any]:
        """반환 dict의 정확한 키 이름(카메라 3개의 실제 이름, observation.state의 44차원
        순서)은 **미확인** — roco_board.yaml:17-24 주석이 기술하는 순서(EE 14 + 관절위치 14 +
        관절속도 14 + 그리퍼 2)는 사용자가 RoCo 공식 데이터셋 info.json을 직접 읽어 확인한
        값이지, 이 레포가 실제 평가 환경과 통신해서 확인한 값이 아니다. "task" 키는 여기서
        채우지 않아도 된다 — run_closed_loop가 iteration마다 직접 덮어쓴다
        (closed_loop.py:163)."""
        ...

    def execute(self, action: "torch.Tensor") -> None:
        """action: shape (14,) — predict_action_chunk_with_decisions가 돌려준 (1, chunk_size, 14)
        청크에서 한 스텝을 슬라이스한 것(closed_loop.py:293,296 `_execute_chunk`), postprocessor를
        이미 거친(역정규화된) 값이다. action_space="ee"이므로 엔드이펙터 **절대** 목표
        (EE xyz+rxryrz 6 + 그리퍼 1, 팔당) — roco_board.yaml:12-16 주석 기준. 로봇 API가 실제로
        이 포맷을 그대로 받는지는 **미확인**."""
        ...

    def subtask_finished(self) -> bool: ...  # closed_loop.py:98과 동일
    def settle(self) -> None: ...            # closed_loop.py:99와 동일
    def safe_stop(self) -> None: ...         # closed_loop.py:100과 동일
    def run_recovery(self, recovery_id: str) -> None: ...  # 선택. 없으면 RECOVER가 RETRY로 처리됨
```

**호출부**:

```python
actions = run_closed_loop(
    policy, env, decision_logic, goal=task_instruction, start_node="...",
    preprocessor=preprocessor, postprocessor=postprocessor,  # 5절 — 빠뜨리면 바로 깨짐
    max_subtask_steps=???,  # 아래 참고 — 미확인/미설정
)
```

### n_action_steps 관련 중요한 발견

**`closed_loop.py`는 `config.n_action_steps`를 전혀 참조하지 않는다** —
`grep -n "n_action_steps" closed_loop.py modeling_smolvla.py` 결과, `n_action_steps`가 실제로
쓰이는 곳은 `modeling_smolvla.py:273`(큐 길이)과 `:534`(`select_action()`이 청크에서
`[:self.config.n_action_steps]`만 큐에 넣는 부분) **뿐**이다. `predict_action_chunk_with_decisions`
(closed_loop.py가 쓰는 함수)는 큐를 쓰지 않고 `actions = actions[:, :, :original_action_dim]`
(채널만 자름, modeling_smolvla.py:492)으로 끝 — **시간축은 자르지 않는다.**

`_execute_chunk`(closed_loop.py:260-303)는 `env.subtask_finished()`가 True가 되거나
`max_steps`(=`run_closed_loop`의 `max_subtask_steps` 인자에서 옴, `:247-248`)에 닿을 때까지
**받은 청크를 끝까지(`actions.shape[1]`, 기본 RoCo `chunk_size=20`) 실행한다.** `max_subtask_steps`를
명시적으로 안 주면(기본 `None`) 길이 제한이 아예 없다 — `grep -rn "max_subtask_steps\s*="`
결과 이 값을 설정하는 코드는 `tests/test_closed_loop.py:353`(테스트 전용, 값 2) **한 곳뿐**이고,
`config.n_action_steps`를 `max_subtask_steps`에 연결하는 코드는 레포 어디에도 없다(0건).

**즉 `roco_board.yaml:97`의 주석("매 5스텝마다 재계획(replan) — 폐루프 피드백 반영 빈도")은
`run_closed_loop`의 실제 동작과 맞지 않는다** — `n_action_steps=5`를 설정해도, 이 값을
`run_closed_loop(..., max_subtask_steps=5)`로 **직접 넘기지 않는 한** 폐루프 러너는 매 5스텝이
아니라 매 20스텝(전체 chunk)마다만 재계획한다(또는 `env.subtask_finished()`가 더 일찍 True가
될 때). RoCo 평가 인터페이스를 짤 때 이 값을 명시적으로 연결해야 하며, 현재 어떤 스크립트도
이렇게 연결하지 않는다.

---

## 7. 남은 작업 목록

| # | 항목 | 상태 | 우선순위 | 난이도 | 비고 |
|---|---|---|---|---|---|
| 1 | GradNorm 비활성 태스크 처리 수정 | **이미 완료** | — | — | 커밋 `3c387e5`("Fix GradNorm bias from always-zero force_loss via explicit active flags")가 이미 `configurable-ard`(현재 HEAD)의 조상임을 `git merge-base --is-ancestor 3c387e5 HEAD`로 직접 확인. `ard.py:339-413`의 `active`/`newly_active`/`active_idx` 로직이 그 구현. 사용자가 목록에 넣었지만 재확인 결과 할 일이 아니라 완료 항목이다. |
| 2 | VLM LoRA 연결 여부 | **연결 안 됨** | 중 | 중 | PEFT 타겟 모듈은 정의돼 있다(`modeling_smolvla.py:807-816` `_get_default_peft_targets`, VLM q/v_proj + 공용 projection 타겟) — base `PreTrainedPolicy`/`pretrained.py` 인프라가 이걸 쓸 수 있게 돼 있다. 하지만 이 프로젝트의 실제 학습 스크립트 `scripts/train_ard.py`(440줄)에는 `peft`/`lora`/`freeze_vision` 문자열이 **0건**이다(직접 grep 확인) — 즉 `roco_board.yaml:113-116` 주석이 말하는 "VLM은 LoRA" 계획이 학습 스크립트에 전혀 와이어링되지 않았다. `--use-peft` 같은 CLI 플래그를 `train_ard.py`에 추가하고 `_validate_peft_config`(`modeling_smolvla.py:818-827`) 경로를 실제로 타는지 확인 필요. |
| 3 | role classifier 결정 | **미구현, 설계 결정 필요** | 중 | RoCo 1차 제출 범위에선 낮음(현재 role_map 고정으로 충분), 학습형으로 가면 높음 | 2.3절 확인대로 코드에 없음. RoCo처럼 "왼팔이 주작업"이 거의 항상 성립하는 과제라면 `role_map` 고정으로 충분할 수 있다 — 동적 분류기가 꼭 필요한지부터 결정해야 한다. |
| 4 | `per_timestep_norm` a·s 맞추기 | **미착수**(의도적으로 보류) | RoCo 제출 전 필수 | 낮음(스크립트는 이미 있음) | `scripts/fit_timestep_norm.py`가 이미 존재(`roco_board.yaml:34` 참고)하지만 HF Hub 차단으로 실제 RoCo 데이터셋에 돌려본 적이 없다(`roco_board.yaml:32-36` 명시) — 지금은 항등값(a=1,s=0)이라 `per_timestep_norm=True`가 사실상 아무 효과가 없는 상태로 켜져 있다. |
| 5 | **(신규 발견) `n_action_steps`가 폐루프 러너에 연결 안 됨** | 미구현 | RoCo 제출 전 필수(요청한 "청크 실행" 의미론과 직결) | 낮음(파라미터 하나 넘기면 됨, 단 설계 결정 필요 — 매 호출 `max_subtask_steps=config.n_action_steps`를 자동으로 넘길지, 호출자가 명시하게 둘지) | 6절 참고. `closed_loop.py`에 `n_action_steps` 참조가 전혀 없음을 grep으로 확인. |
| 6 | **(신규 발견) 오프라인 토크나이저 생성 경로 없음** | 미구현 | 테스트 인프라 개선 — 중 | 낮음 | `make_smolvla_pre_post_processors`가 `AutoTokenizer.from_pretrained`를 하드코딩 호출(`processor_smolvla.py:138-143`), `tests/conftest.py`가 이를 안 막아줌. `tiny_smolvlm_monkeypatch`에 `AutoTokenizer.from_pretrained` 패치를 추가하거나, `make_smolvla_pre_post_processors`에 `tokenizer=` 주입 인자를 추가하면 해결. |
| 7 | GradNorm 전체-정책(end-to-end) 테스트 부재 | 테스트 구멍 | 중 | 낮음 | 4절 표 — `build_tiny_policy(use_gradnorm=True)`로 forward 전체를 돌리는 테스트가 0건. 단위 테스트(`ard.py`의 `GradNormLambdas`를 직접 호출)만 있음. |
| 8 | role_map의 end-to-end(실제 forward 라우팅) 테스트 부재 | 테스트 구멍 | 중 | 낮음 | 4절 표 — `role_map`이 설정값을 올바르게 바꾸는지만 테스트됐고, 그 값이 실제로 올바른 물리적 팔에 잔차를 라우팅하는지(`resolve_actuator_is_first`→`combine_by_role` 체인 전체)는 full-policy 테스트가 없다. |
| 9 | 토큰 프루닝 / RTC 전체-정책 테스트 부재 | 테스트 구멍 | 낮음 | 낮음 | 4절 표 — `use_token_pruning=True`/RTC를 켠 end-to-end 테스트가 0건 또는 미확인. |
| 10 | ARD "더하기" 로직이 학습/추론에 중복 구현 | 코드 품질(버그는 아직 아님) | 낮음 | 낮음~중(리팩터 범위에 따라) | 1.3절 — `modeling_smolvla.py:1242-1259`와 `:1761-1777`이 같은 로직을 따로 유지한다. 둘 중 하나만 고치고 다른 쪽을 잊으면 학습/추론이 조용히 갈라질 위험. |
| 11 | 실제 RoCo 데이터셋으로 아무것도 돌려본 적이 없음 | 가장 큰 미확인 | 최우선(전체 작업의 전제) | HF Hub 접근 권한에 달림 — 이 세션에서는 해결 불가 | 지금까지의 모든 검증(`test_roco_preset_tiny_model_trains_and_infers` 포함 5절 점검)이 합성 모델+가짜 데이터다. 실제 SmolVLM2 가중치, 실제 44/14차원 통계, 실제 토크나이저로 한 번도 안 돌려봤다. |

---

## 8. 논문 제목·설명과 구현의 어긋남

**중요한 전제**: 이 레포 어디에도 논문 원고(.tex, abstract, 본문)가 없다 — `find . -iname "*.tex"`,
`find . -iname "*paper*"`(모든 브랜치 포함, `main`/`configurable-ard`/`closed-loop-structure`/
`lac-inference`/`lac-closed-loop` 전부 `git ls-tree`로 확인) 결과 **그림 파일(`paper/figures/`,
`paper/scripts/`) 외에는 아무것도 없다.** 따라서 아래는 사용자가 이번 요청에서 준 제목 문자열
("ARD-VLA: Asymmetric Residual Dual Action Heads for Bimanual Fine Manipulation") + 이 레포에
있는 유일한 아키텍처 "설명" 자료(`paper/figures/fig2_architecture.*`, 이전 세션에서 "code-verified"로
커밋됨 — 045cfd0)를 구현과 대조한 것이다. **다른 논문 텍스트가 없으므로 더 비교할 대상이 없다.**

| 제목/설명 요소 | 구현과 일치? | 근거 |
|---|---|---|
| "Asymmetric Residual" | **일치** | 1.3절 — ARD head 출력은 `action_out_proj` 출력에 더해지는 residual이고(`modeling_smolvla.py:1252-1259`), Stabilizer/Actuator 두 역할에 비대칭 손실 가중치 `ard_alpha: float = 0.3`/`ard_beta: float = 0.7`(`configuration_smolvla.py:181-182`, 직접 확인)로 결합된다(`total = alpha * stab_loss + beta * act_loss`, `ard.py:673`). `ard_symmetric=True`면 이 값이 0.5/0.5로 강제 덮어써진다(`configuration_smolvla.py:478-485`) — 비대칭 가정 자체를 끄는 대조군이라 제목과는 별개 모드. |
| "Dual Action Heads" | **일치** | `AsymmetricResidualHeads`는 정확히 두 개의 head(`stabilizer_head`, `actuator_head`, `ard.py:222-231`)를 만든다. |
| "Bimanual" | **일치, 단 범위 제한 있음** | 코드는 "선두 `2*arm_dim` 채널 = 왼팔+오른팔"이라는 2-팔 가정을 구조적으로 강제한다(`ard.py:22-25`, `split_by_role`/`combine_by_role`이 정확히 2개 블록만 다룸) — 3팔 이상에는 애초에 적용 불가. 이름과 일치하는 제약이지 어긋남은 아니다. |
| "Fine Manipulation" | **구현과 어긋난다기보다, 아직 입증할 실험이 없음** | 정밀 조작 성능을 뒷받침할 실제 평가 결과가 이 레포에 전혀 없다(7절 #11) — Fig.4(`paper/scripts/make_fig4.py`)도 실제 로그 없이 스모크 테스트만 통과한 상태(이전 세션 작업). 이건 "제목이 구현과 다르다"가 아니라 "제목이 주장하는 바를 뒷받침할 데이터가 아직 없다"는, 범위가 다른 문제라서 구분해서 적는다. |
| Fig.2가 그리는 아키텍처(`W_out 720→32`, `Stabilizer/Actuator head 720→360→GELU→360→8`) | **일치하되, "어떤 설정"을 그렸는지 암묵적** | `nn.GELU()`(`ard.py:224`,`229`) 사용은 일치. 그림의 "32"(max_action_dim)/"8"(arm_dim)은 `ard_arm_dim` 기본값 8(`configuration_smolvla.py:172` "팔 하나당 자유도(관절 7+그리퍼 1)")과 OpenArm 16채널 전제를 그린 것이지, **RoCo 프리셋(`ard_arm_dim=7`, 14채널, `roco_board.yaml:50-51`)을 그린 게 아니다.** 이건 오류가 아니라 — 그림이 ARD-VLA의 원래(OpenArm) 기본 설정을 그렸을 뿐, RoCo용 치수는 별도라는 점이 그림 자체에는 명시돼 있지 않다는 지적이다. RoCo로 논문을 쓴다면 이 그림(또는 별도 그림)에 "ard_arm_dim=7(RoCo)" 같은 라벨이 필요하다. |

---

## 부록: 이 문서를 쓰며 실행한 명령/스크립트

- `git fetch origin`, `git log --oneline --all`, `git merge-base --is-ancestor <A> <B>` (3절 서브에이전트가 직접 실행)
- `.venv/bin/python -m pytest tests/ -q` (4절 서브에이전트가 직접 실행, 236 passed)
- `/tmp/.../scratchpad/audit_e2e.py` — 이 문서 작성자가 직접 작성하고 실행(5절). 레포에는 커밋하지 않음(일회성 점검 스크립트, 사용자가 "코드는 고치지 마"라고 명시했으므로 레포 파일로 남기지 않았다).
