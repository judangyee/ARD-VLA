# 설정 가능한 ARD (Configurable ARD) — RoCo Challenge 프리셋

[← README로 돌아가기](../README.md)

ARD-VLA는 원래 OpenArm 16채널(팔당 8 DoF, 왼팔=Stabilizer 고정, chunk 50, pos_loss는 L2)
전제로 설계됐습니다. RoCo Challenge처럼 대회마다 로봇/액션 공간/우세 팔이 바뀌는 과제에
재사용하려고, 이 하드코딩을 config 손잡이 5개(`role_map`/`action_space`/`pos_loss_type`/
`per_timestep_norm`/`ard_angle_dims`)로 뺐습니다. **이번 작업은 구조 구현 + 작은
합성 모델 검증까지입니다** — 실제 SmolVLM2 가중치나 RoCo 공식 데이터셋에서는 돌려보지
않았습니다(아래 "검증한 것 / 못 한 것" 참고).

## 조사 결과: 무엇이 실제로 하드코딩돼 있었나

| 항목 | 조사 전 예상 | 실제 |
|---|---|---|
| 액션 차원(16)/팔별 슬라이스/그리퍼 인덱스 | 리터럴 16이 박혀있을 것 | **이미 config화됨** — `ard_arm_dim`/`ard_gripper_dim`(리터럴 16/8은 grep 0건). 진짜 하드코딩은 `split_by_role`/`combine_by_role`/`AsymmetricResidualHeads`(`ard.py`)가 "연속·동일길이·왼팔먼저" 구조를 함수 시그니처(`arm_dim: int` 하나)로 강제하는 것 |
| 역할-팔 매핑(왼팔=Stabilizer) | 하드코딩 | **이미 config화됨** — `ard_default_actuator_arm`. `role_map`은 그 위에 얹은 이름만 더 직관적인 래퍼 |
| 손실 종류(pos_loss가 L2) | 하드코딩 | 하드코딩 맞음 — `modeling_smolvla.py`의 `F.mse_loss(u_t, v_t, reduction="none")` 한 곳 |
| chunk_size/n_action_steps | 혹시 어딘가 무시될 수도 | **이미 config대로 동작** — `n_action_steps <= chunk_size` 검증까지 있음. LAC/폐루프 러너도 둘 다 config 값만 읽음(하드코딩 없음) |
| 액션 정규화 | 차원별 평균/표준편차만? | 맞음 — `NormalizerProcessorStep`/`UnnormalizerProcessorStep`이 시간축 awareness 전혀 없이 차원별 통계만 사용 |
| 비전 인코더 freeze/LoRA | 확인 필요 | `freeze_vision_encoder`(기본 True)**가 이미 존재**. `_get_default_peft_targets()`의 타겟 정규식도 이미 차원 독립적(이름 기반) |
| quantization | 기존 옵션이 있을 것으로 예상 | **repo 전체에 존재하지 않음**(grep 0건) — 추가하지 않기로 결정(아래 참고) |

**결론**: 숫자(16/8)는 애초에 하드코딩이 아니었다 — 진짜 바꿔야 했던 건 (a) 메인 회귀
손실의 종류, (b) 액션 공간 종류(joint vs ee)가 mirror_sign에 미치는 영향, (c) 청크 내
위치별 정규화(신규 기능), (d) 위 설정들을 한 번에 담아줄 프리셋 파일. 그래서 기존
`ard_arm_dim`/`ard_gripper_dim`/`ard_default_actuator_arm`은 전혀 건드리지 않고, 그 위에
얇은 래퍼(`role_map`)와 독립적인 신규 필드(`action_space`/`pos_loss_type`/
`per_timestep_norm`)만 추가했습니다 — `modeling_smolvla.py`의 기존 7곳 슬라이스 조립
코드는 리팩터링하지 않았습니다(블라스트 radius 최소화, 임의 비연속/비대칭 슬라이스는
현재 어떤 요구사항에도 필요하지 않음).

## 새 설정 필드

| 필드 | 기본값 | 의미 |
|---|---|---|
| `role_map` | `None` | `{"stabilizer": "left"\|"right", "actuator": "left"\|"right"}`. 주면 `ard_default_actuator_arm = role_map["actuator"]`로 덮어씀(단일 소스 유지) |
| `action_space` | `"joint"` | `"joint"`\|`"ee"`. `"ee"`면 `lac_mirror_sign`을 강제로 `None`(미러링 없음)으로 덮어쓰고 경고 로그 — 엔드이펙터 델타에는 "관절 축 미러링"이라는 전제가 성립하지 않기 때문 |
| `pos_loss_type` | `"l2"` | `"l2"`\|`"l1"`\|`"smooth_l1"`. 메인 flow-matching 회귀 손실(`modeling_smolvla.py`의 `F.*_loss(u_t, v_t)`) 종류 |
| `per_timestep_norm` | `False` | 켜면 청크 내 실행 위치 t(0..chunk_size-1)마다 σ(t)=a+s·√t로 추가 스케일 |
| `per_timestep_norm_a`/`per_timestep_norm_s` | `1.0`/`0.0` | 스칼라 또는 (실제 액션 차원,) 리스트. `scripts/fit_timestep_norm.py`로 피팅 |
| `ard_angle_dims` | `None` | 각 팔 블록 내 회전(라디안) 채널의 상대 인덱스 리스트(0..`ard_arm_dim`-1). 메인 pos_loss 계산에서 이 채널들의 (목표-예측) 차이를 [-π, π)로 감싸 2π 경계 불연속을 없앤다. `use_ard=True`일 때만 의미 있음 |

전부 기본값이 기존 동작과 동일해서 **꺼져 있으면 bit-for-bit 동일**합니다
(`tests/test_configurable_ard.py::test_new_config_fields_default_to_noop`).

### `pos_loss_type` — 왜 L1/Smooth-L1인가

L2(MSE)는 예측이 목표에 가까워질수록 기울기(`2*(pred-target)`)가 0으로 사라져서, 거의
맞았는데 "끝까지 밀어넣지" 않는 경향이 있습니다. L1의 기울기는 거리와 무관하게 크기가
항상 1이라 이 문제가 없습니다 — RoCo Challenge 1회 우승팀이 L2→L1 전환으로 이 문제를
해결했다고 보고했습니다(arXiv 2603.15469). `smooth_l1`은 그 중간(큰 오차엔 L1처럼, 작은
오차엔 L2처럼) 동작합니다. 이 성질은 순수 수학적 사실이라
`tests/test_configurable_ard.py::test_l1_gradient_does_not_vanish_near_target_unlike_l2`로
직접 확인했고, 실제 모델 wiring은
`test_pos_loss_type_dispatches_to_matching_torch_loss_fn`이 `torch.nn.functional.*_loss`
호출을 가로채서 확인합니다.

### `per_timestep_norm` — 왜 청크 내 위치별로 스케일을 다르게 주는가

먼 미래(청크 뒤쪽 위치)일수록 예측 분산이 커지는 경향이 있다는 관찰을 바탕으로, LeHome
우승 보고서(arXiv 2606.27163)가 위치별 정규화 스케일을 두는 방식을 보고했습니다. 구현은
**표준 차원별(평균/표준편차) 정규화와 독립적인, 모델 경계에서만 적용되는 추가 레이어**
입니다:

- 학습: `VLAFlowMatching.forward()`가 ground-truth `actions`를 받는 즉시 σ(t)로 나눔
  (noise/x_t/u_t 계산 전).
- 추론: `_denoise_loop()`(모든 `sample_actions*` 변형이 공유하는 단 하나의 반환 지점)가
  최종 `x_t`를 σ(t)로 다시 곱해서 되돌림.
- 그 사이(denoise 루프 전체, LAC 포함)는 전부 "스케일된 공간"에서 동작 — 표준
  `NormalizerProcessorStep`은 전혀 건드리지 않음.

**LAC와 동시에 켜는 경우 주의**: LAC(`lac_mode`)의 ρ/ω 임계값(`lac_tau_rho`/`lac_tau_omega`
등)은 "스케일된 공간" 기준으로 계산되게 됩니다. `per_timestep_norm`을 켜면 그 임계값들의
의미가 미세하게 바뀔 수 있어, 재캘리브레이션 전까지는 함께 켜지 않는 걸 권장합니다 — RoCo
프리셋이 `lac_mode="none"`을 쓰는 이유입니다.

`scripts/fit_timestep_norm.py --dataset-repo-id <repo> --chunk-size <N>`로 실제 데이터셋의
위치별 정규화된 액션 표준편차에 최소제곱 피팅해서 a/s를 구합니다. **이 샌드박스는 HF Hub
접근이 막혀 있어 실행해보지 못했습니다** — 로컬/GPU 환경에서 돌려야 합니다.

### `ard_angle_dims` — 왜 회전 채널에 각도 래핑이 필요한가

RoCo처럼 `action_space="ee"`에서 rx/ry/rz(라디안)는 ±π에서 순환합니다 — 예측이 3.14,
목표가 -3.14라면 물리적으로는 거의 같은 각도인데 raw 차이는 ~2π(약 6.28)로 보여서
L1/L2/Smooth-L1 손실과 그 그래디언트가 거짓으로 커집니다. `ard_angle_dims`를 설정하면
(`ard.py`의 `wrap_to_pi`/`build_angle_channel_mask`/`apply_angle_aware_target`),
`modeling_smolvla.py`의 pos_loss 계산에서 그 채널들의 `(u_t - v_t)` 차이를 `[-π, π)`로
감싼 뒤, **기존 `F.mse_loss`/`F.l1_loss`/`F.smooth_l1_loss` 호출부를 그대로 둔 채** 그
자리에 넣을 "각도-인식 타깃"(`u_t - wrapped_diff`)으로 바꿔치기합니다 — 이 함수들이
내부적으로 입력-타깃 차이만 보기 때문에 가능한 트릭입니다.

**단위 관련 주의(미검증)**: `wrap_to_pi`는 입력이 실제 라디안 단위일 때만 정확합니다.
하지만 이 지점의 `u_t`/`v_t`는 이미 표준 차원별(평균/표준편차) z-score 정규화를 거친
값입니다(`NormalizerProcessorStep`이 `policy.forward()` 호출 **전에** 적용됨, 아래 "확인한
것" 참고) — 그 채널의 std가 정확히 1이 아니면 경계가 `2π`가 아니라 `2π/std`로 바뀝니다.
모델에는 `std`가 전달되지 않아 이 보정을 하지 못하므로, 표준 정규화를 쓰는 설정(RoCo
포함)에서는 **근사치**입니다. 정확한 경계가 필요하면 실제 std로
`2π/std`를 계산해 두 번째 wrap을 추가하는 식으로 확장해야 합니다 — 지금은 구현하지
않았습니다.

### `action_space`/`role_map` — RoCo Challenge 공식 스키마와의 매핑

RoCo Challenge 공식 데이터셋(`rocochallenge2025/rocochallenge2026_Industrial_Assembly`,
LeRobot v3.0, ~1.1GB)은 **이 샌드박스에서 여전히 HF Hub 접근이 막혀 있어 meta/info.json을
직접 열어보지 못합니다**. 다만 **사용자가 직접 확인해서 전달한 스키마(2026-10-29)** 는
다음과 같습니다 — action 14차원 순서:

```
left_ee_x, left_ee_y, left_ee_z, left_ee_rx, left_ee_ry, left_ee_rz, left_gripper,
right_ee_x, right_ee_y, right_ee_z, right_ee_rx, right_ee_ry, right_ee_rz, right_gripper
```

즉 팔별로 연속된 **[엔드이펙터 pose 6(xyz+rxryrz) + 그리퍼 1] = 7채널** 블록이 왼팔,
오른팔 순서로 이어집니다. 이로부터:

- `ard_arm_dim=7`, `ard_gripper_dim=1`(마지막 1채널=그리퍼) — **확인됨**, 추측이 아니라
  이 스키마와 정확히 일치합니다.
- `ard_angle_dims=[3, 4, 5]` — 각 7채널 블록 내 rx/ry/rz의 상대 인덱스. **확인됨**(스키마
  순서 그대로).
- 채널이 관절이 아니라 엔드이펙터 목표 pose라서 `action_space="ee"`.
- "대부분 왼팔만 사용"(공개 설명, 이 부분은 여전히 **미확인** — 스키마 자체와는 별개)
  → 왼팔이 실제 작업(Actuator), 오른팔이 보조(Stabilizer) →
  `role_map={"stabilizer": "right", "actuator": "left"}`.

## RoCo Challenge 프리셋 (`configs/presets/roco_board.yaml`)

```python
import yaml
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

overrides = yaml.safe_load(open("configs/presets/roco_board.yaml"))
config = SmolVLAConfig(input_features=..., output_features=..., **overrides)
```

프리셋 파일 자체에 각 값마다 한 줄 이유 주석이 있습니다. 요약:

| 필드 | 값 | 이유 |
|---|---|---|
| `role_map` | `{stabilizer: right, actuator: left}` | 공개 설명: 왼팔이 실제 작업 수행(이 부분은 미확인) |
| `ard_arm_dim`/`ard_gripper_dim` | `7`/`1` | **확인된 스키마**(팔당 ee pose 6 + 그리퍼 1) |
| `action_space` | `ee` | 채널이 관절이 아니라 엔드이펙터 목표 |
| `ard_angle_dims` | `[3, 4, 5]` | **확인된 스키마**의 rx/ry/rz 상대 인덱스. 단위/정규화 상호작용은 미검증(위 참고) |
| `pos_loss_type` | `l1` | RoCo 1회 우승팀 보고(arXiv 2603.15469) |
| `chunk_size`/`n_action_steps` | `20`/`5` | 조립 작업에 맞춘 재계획 빈도(공개 스펙 10fps 기준 가정) |
| `per_timestep_norm` | `true`(a/s는 미피팅 항등값) | LeHome 보고(arXiv 2606.27163). 실데이터 피팅 필요 |
| `freeze_vision_encoder` | `true` | 기존 기본값과 동일(명시적으로 박아 학습 범위를 드러냄) |
| `lac_mode` | `none` | `per_timestep_norm`과의 상호작용(위 참고)을 재캘리브레이션 전까지 피함 |
| (quantization 없음) | — | 코드베이스에 그런 옵션이 아직 없음(아래 참고) |

## quantization을 추가하지 않은 이유

작업 지시에는 "기존 quantization 옵션은 그대로 두되 RoCo 프리셋에선 None"이라고 돼 있었지만,
실제로 `configuration_smolvla.py`를 포함해 레포 전체에 `quantiz`로 시작하는 식별자가 **0건**
입니다 — 그런 옵션이 존재하지 않습니다. 없는 기능을 "그대로 두는" 것은 불가능하고, 쓰이지도
않을 placeholder 필드를 새로 만드는 건 YAGNI라 추가하지 않기로 했습니다(이 결정은 작업 전
사용자에게 확인받았습니다). 실제로 양자화 지원이 필요해지면 그때 config 필드 + 로딩 경로를
함께 추가하는 게 맞습니다.

## 새/수정 파일

- `configuration_smolvla.py` — `role_map`/`action_space`/`pos_loss_type`/`per_timestep_norm`
  (+`_a`/`_s`)/`ard_angle_dims` 필드와 `__post_init__`/`validate_features()` 검증·파생 로직.
- `ard.py` — `compute_timestep_norm_sigma`/`apply_timestep_norm`/`invert_timestep_norm`
  (순수 함수 3개, 패딩 채널은 identity로 채움), `wrap_to_pi`/`build_angle_channel_mask`/
  `apply_angle_aware_target`(각도 래핑 순수 함수 3개).
- `modeling_smolvla.py` — `F.mse_loss` 호출을 `pos_loss_type` 3분기로 교체(+ `ard_angle_dims`
  설정 시 각도-인식 타깃으로 바꿔치기), `forward()`/`_denoise_loop()` 경계에서
  `per_timestep_norm` 스케일/역스케일.
- `scripts/fit_timestep_norm.py` (신규) — 데이터셋에서 σ(t)의 a/s 피팅.
- `configs/presets/roco_board.yaml` (신규).
- `tests/test_configurable_ard.py` (신규).

## 검증한 것 / 못 한 것

**검증함** (`tests/test_configurable_ard.py`, 작은 합성 SmolVLM 백본, Hub 접근 없음):
- 새 필드 전부 기본값일 때 기존 forward/backward와 bit-for-bit 동일.
- `ard_arm_dim=7`(14차원, 7+7) 레이아웃에서 `AsymmetricResidualHeads` 출력 차원이 자동으로
  맞춰져 forward/backward 정상 동작.
- `role_map` 반전 시 `ard_default_actuator_arm`과 `resolve_actuator_is_first()`의 라우팅이
  실제로 반대로 뒤집히는지. 잘못된 `role_map`(키 누락/같은 팔) 거부.
- `pos_loss_type` 3종: 수학적 성질(L2 기울기 소실, L1은 크기 1 유지)과, 실제
  `modeling_smolvla.py`가 올바른 `torch.nn.functional.*_loss`로 분기하는지(함수 호출
  가로채기로 직접 확인). 잘못된 값 거부.
- `per_timestep_norm`: 스칼라/채널별(+패딩) 양쪽 다 정규화→역정규화 왕복 오차 거의 0, 켜면
  실제로 loss가 바뀌면서도 finite 유지, 추론 경로(`predict_action_chunk`)도 finite. `a<=0`
  거부.
- `action_space="ee"`가 `lac_mirror_sign`을 강제로 `None`으로 덮어쓰고 경고를 남기는지(그리고
  `"joint"`에서는 그대로 두는지). 잘못된 값 거부.
- RoCo 프리셋 yaml이 유효한 `SmolVLAConfig`로 로드되는지(derived 값 포함, `ard_angle_dims`
  포함), 그리고 그 값 그대로(14차원=7+7) 작은 합성 모델에서 몇 스텝 학습(loss.backward())/
  추론(predict_action_chunk)이 정상 동작하는지.
- `wrap_to_pi`가 2π 경계 불연속을 없애는지, `build_angle_channel_mask`가 양팔 블록 둘 다에
  적용되고 패딩 채널은 건너뛰는지, `apply_angle_aware_target`이 마스크된 채널만 wrap하는지.
  `ard_angle_dims=None`(기본)이면 `apply_angle_aware_target`이 **호출조차 되지 않는지**
  (함수 호출 가로채기로 직접 확인 — bit-exact 보장의 근거). 잘못된 `ard_angle_dims`(빈
  리스트/중복/범위 밖/`use_ard=False`와 조합) 거부.
- `NormalizerProcessorStep`이 실제로 차원별 독립 통계(`(tensor-mean)/std`, 코드 직접 확인)를
  쓰는지 — 위치(m)/회전(rad) 등 단위가 달라도 각 채널이 자기 자신의 mean/std로만 정규화되므로
  문제없음(수정 불필요, 코드 확인만).
- LAC(`lac.py`)의 ρ/ω가 `.norm(dim=-1)`/코사인 유사도로 계산되는 공간을 코드로 추적 —
  `action_space`("joint"/"ee")와 무관하게 항상 "v_t가 사는 공간"(표준 정규화 + 선택적
  `per_timestep_norm` 스케일이 적용된 공간)에서 계산됨을 확인(조사만, RoCo 프리셋은
  `lac_mode="none"`이라 수정 보류).
- 기존 전체 테스트 스위트(137개, `test_ard*`/`test_lac*`/`test_closed_loop*`/`test_judge*`
  등) 회귀 없음.

**못 함 (범위 밖, 구조/합성 데이터 검증만 — README의 "검증됨 vs 미검증" 관례)**:
- `per_timestep_norm_a`/`_s`를 실제 데이터로 피팅하지 못함(`scripts/fit_timestep_norm.py`
  자체는 작성했지만 실행 불가) — 프리셋은 항등값(a=1,s=0)으로 둠.
- 실제 SmolVLM2 가중치, 실제 로봇/시뮬레이터, chunk_size=20/n_action_steps=5, 10fps/시연
  200개가 실제 RoCo 과제에서 적절한 값인지 전혀 실측하지 않음. "대부분 왼팔만 사용"(→
  `role_map`의 근거)도 공개 설명 그대로이고 스키마처럼 직접 재확인한 건 아님.
- `ard_angle_dims`의 [-π,π) 경계가 표준 z-score 정규화 이후에도 정확한지(실제로는 그
  채널의 std로 나뉜 `2π/std` 경계여야 함) — std를 모델에 전달하지 않아 근사치로만 구현.
- `per_timestep_norm`+`lac_mode`를 동시에 켠 상태의 실제 효과(ρ/ω 임계값 재캘리브레이션
  필요 여부)는 분석만 했고 실측하지 않음 — RoCo 프리셋은 이 조합을 피함(`lac_mode="none"`).

```bash
pytest tests/test_configurable_ard.py tests/test_ard.py tests/test_ard_integration.py tests/test_lac.py tests/test_lac_closed_loop.py
```
