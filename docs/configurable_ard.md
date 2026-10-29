# 설정 가능한 ARD (Configurable ARD) — RoCo Challenge 프리셋

[← README로 돌아가기](../README.md)

ARD-VLA는 원래 OpenArm 16채널(팔당 8 DoF, 왼팔=Stabilizer 고정, chunk 50, pos_loss는 L2)
전제로 설계됐습니다. RoCo Challenge처럼 대회마다 로봇/액션 공간/우세 팔이 바뀌는 과제에
재사용하려고, 이 하드코딩을 config 손잡이 4개(`role_map`/`action_space`/`pos_loss_type`/
`per_timestep_norm`)와, 손실과는 완전히 독립적인 데이터 전처리 옵션(`angle_unwrap_dims`)
으로 뺐습니다. **이번 작업은 구조 구현 + 작은
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
| `angle_unwrap_dims` | `None` | 액션 벡터의 절대 채널 인덱스 리스트. ARD/pos_loss와 완전히 독립 — `use_ard` 여부와 무관하게 동작하는 **데이터 전처리** 옵션(아래 참고) |

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

### `angle_unwrap_dims` — 회전 채널의 2π 경계 불연속, pos_loss가 아니라 전처리에서 다룬다

**이전 설계(되돌림)**: 처음엔 메인 pos_loss(`F.mse_loss(u_t, v_t)` 등)에서 회전 채널의
`(u_t - v_t)` 차이를 `[-π, π)`로 감싸는 `ard_angle_dims`를 추가했었다. **틀린 설계였다**:
1. `u_t`/`v_t`는 raw 액션이 아니라 flow-matching **속도장**(`u_t = noise - actions`)이다
   — 각도의 주기성은 액션 "값"에 적용되는 물리적 성질이고, 속도장에는 그 주기성이 그대로
   옮겨가지 않는다. 속도장 오차를 `±π`로 접으면, 실제로는 정상적인(각도 래핑과 무관한)
   큰 오차까지 왜곡해서 깎아버린다.
2. 구현했던 `ard_angle_dims=[3, 4, 5]`(RoCo 프리셋)는 각 팔 블록 내 **상대** 인덱스라서
   `build_angle_channel_mask`가 왼팔 블록[3,4,5]과 오른팔 블록[10,11,12] 둘 다에 자동으로
   적용되긴 했지만(설계상 의도된 동작), 설정값 자체가 "절대 인덱스로 왼팔만 쓴 것처럼"
   오해하기 쉬운 표기였다.

그래서 `ard_angle_dims`/`apply_angle_aware_target`/`build_angle_channel_mask`를 전부
되돌렸다(`ard.py`/`modeling_smolvla.py`/`configuration_smolvla.py`, PROGRESS.md 참고) —
pos_loss는 다시 순수하게 `F.*_loss(u_t, v_t)`만 계산한다.

**새 설계**: 각도 문제를 **메인 손실이 아니라 데이터 전처리 단계**에서 다룬다 —
`angle_unwrap_dims`(액션 벡터의 **절대** 채널 인덱스 리스트, 예: RoCo의 양팔 rx,ry,rz =
`[3, 4, 5, 10, 11, 12]`)를 설정하면 `processor_smolvla.py`의 두 `ProcessorStep`이 켜진다:

- **`AngleUnwrapProcessorStep`**(학습 전, `NormalizerProcessorStep`보다 먼저): 액션 청크의
  raw 값을 시간축(청크 내부)으로 연속으로 편다 — `np.unwrap`과 동일한 원리로, 연속된 두
  프레임의 `wrap_to_pi(차이)`를 누적합해서 재구성한다. `LeRobotDataset`의
  `delta_timestamps` 윈도우는 에피소드 경계를 넘지 않으므로, "청크 안에서만" 펴는 것으로
  "에피소드 경계를 넘어 펴지 않는다"는 요구를 자동으로 만족한다.
- **`AngleRewrapProcessorStep`**(추론 후, 파이프라인의 마지막 단계): 로봇에 보내기 직전,
  연속 표현으로 다루던 채널을 다시 `[-π, π)`로 감싼다(`AngleUnwrapProcessorStep`의 정확한
  역연산).

둘 다 `angle_unwrap_dims=None`(기본)이면 파이프라인에 들어가지도 않아 bit-exact다. 둘 다
`NormalizerProcessorStep`처럼 배치에 액션이 없으면(실제 로봇 추론 입력) 에러 없이 조용히
통과시킨다.

**이 전처리가 올바른지는 rx/ry/rz가 어떤 회전 표현인지에 전적으로 달려 있다** — 아래
"오일러각 vs 회전벡터" 참고. 회전벡터(axis-angle)라면 세 채널을 독립적으로 unwrap하는 건
수학적으로 틀렸다(세 채널이 "하나의 회전"을 이루는 벡터라서, 축별 주기성이라는 가정 자체가
성립하지 않는다).

### 오일러각 vs 회전벡터(axis-angle) — rx/ry/rz는 어느 쪽인가 (미확인)

RoCo 스키마 자체는 채널 이름만 알려줄 뿐 회전 표현 방식을 명시하지 않는다. 이 레포/코드
안에서 직접적인 증거는 찾지 못했지만, 참고할 만한 간접 근거 두 가지가 있다:

1. **이 레포가 vendor로 들고 있는 LeRobot 자체의 선례**: `third_party/lerobot/src/lerobot/
   processor/env_processor.py`의 `LiberoProcessorStep`이 LIBERO 환경의 엔드이펙터
   회전을 쿼터니언에서 **axis-angle(회전벡터, 3D)**로 변환해서 state에 쓴다(`_quat2axisangle`,
   `third_party/lerobot/src/lerobot/utils/rotation.py`에 `Rotation.as_rotvec`/`from_rotvec`도
   있음) — LeRobot 생태계에서 "엔드이펙터 3D 회전"을 다룰 때 axis-angle이 흔한 선택임을
   보여준다.
2. **산업용 로봇의 통상적인 TCP pose 관례**: Universal Robots 등에서 "x, y, z, rx, ry, rz"
   형태의 TCP pose 표기는 관례적으로 rx,ry,rz가 **회전벡터**(방향=회전축, 크기=회전각)를
   가리키는 경우가 많다 — "rx,ry,rz"라는 이름 자체가 오일러각(보통 roll/pitch/yaw로
   부름)보다 회전벡터 쪽 관례에 더 가깝다.

**결론: 확정적이지 않지만, rx/ry/rz가 오일러각보다 회전벡터일 가능성을 시사하는 간접
근거가 있다.** 회전벡터라면 `angle_unwrap_dims`를 이 채널들에 적용하는 것은 **부적절**
하다 — RoCo 프리셋은 그래서 이 옵션을 꺼둔 채로 둔다. `scripts/check_rotation_range.py`가
실제 데이터의 삼중값(rx,ry,rz) 노름 통계를 출력해 참고용 힌트를 주지만(노름이 거의 항상
`≤ π`면 회전벡터 쪽에 더 가까움), **최종 판단은 RoCo 데이터 생성 코드/스키마 문서를 직접
확인해야 한다** — 이 레포에서는 그 소스에 접근하지 못했다.

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
- 채널이 관절이 아니라 엔드이펙터 목표 pose라서 `action_space="ee"`.
- rx/ry/rz 세 채널이 오일러각인지 회전벡터(axis-angle)인지는 스키마 자체로는 알 수
  없습니다 — **미확인**(위 "오일러각 vs 회전벡터" 참고).
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
| `angle_unwrap_dims` | *(미설정, 꺼짐)* | rx/ry/rz가 오일러각인지 회전벡터인지 미확인이라 일단 꺼둠 — `scripts/check_rotation_range.py`로 확인 후 켤지 결정 |
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

## 요구사항 6 현황: `freeze_vision_encoder` / 액션 expert 전체 학습 / VLM LoRA (보고, 미수정)

코드를 추적해서 확인한 현재 상태 — 이번 작업에서 아무것도 바꾸지 않았다(사용자가
"보고만" 요청):

- **`freeze_vision_encoder: bool = True`**(기본) — 존재함. 다만 **또 다른 기본값
  `train_expert_only: bool = True`가 이미 VLM 전체(비전 인코더 포함)를 얼린다**
  (`smolvlm_with_expert.py`의 `set_requires_grad()`: `train_expert_only=True`면
  `self.vlm`(비전+언어 전체)의 `requires_grad`를 전부 `False`로 설정) — 그래서
  `freeze_vision_encoder`는 `train_expert_only=False`일 때만(즉 VLM 일부를 추가로 학습할
  때만) 독립적인 효과가 있다. 기본 조합(둘 다 기본값)에서는 중복이다.
- **"액션 expert 전체 학습"** — **기본적으로 이미 그렇다.** `set_requires_grad()`는
  `lm_expert`(액션 expert, VLM과는 별개의 가중치)의 파라미터를 얼리지 않는다(분산학습용
  미사용 `lm_head`만 예외) — PEFT를 쓰지 않는 기본 경로에서는 전체 파인튜닝이다.
- **"VLM은 LoRA"** — **요청대로 구현돼 있지 않다.** 이게 이번 조사의 핵심 발견:
  - `modeling_smolvla.py`의 `_get_default_peft_targets()`가 잡는 타겟은
    `model.vlm_with_expert.lm_expert.*.(q|v)_proj` + `state_proj`/`action_*_proj` 들이다
    — **"lm_expert"는 VLM이 아니라 액션 expert다**(`smolvlm_with_expert.py`에서
    `self.vlm`과 `self.lm_expert`는 완전히 별개의 모델 인스턴스로 생성됨). 즉 PEFT를
    켜면 LoRA가 VLM이 아니라 **액션 expert**의 attention에 붙고, VLM(`self.vlm`) 자체는
    `_get_default_peft_targets()`의 타겟 목록에 전혀 없다.
  - 게다가 PEFT 진입점(`PreTrainedPolicy.wrap_with_peft()`, `pretrained.py`)이 **이 레포의
    `scripts/train_ard.py`에서 전혀 호출되지 않는다** — `--use-peft` 같은 CLI 플래그가
    없다. 즉 PEFT/LoRA 메커니즘 자체는 업스트림 LeRobot에서 상속받아 코드상 존재하지만,
    이 레포의 학습 스크립트로는 지금 당장 켤 방법이 없다.
  - **결론**: "VLM은 LoRA"를 실제로 구현하려면 (a) `_get_default_peft_targets()`에
    `model.vlm_with_expert.vlm.*`(실제 VLM 모듈) 쪽 타겟을 추가하고, (b)
    `train_ard.py`에 PEFT를 켜는 CLI 경로를 새로 추가해야 한다 — 둘 다 아직 안 돼 있다.
    이번 작업 범위(RoCo 프리셋)에서는 수정하지 않았다(보고만 요청받음).

## 조사 보고: 기존 기능들이 별도 브랜치에 있는지 (quantization/GradNorm/token_pruning/
gradient_checkpointing/profile_memory.py/layer_importance.py)

`git branch -a`로 전체 브랜치를 확인하고(`claude/smolvla-research-setup-x29mi7`,
`closed-loop-structure`, `lac-inference`, `lac-closed-loop`, `main`, 그리고 현재
`configurable-ard`), `git log configurable-ard..<branch>`로 각 브랜치에 `configurable-ard`
에 없는 커밋이 있는지 전부 확인했다. **결과: 전부 0개** — 즉 다섯 기능 전부 이미
`configurable-ard`(가 상속한 `lac-closed-loop` ← `lac-inference` ← `closed-loop-structure`
← `main`의 머지 체인)에 **이미 머지돼 있다.** 합칠 별도 브랜치가 없다. (`main`만
`configurable-ard`에 없는 커밋 1개가 있었지만, 확인해보니 2026년 8월의 아주 오래된 초기
PR 병합 커밋이고 — 그 이후 모든 작업이 다른 브랜치 경로로 이어져서 트리 내용 자체는
`configurable-ard`가 전부 상위호환함. 실질적으로 합칠 고유 작업 없음.)

항목별 현재 위치와 이번 작업과의 충돌 가능성:

| 기능 | 위치 | configurable-ard와 충돌 가능성 |
|---|---|---|
| quantization | **존재하지 않음**(레포 전체 grep 0건) | 없음(아예 없어서) |
| GradNorm | `ard.py`의 `GradNormLambdas`, `use_gradnorm`/`gradnorm_alpha`/`gradnorm_lr` | smooth/force/traj 레귤러라이저 가중치만 건드림(메인 pos_loss와 무관). `per_timestep_norm`을 켜면 x0_hat 스케일이 바뀌어 GradNorm이 균형 잡는 그래디언트 norm 크기도 같이 바뀔 수 있음(둘 다 켤 때 참고할 사항, 버그는 아님) |
| token_pruning | `token_pruning.py`, `use_token_pruning`/`token_pruning_k_final`/`token_pruning_k_key` | 비전 토큰 개수만 건드림 — 액션/role_map/pos_loss/정규화와 완전히 무관. 충돌 없음 |
| gradient_checkpointing | `smolvlm_with_expert.py`의 `gradient_checkpointing_enable/disable()` | 백본 forward의 메모리 레이아웃만 바꿈(수치적으로 투명해야 함), per_timestep_norm/pos_loss_type은 전부 그 바깥(VLM forward 전/후)에서 계산됨. 충돌 없음 |
| `scripts/profile_memory.py` | 이미 `scripts/`에 존재, `--ard-arm-dim` 등 CLI로 전달 | role_map/action_space/pos_loss_type/per_timestep_norm/angle_unwrap_dims를 전혀 모름(그 기능들을 테스트하지 않음) — 충돌은 없지만 이번에 추가한 옵션들을 켜고 메모리를 프로파일하고 싶으면 CLI 인자를 추가해야 함(아직 안 함) |
| `scripts/layer_importance.py` | 이미 `scripts/`에 존재 | ARD/액션 관련 설정을 전혀 참조하지 않음 — 충돌 없음 |

## 새/수정 파일

- `configuration_smolvla.py` — `role_map`/`action_space`/`pos_loss_type`/`per_timestep_norm`
  (+`_a`/`_s`)/`angle_unwrap_dims` 필드와 `__post_init__`/`validate_features()` 검증·파생
  로직.
- `ard.py` — `compute_timestep_norm_sigma`/`apply_timestep_norm`/`invert_timestep_norm`
  (순수 함수 3개, 패딩 채널은 identity로 채움). (`wrap_to_pi`/`build_angle_channel_mask`/
  `apply_angle_aware_target`은 추가했다가 되돌림 — 위 "되돌림" 참고.)
- `modeling_smolvla.py` — `F.mse_loss` 호출을 `pos_loss_type` 3분기로 교체,
  `forward()`/`_denoise_loop()` 경계에서 `per_timestep_norm` 스케일/역스케일.
- `processor_smolvla.py` — `AngleUnwrapProcessorStep`/`AngleRewrapProcessorStep`(+ 내부
  `_wrap_to_pi`/`_unwrap_angles_along_time` 순수 함수), `angle_unwrap_dims` 설정 시에만
  파이프라인에 삽입.
- `scripts/fit_timestep_norm.py` (신규) — 데이터셋에서 σ(t)의 a/s 피팅.
- `scripts/check_rotation_range.py` (신규) — 회전 채널 ±π 경계 도약/삼중값 노름 점검.
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
- RoCo 프리셋 yaml이 유효한 `SmolVLAConfig`로 로드되는지(derived 값 포함), 그리고 그 값
  그대로(14차원=7+7) 작은 합성 모델에서 몇 스텝 학습(loss.backward())/추론
  (predict_action_chunk)이 정상 동작하는지.
- `AngleUnwrapProcessorStep`이 ±π 경계를 넘는 raw 궤적을 실제로 연속으로 펴는지(지정 안 한
  채널은 그대로), `AngleRewrapProcessorStep`이 그 역연산인지(round-trip), 둘 다 액션이
  없는 입력(실제 추론 전처리 입력)을 에러 없이 통과시키는지. `angle_unwrap_dims=None`
  (기본)이면 `make_smolvla_pre_post_processors()`가 만드는 실제 파이프라인에 두
  ProcessorStep이 전혀 들어가지 않고, 설정하면 각각 정확히 하나씩 들어가는지(토크나이저
  로드만 몽키패치로 피하고 나머지는 실제 파이프라인 구성 함수를 직접 호출해 확인).
  잘못된 `angle_unwrap_dims`(빈 리스트/중복/범위 밖) 거부.
- `NormalizerProcessorStep`이 실제로 차원별 독립 통계(`(tensor-mean)/std`, 코드 직접 확인)를
  쓰는지 — 위치(m)/회전(rad) 등 단위가 달라도 각 채널이 자기 자신의 mean/std로만 정규화되므로
  문제없음(수정 불필요, 코드 확인만).
- LAC(`lac.py`)의 ρ/ω가 `.norm(dim=-1)`/코사인 유사도로 계산되는 공간을 코드로 추적 —
  `action_space`("joint"/"ee")와 무관하게 항상 "v_t가 사는 공간"(표준 정규화 + 선택적
  `per_timestep_norm` 스케일이 적용된 공간)에서 계산됨을 확인(조사만, RoCo 프리셋은
  `lac_mode="none"`이라 수정 보류).
- 기존 전체 테스트 스위트(136개, `test_ard*`/`test_lac*`/`test_closed_loop*`/`test_judge*`
  등) 회귀 없음.

**못 함 (범위 밖, 구조/합성 데이터 검증만 — README의 "검증됨 vs 미검증" 관례)**:
- `per_timestep_norm_a`/`_s`를 실제 데이터로 피팅하지 못함(`scripts/fit_timestep_norm.py`
  자체는 작성했지만 실행 불가) — 프리셋은 항등값(a=1,s=0)으로 둠.
- 실제 SmolVLM2 가중치, 실제 로봇/시뮬레이터, chunk_size=20/n_action_steps=5, 10fps/시연
  200개가 실제 RoCo 과제에서 적절한 값인지 전혀 실측하지 않음. "대부분 왼팔만 사용"(→
  `role_map`의 근거)도 공개 설명 그대로이고 스키마처럼 직접 재확인한 건 아님.
- rx/ry/rz가 오일러각인지 회전벡터인지 **미확인**(간접 근거만 있음, 위 참고) —
  `scripts/check_rotation_range.py`도 HF Hub 차단으로 실행 못 해봄. 그래서
  `angle_unwrap_dims`는 RoCo 프리셋에서 꺼둔 채로 둠.
- `per_timestep_norm`+`lac_mode`를 동시에 켠 상태의 실제 효과(ρ/ω 임계값 재캘리브레이션
  필요 여부)는 분석만 했고 실측하지 않음 — RoCo 프리셋은 이 조합을 피함(`lac_mode="none"`).

```bash
pytest tests/test_configurable_ard.py tests/test_ard.py tests/test_ard_integration.py tests/test_lac.py tests/test_lac_closed_loop.py
```
