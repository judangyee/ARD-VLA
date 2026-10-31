# 복구 데이터 생성 파이프라인 — 설계 문서

[← README로 돌아가기](../README.md)

> **이 문서는 설계만 다룬다. 구현 코드는 이번 작업에 포함하지 않는다.**
> 표기 규칙: **확인됨**(코드를 직접 읽거나 실행해서 확인한 사실) /
> **추정**(근거는 있지만 직접 확인하지 않음) / **미확인**(확인할 방법이
> 아직 없거나 범위 밖). 모든 절에서 이 셋을 구분해서 표기한다.

## 0. 요약

RoCo 시뮬 채점의 40%가 에러 복구라서, "정상 시연만" 모은 데이터로는 한 번
어긋나면 정책이 복구하는 법을 배우지 못한다. 이 문서는 **정상 시연 중간에
실패를 의도적으로 주입하고, 그 뒤 복구 동작을 이어 붙여 기록하는** 파이프라인을
설계한다 — RoCo 전용 하드코딩 없이, 과제 독립적인 구조로.

**핵심 결론(미리 요약)**:

1. 지금 쓸 수 있는 ARD-Gen(`/home/user/ard-gen/ard-gen`)은 **물리 시뮬레이션/
   양팔 동시 실행/씬 랜덤화는 이미 꽤 성숙하지만, RoCo가 요구하는 데이터
   스키마(카메라, 44차원 state, 14차원 action, LeRobot v3.0)와는 전혀 다른
   더 단순한 스키마(카메라 없음, 3~7차원 state, 1~3차원 action, v2.1 근사
   npz)로만 되어 있다**(아래 1절, 전부 코드로 직접 확인함). 실패 주입/복구
   개념 자체도 전혀 없다.
2. 시뮬 백엔드(MuJoCo/Isaac) 추상화는 **지금 존재하지 않는다** — 태스크
   추상화(`BaseTaskEnv`)는 있지만 그 구현체들이 MuJoCo API를 직접 호출한다.
   2절에서 새 경계를 제안한다.
3. ARD-VLA(학습 쪽)는 **프레임별 loss_weight를 지원하지 않는다** — 다만
   거의 같은 모양의 메커니즘(`actions_is_pad` 마스킹)이 이미 있고, 그마저도
   **키 이름이 실제로 생성되는 키와 달라 지금 항상 비활성 상태라는 걸
   확인했다**(4·6절에서 다룸). 새 메커니즘은 이 버그를 반복하지 않아야 한다.

## 1. 현황 조사: ARD-Gen에서 뭘 재사용할 수 있나

### 1.1 재사용 가능한 것 (확인됨)

| 모듈 | 위치 | 재사용 가치 |
|---|---|---|
| 태스크 레지스트리 | `sim/task_registry.py` (`TASK_REGISTRY`, `TaskConfig`, `BaseTaskEnv`) | `--task` 하나로 여러 태스크를 다형적으로 다루는 구조 자체는 그대로 쓸 수 있다. 다만 지금은 "게인 탐색" 중심 API(`gains_from_vector`, CMA-ES 전용 필드)라 복구 파이프라인에 맞게 얇은 레이어를 더 얹어야 한다(2절). |
| 공유 씬 샘플링 | `pipeline/scene_sampler.py` | friction/clearance/object offset 무작위화 패턴(1단계)은 구조 그대로 재사용 가능. 다만 지금은 **물리 파라미터만** 다룬다 — 시각 랜덤화는 전혀 없다(5절). |
| 양팔 동시 실행 설계 | `sim/stabilizer.py`, PIPELINE.md 3단계 | **왼팔(고정)+오른팔(조작) 동시 시뮬레이션**이라는 핵심 아이디어가 이미 실측으로 검증돼 있다(Stabilizer 유무로 성공률이 71%→38%까지 벌어짐 등, PIPELINE.md 3단계). 4절(양팔 시연)의 기반이 된다. |
| "접근→정착→후퇴→최종정착" 단계 머신 | `sim/tacker_openarm_env.py` | tacker 태스크에 이미 구현된 이 4단계 패턴(`release_grip_after` 옵션 포함)이 **"실패 유형: 잘못된 부품 삽입"의 복구 동작(빼기→재접근)과 구조적으로 거의 동일**하다(3절). |
| Jacobian 기반 computed-torque 제어 | `sim/peg_in_hole_bimanual_openarm_sim.py` | OpenArm 7-DOF 양팔의 실제 제어 코드 — "위치 어긋남" 실패의 복구(더 큰 오차로부터 재수렴)는 **이 컨트롤러를 그대로 다시 돌리는 것**과 같다(3절, 추가 코드 거의 불필요). |
| `qfrc_constraint` 기반 힘 로깅 | `sim/tacker_env.py`/`sim/tacker_openarm_env.py` | freejoint 바디의 XML `<sensor>` force/torque가 **항상 0이 되는 구조적 결함을 이미 겪고 고친 경험**(PIPELINE.md, "실측 버그 발견" 절) — 복구 데이터의 `failure_type`/`phase` 판정에 힘 신호가 필요하면 이 패턴을 그대로 써야 한다. |
| Judge/Selector 실패 분류 헤드 | ARD-VLA `ard.py`/`decision_logic.py` (`judge_num_failure_types`, `judge_failure_type`, `failure_type_to_recovery`) | **학습 쪽에 이미 "실패 유형 정수 ID" 라벨을 받는 자리가 있다**(`configuration_smolvla.py:275`, `modeling_smolvla.py:678`). 복구 데이터의 `failure_type` 라벨 taxonomy를 여기 맞춰 설계하면 Judge 헤드 학습에도 바로 쓸 수 있다. |
| `mujoco.Renderer` 오프스크린 렌더링 | `render_episode.py`/`render_bimanual.py` 등 | 임의 해상도(`--width`/`--height`)로 여러 카메라를 동시에 오프스크린 렌더링하는 코드가 이미 있다 — 다만 **지금은 사후 시각 확인용 mp4를 만들 때만 쓰이고, 데이터셋 저장 경로에는 전혀 연결돼 있지 않다**(아래 1.2). |

### 1.2 부족한 것 (확인됨 — 코드로 직접 확인)

| 항목 | RoCo 목표 | ARD-Gen 현재 | 근거 |
|---|---|---|---|
| 데이터셋 포맷 | LeRobot **v3.0**(parquet + 비디오) | **"v2.1-approx"** 문자열을 자체적으로 박은 **npz 근사**("동일이 아니라 호환"이라고 스스로 명시) | `pipeline/to_lerobot.py:56` (`CODEBASE_VERSION = "v2.1-approx"`), 같은 파일 상단 docstring. 이 레포(`ARD-VLA`)가 실제로 쓰는 `LeRobotDataset`은 `CODEBASE_VERSION = "v3.0"`이라 **지금 ARD-Gen 출력은 그대로 로드되지 않는다**(`third_party/lerobot/.../lerobot_dataset.py:83`). |
| 카메라 | 3개(head/left_hand/right_hand), 240×320×3, 매 프레임 데이터셋에 저장 | **0개 — 데이터셋에 이미지가 전혀 없다** | `pipeline/to_lerobot.py:197`: `"video_path": None,  # 시뮬레이션이라 카메라 없음`. `mujoco.Renderer`는 있지만 `run_episode()`/`to_lerobot.py` 어디서도 호출되지 않는다(렌더 스크립트는 완성된 npz를 **사후에** 다시 읽어 mp4만 만든다). |
| fps | 10fps | **100fps**(제어 주기 dt=0.01s) | `pipeline/to_lerobot.py:55`, `sim/peg_in_hole_bimanual_openarm_sim.py:221-222`(`N_SUBSTEPS=5`, `timestep=0.002s` → dt=0.01s, 실측 확인). 10fps로 맞추려면 매 10번째 제어 스텝만 프레임으로 저장해야 한다(단순 10배 다운샘플, 확인됨 — 별도 리샘플링 불필요). |
| action 차원/의미 | 14차원, 팔당 [EE 절대 pose(xyz+rxryrz) 6 + 그리퍼 1], Euler XYZ rad, wrap 금지 | 태스크마다 **1~3차원**, 전부 `delta_x/y/z`(상대 위치 델타) 또는 `omega_command_rad_s`(스칼라) — 그리퍼/회전/절대값 전혀 없음 | `pipeline/to_lerobot.py:72-76`(`_ACTION_NAMES`). |
| state 차원/의미 | 44차원: EE pos+quat 14, 관절 위치 14, 관절 속도 14, 그리퍼 2 | 태스크마다 **4~7차원**, EE 위치(xyz)+손목회전 하나만(쿼터니언/관절위치/관절속도/그리퍼 없음) | `pipeline/to_lerobot.py:61-71`(`_STATE_NAMES`). |
| 프레임별 메타데이터 | `loss_weight`/`failure_type`/`phase`/`source` | **전혀 없음** — 에피소드 단위 스칼라(`success`, `insertion_depth`, `force_max`)와 5단계가 채우는 **에피소드 단위** `language`(dict)만 있다 | `pipeline/episode_io.py` 상단 docstring, `pipeline/to_lerobot.py:149-177`(프레임 배열은 `observation.state`/`action`/`episode_index`/`frame_index`/`timestamp`/`task_index` 6개뿐). |
| 실패 주입/복구 | 핵심 기능 | **없음** — PIPELINE.md 전체에 "실패 주입"/"복구" 개념 자체가 등장하지 않는다(필터링 단계는 실패 에피소드를 **버리기만** 한다, `pipeline/filter_episodes.py` 설명 — "실패 에피소드는 보관하지 않는다") | PIPELINE.md 4단계. |
| 시각 도메인 랜덤화 | 조명/색/텍스처/카메라 | **없음** — `scene_sampler.py`는 friction/clearance/위치 오프셋만 다룬다. XML 자산에 `<texture>`/`<material>` 자체가 없고(전부 단색 `rgba`), 조명은 `<light>` 2개가 전 태스크에서 고정값 | `pipeline/scene_sampler.py` 전체, `assets/peg_in_hole_bimanual_openarm.xml:151-152`(조명 2개 고정), 같은 파일에 `<texture>` 태그 grep 0건. |
| 프레임별 loss weight 학습 지원 | 필수 | **없음** — `scripts/train_ard.py`/`modeling_smolvla.py`에 `loss_weight`류 키를 읽는 코드가 전혀 없다(ARD 전용 scalar 가중치 `judge_loss_weight`류와는 다른 것 — 그건 하이퍼파라미터고, 데이터에서 오는 프레임별 가중치가 아니다) | `modeling_smolvla.py` grep 결과(6절에서 상세). |

### 1.3 양팔 지원 정도 (확인됨)

- **물리적으로는 이미 충분히 성숙하다**: 왼팔(Stabilizer, 고정/파지)과
  오른팔(Actuator, 조작)이 **같은 `MjModel`/`MjData`를 공유하며 매 스텝
  동시에** 시뮬레이션된다(`PIPELINE.md` 3단계 "4. 양팔 통합"). Stabilizer가
  없으면 Actuator 성공률이 떨어진다는 걸 여러 태스크에서 N=150~300 규모로
  반복 실측했다(peg_in_hole 71%→38%, tacker 100%→37.3% 등).
- **그러나 역할-팔 매핑이 RoCo와 반대다**: ARD-Gen의 기본 패턴은 **왼팔=
  Stabilizer, 오른팔=Actuator**다(`PIPELINE.md` 배경 절: "오른팔(Actuator):
  도구를 쥐고 미세조작", "왼팔(Stabilizer): 물체를 고정"). 반면
  `docs/configurable_ard.md`가 확인한 RoCo 공식 데이터는 **왼팔=Actuator,
  오른팔=Stabilizer**(`role_map={"stabilizer": "right", "actuator": "left"}`,
  "대부분 왼팔만 사용"하는 공개 스펙 때문). ARD-VLA 쪽은 `role_map`으로 이미
  이 반전을 흡수하므로(기존에 처리된 문제), **복구 데이터 생성 쪽에서도
  팔-역할 매핑을 config로 노출해야 한다** — 하드코딩하면 RoCo와 또 반대로
  나올 위험이 있다.
- **OpenArm 7-DOF 양팔 전용 코드는 1개 태스크(peg_in_hole)뿐**:
  `cap_twist`/일반 `tacker`는 여전히 "가상 3-슬라이드 EE Stabilizer"(IK 불필요,
  실제 팔 기구학 없음) 패턴이다. `tacker_openarm`/`insertion`만 OpenArm 양팔
  실제 기구학(7-DOF Jacobian IK + computed-torque)을 쓴다. 즉 **"실제 OpenArm
  두 팔이 둘 다 진짜 팔로 동시에 동작하는" 코드 경로는 지금 2개 태스크로만
  검증돼 있다** — 전부 수치는 확인됐지만, 그 두 개를 넘어서는 일반성은
  **추정**이다(설계상 태스크 무관화는 돼 있지만, 세 번째 사례로 실제
  검증한 적은 없음).

## 2. 아키텍처: 시뮬 백엔드를 인터페이스로 분리

### 2.1 지금 상태 (확인됨): 백엔드 추상화가 없다

`sim/base_task_env.py`의 `BaseTaskEnv`는 **태스크 추상화**(reset/step/
compute_reward/is_success)만 제공하고, 그 구현체(`PegInHoleOpenArmEnv` 등)는
내부에서 `mujoco.mj_step`/`mujoco.mj_jacSite`/`mujoco.mj_jac`/
`mujoco.Renderer`를 **직접** 호출한다(확인됨, 예:
`sim/peg_in_hole_bimanual_openarm_sim.py:930`의 `mujoco.mj_step(...)`). Isaac
Sim으로 바꾸려면 이 구현체들을 전부 다시 써야 한다 — 지금 구조로는 "백엔드만
바꿔 끼우기"가 안 된다.

### 2.2 제안: `SimBackend` 계층을 새로 둔다

```
┌─────────────────────────────────────────────┐
│ 파이프라인 (씬 샘플링 / 실패 주입 / 필터링 / │  ← 백엔드 무관 (안 바뀜)
│ 언어 라벨링 / LeRobot v3.0 writer)            │
├─────────────────────────────────────────────┤
│ TaskLogic (reset 조건 / 성공 판정 / reward)   │  ← 백엔드 무관 (안 바뀜,
│   = 지금의 BaseTaskEnv 서브클래스, 단 물리    │     지금 BaseTaskEnv와 거의
│   API를 직접 안 부르고 SimBackend를 통해서만  │     동일한 모습이지만 내부
├─────────────────────────────────────────────┤     구현만 백엔드 경유로 바뀜)
│ FaultInjector (실패 주입/복구 판정, 3절)      │  ← 백엔드 무관 (안 바뀜,
│   = SimBackend의 teleport/force/weld API만    │     SimBackend 추상 위에서만
│   사용, MuJoCo/Isaac API를 직접 모름          │     동작하도록 새로 설계)
├─────────────────────────────────────────────┤
│ 컨트롤러 (admittance / computed-torque IK)    │  ← **바뀜** (Jacobian 계산,
│   = 지금 sim/*_sim.py의 핵심부                │     질량행렬 등 물리 엔진별
│   mj_jacSite/mj_fullM을 SimBackend 메서드로   │     API가 근본적으로 다름)
├─────────────────────────────────────────────┤
│ SimBackend (신규 인터페이스)                  │  ← 교체 지점
│   MuJoCoBackend(지금) / IsaacBackend(나중)    │
└─────────────────────────────────────────────┘
```

**`SimBackend`가 제공해야 하는 최소 메서드 (제안, 구현 아님)**:

| 메서드 | 입출력 | MuJoCo 구현(지금 코드에 이미 있는 것) |
|---|---|---|
| `step(ctrl: dict) -> None` | 제어 신호 적용 후 한 틱 진행 | `mujoco.mj_step(model, data, nstep=N_SUBSTEPS)` |
| `get_body_pose(name) -> (pos, quat)` | 월드 좌표 | `data.xpos[body_id]`/`data.xquat[body_id]` |
| `set_body_pose(name, pos, quat)` | **teleport**(실패 주입용) | freejoint body의 `qpos` 직접 덮어쓰기(이미 reset()의 IK 루프에서 쓰는 패턴, "FK로 직접 qpos를 써야 한다" — `sim/peg_in_hole_bimanual_openarm_sim.py` 주석) |
| `apply_external_force(body, force, torque, duration)` | 외력(드랍 시뮬레이션용) | `data.xfrc_applied[body_id]` 또는 tacker의 velocity-kick 패턴(`qvel` 직접 수정) |
| `set_equality_active(name, active: bool)` | weld 등 제약 켜고 끄기(그립 해제/재파지) | `model.eq_active0`/`data.eq_active` — Stabilizer weld가 이미 이 방식으로 동작(`sim/stabilizer.py`) |
| `get_jacobian(point, body) -> ndarray` | 컨트롤러용 | `mujoco.mj_jacSite`/`mujoco.mj_jac` |
| `get_mass_matrix() -> ndarray` | computed-torque용 | `mujoco.mj_fullM` |
| `get_contact_force(body_a, body_b) -> ndarray` | 접촉력(완료 판정/force 신호) | `mujoco.mj_contactForce` (tacker가 이미 freejoint 센서 버그를 피해 쓰는 `qfrc_constraint` 패턴도 여기 포함) |
| `render(camera, width, height) -> ndarray(H,W,3)` | 카메라 프레임 | `mujoco.Renderer(...).render()` |
| `get_sensor(name) -> ndarray` | F/T 등 | `data.sensor(name).data` |

Isaac Sim 쪽 구현(`IsaacBackend`)은 **API는 미확인**(아래 "미해결 질문"
참고)이라 지금 만들 수 없다 — 다만 위 표의 메서드 시그니처는 물리 엔진에
의존하지 않는 추상 수준으로 골랐으므로(좌표/쿼터니언/Jacobian/렌더는 두
엔진 다 갖는 일반적 개념), **Isaac 쪽이 정확히 이 11개 메서드만 채우면
위 세 계층(파이프라인/TaskLogic/FaultInjector)은 한 줄도 안 바뀐다**는 게
이 설계의 목표다.

### 2.3 바뀌는 것 / 안 바뀌는 것 요약

| 백엔드를 MuJoCo→Isaac으로 바꿀 때 | 바뀜 / 안 바뀜 |
|---|---|
| `SimBackend` 구현체 | **바뀜**(새로 작성) |
| 컨트롤러(admittance/computed-torque)의 Jacobian·질량행렬 호출부 | **바뀜**(`SimBackend` 메서드 호출로 이미 교체돼 있었다면 호출 코드 자체는 안 바뀌지만, 수치적 거동은 물리 엔진 차이로 달라질 수 있음 — 7절 참고) |
| XML 자산(MJCF) | **바뀜**(Isaac은 USD/URDF 계열 — 지오메트리/관성/액추에이터를 다시 옮겨야 함, 손으로 맞춘 home pose/grasp anchor 값도 재검증 필요할 가능성 높음) |
| `TaskConfig`/`TASK_REGISTRY`/`tasks/*.yaml` 스키마 | **안 바뀜** |
| `scene_sampler.py`의 scene_config 스키마(friction/clearance/offset 등 "물리적 의미") | **안 바뀜**(수치 범위는 재보정 필요할 수 있음 — 엔진마다 마찰/접촉 모델이 다름) |
| `FaultInjector`의 실패 유형 정의/주입 타이밍 로직(3절) | **안 바뀜**(전부 `SimBackend` 메서드로만 구현되므로) |
| 성공/보상 판정(`compute_reward`/`is_success`) | **안 바뀜**(기하학적 양 — 거리/깊이 — 으로 정의되고, 그 값은 `SimBackend.get_body_pose` 등으로만 읽으므로) |
| LeRobot v3.0 writer, 언어 라벨링, 필터링 파이프라인 | **안 바뀜** |
| ARD-VLA 학습 코드(`scripts/train_ard.py`, `processor_smolvla.py`) | **안 바뀜**(데이터가 이미 LeRobot v3.0 스키마로 통일돼 있으므로 백엔드를 몰라도 됨 — 이게 이 설계의 핵심 목적이다) |

## 3. 실패 유형 분류

세 유형 모두 **"정상 시연의 진행률(progress fraction) t ∈ (0.2, 0.8)
구간에서 무작위 주입"** 원칙을 쓴다 — 절대 스텝 수가 아니라 "그 에피소드
나머지 길이 대비 비율"로 정하면 태스크마다 길이가 달라도(peg_in_hole
수백 스텝, tacker 수십 스텝 등) 같은 로직을 재사용할 수 있다(과제 독립적
설계 원칙). 너무 이르면(t<0.2) "정상 진행" 구간이 짧아 라벨링 가치가 적고,
너무 늦으면(t>0.8) 복구할 시간/공간이 부족해진다 — 이 경계값 0.2/0.8은
**제안값(미검증)**이다.

| 실패 유형 | 주입 시점/방법 (제안, `SimBackend` 메서드로 구현) | 복구 전문가 동작 정의 | 재사용 가능한 기존 코드 | 신규 구현 비용 |
|---|---|---|---|---|
| **부품 떨어뜨림**(part_dropped) | 파지 중 `set_equality_active(grasp_weld, False)` + `apply_external_force`로 가벼운 교란(순수 weld 해제만으로는 관성 때문에 안 떨어질 수 있음, 추정) | ① 팔을 물체 쪽으로 재접근(`run_approach_phase()` 패턴 재사용) ② 재파지(weld 재활성화) ③ 중단됐던 지점부터 원래 태스크 재개 | `sim/stabilizer.py`의 approach/weld 로직(확인됨, 왼팔 Stabilizer가 이미 이 패턴으로 "접근→weld" 동작) | **낮음** — 기존 approach 로직을 Actuator 쪽에도 그대로 적용 |
| **위치 어긋남**(misalignment) | 파지는 유지한 채 `set_body_pose`로 물체(또는 팔이 쥔 대상)를 정상 오프셋 범위의 2~3배(제안값, 미검증) 바깥으로 teleport | 기존 admittance/computed-torque 컨트롤러를 **그대로** 다시 돌려서(목표는 그대로, 현재 오차만 커짐) 수렴 | `sim/peg_in_hole_bimanual_openarm_sim.py`의 제어 루프 자체(확인됨 — 이 컨트롤러는 원래부터 "오차에서 수렴"하도록 설계됨, `evaluate_generalization.py`가 이미 다양한 오프셋에서의 수렴을 검증) | **거의 0** — 새 컨트롤러 불필요, 주입 로직만 추가 |
| **잘못된 부품 삽입**(wrong_part_insertion) | Actuator가 N스텝 동안 의도적으로 "틀린 목표"(예: 비어있는 보조 소켓)를 추종하도록 목표 치환 | ① 빼기(retract, 목표를 현 위치보다 위로) ② 올바른 목표로 재접근 | `sim/tacker_openarm_env.py`의 "접근→정착→후퇴→최종정착" 4단계(확인됨, `release_grip_after` 옵션까지 이미 있음) | **높음** — 이 유형만 "틀린 목표가 될 다른 소켓/구멍"이 씬에 물리적으로 존재해야 하는데, 지금 어떤 태스크 XML에도 그런 2번째 타겟이 없음(확인됨, 전부 단일 목표 구조). 새 지오메트리 설계 필요 |

**공통 설계 노트**:
- 세 유형 다 `FaultInjector.inject(backend, fault_type, scene_state) ->
  injected_state`처럼 **`SimBackend` 메서드만 사용**하도록 짜면(2절),
  특정 태스크의 물체 이름/geom 이름에 의존하지 않고 `tasks/*.yaml`에 "주입
  대상 body 이름"만 선언하는 방식으로 과제 독립화할 수 있다(제안).
- `failure_type`은 정수 ID로 기록해서 ARD-VLA의 `judge_failure_type`
  배치 키(확인됨, `modeling_smolvla.py:678`)와 taxonomy를 맞춘다 — 정상
  구간(실패 없음)은 `-100`(PyTorch `CrossEntropyLoss`의 `ignore_index`
  관례, 확인됨)으로 채운다.
- "복구 성공"의 판정은 `is_success()`를 **실패 주입 이후에도 그대로**
  호출하면 된다(이미 태스크별로 구현돼 있으므로 새로 안 만들어도 됨) —
  다만 "복구에 쓴 시간"이 허용 스텝 예산을 넘기면 그 에피소드는 실패로
  버린다(기존 필터링 원칙 그대로 적용).

## 4. 양팔 시연 — 한 팔 고정 + 다른 팔 복구

**원칙: 점수에 도움이 될 때만 두 팔을 "다르게" 움직인다.** ARD-Gen은
이미 물리적으로 항상 양팔을 동시 시뮬레이션하므로(1.3절), "두 팔 다
시뮬레이션하는 비용"은 설계 선택이 아니라 이미 고정비다 — 진짜 선택은
"**Stabilizer가 복구 동작 중에 적극적으로 반응하게 만들 것인가, 그냥
기존 고정/파지 상태를 유지하게 둘 것인가**"다.

| 실패 유형 | Stabilizer가 반응해야 하는가 | 근거 |
|---|---|---|
| 부품 떨어뜨림 | **반응 필요할 수 있음** — 떨어진 물체가 Stabilizer가 고정하던 대상 자체이거나, 낙하 충격이 Stabilizer의 파지를 흔드는 경우 | Stabilizer 쪽에서 일어난 실패는 Stabilizer가 직접 복구해야 의미가 있다 |
| 위치 어긋남 | **불필요** — Actuator 쪽 컨트롤러 재수렴만으로 해결되는 유형(3절) | Stabilizer를 괜히 움직이면 "두 팔이 함께 복구한다"는 라벨이 사실과 안 맞는 가짜 신호가 됨 |
| 잘못된 부품 삽입 | **불필요**(원칙상) — Actuator의 목표 선택 오류이므로 Actuator 혼자 빼고 다시 접근 | 같은 이유 |

이 표가 보여주듯, **세 유형 중 둘은 "Stabilizer는 그냥 하던 대로 가만히
있는다"가 맞는 설계**다 — 억지로 양팔 협응 동작을 만들면 오히려 RoCo
점수(대부분 Actuator 쪽 과제 완수 여부로 매겨질 것으로 **추정**, 공식
채점 기준은 **미확인**)에 안 맞는 노이즈 라벨을 만들 위험이 있다. "부품
떨어뜨림" 유형 중 **Stabilizer가 쥔 물체 자체가 떨어지는 경우만** 명시적으로
양팔 복구 시연으로 설계한다.

## 5. 도메인 랜덤화

| 축 | 현재 상태 (확인됨) | 제안 |
|---|---|---|
| 조명 | `<light>` 2개, 전 태스크 고정값(`assets/peg_in_hole_bimanual_openarm.xml:151-152`) | 에피소드마다 `light_pos`/`light_diffuse`를 모델 컴파일 후 런타임에 작은 범위로 jitter(MuJoCo가 이 필드들을 런타임에 쓰기 가능하다는 건 **일반적으로 알려진 사실이지만 이 레포에서 실제로 해본 적은 없음 — 미검증**) |
| 색/텍스처 | `<texture>`/`<material>` 태그 0건, 전부 단색 `rgba`(확인됨) | 1차는 `geom_rgba`만 episode마다 무작위(텍스처 없이 색만 바꾸는 게 가장 싼 방법, 제안) — 실제 텍스처 매핑은 더 큰 작업이라 2차로 미룸 |
| 카메라 | 현재 `wide_cam`/`top_cam` 2개, 둘 다 **world 고정**(`mode="targetbody"`), 데이터셋에 저장도 안 됨(1.2절) | RoCo 스키마의 head/left_hand/right_hand 3개를 새로 정의해야 한다(새 작업) — head는 world 고정, left_hand/right_hand는 각 그리퍼 바디에 `mode="fixed"`로 장착(VX300s의 `wrist_cam` 패턴과 동일, `README.md`의 "검증하며 알아낸 것들" 참고). 카메라 pos/quat에도 작은 jitter 추가 가능(제안) |
| 부품 초기 위치 | friction/clearance/offset 랜덤화는 이미 있음(`scene_sampler.py`, 확인됨) | 그대로 재사용. 다만 이 랜덤화 범위가 "실패 주입" 없이도 이미 꽤 넓어서(예: peg_in_hole offset ±1.5cm), 실패 주입으로 만드는 교란(3절, 2~3배)과 겹치지 않도록 조정 필요(제안) |

**중요한 전제**: 위 표의 "제안" 항목은 전부 **MuJoCo 레벨**에서의 구현
방향이다. 2절의 `SimBackend.set_body_pose`/`render` 선에서 추상화되므로
Isaac으로 넘어갈 때 조명/텍스처/머티리얼 구현은 Isaac 쪽에서 다시 설계해야
한다(해당 엔진의 머티리얼 시스템이 다르므로, 7절 참고).

## 6. 검증

### 6.1 생성 데이터 자체의 정합성 (오프라인, 지금 가능)

| 검증 항목 | 방법 | 비용 |
|---|---|---|
| 리플레이 성공 여부 | 저장된 `action` 시퀀스를 같은 씬에 다시 적용해 `is_success()`가 같은 결과를 내는지(결정론적 시뮬레이션이면 100% 일치해야 함) | 낮음 — 기존 `run_episode()` 재호출만 필요 |
| 스키마 검증 | `observation.state`/`action` 차원·dtype·`loss_weight`∈[0,1]·`failure_type`∈{-100}∪{0..K-1}·`phase`∈{normal,fault,recovery} 등을 `LeRobotDataset.create()`가 선언한 `features`와 대조 | 낮음 |
| action/state 통계 비교 | 생성한 데이터의 채널별 mean/std/range를, `docs/configurable_ard.md`가 이미 집계한 **RoCo 공식 데이터 실측 통계**(200 에피소드/121,454프레임)와 나란히 놓고 같은 물리적 단위·같은 범위인지 확인(예: 그리퍼 범위, 오른팔이 "거의 상수"인지) | 낮음(통계만 내면 됨) — **단 ARD-Gen의 OpenArm 모델 자체가 RoCo 로봇과 동일한 기구학/치수인지는 미확인**이므로, 숫자가 다르더라도 "틀렸다"가 아니라 "다른 로봇"일 수 있다는 점을 같이 봐야 함 |
| phase/failure_type 라벨 일관성 | 주입 시점 전/후로 `phase`가 정확히 normal→fault→recovery 순서로만 바뀌는지, `loss_weight`가 fault 구간에서 낮은지(설계 의도대로) | 낮음 — 생성 스크립트 자체의 단위 테스트로 충분 |

### 6.2 "복구 데이터가 실제로 도움이 되는가" — 오프라인으로 판단 가능한가 (정직한 한계)

**이 질문은 7월 전(GPU/Isaac 없음)에는 온전히 답할 수 없다.** 가능한
근사치와 그 한계:

| 근사 방법 | 할 수 있는 것 | 못 하는 것(한계) |
|---|---|---|
| 작은 합성 모델로 ARD-VLA forward/backward만 돌려보기(`tests/test_configurable_ard.py` 패턴과 동일) | loss가 실제로 줄어드는지, `loss_weight`가 gradient에 의도대로 반영되는지(수치적 정합성) | **정책이 실제로 복구를 "배우는지"는 전혀 알 수 없다** — 합성 모델은 무작위 초기화된 작은 백본이라 의미 있는 행동을 배울 능력 자체가 없음 |
| MuJoCo 시뮬레이션 자체에서 "복구 전문가 스크립트"의 성공률 측정(3절 표의 전문가 동작) | "주입한 실패가 설계한 복구 동작으로 실제로 고쳐지는지"(복구 동작 자체의 타당성) | **학습된 정책이 그 전문가 동작을 모방해서 실제로 일반화하는지는 알 수 없다** — 이건 행동 복제 문제 특유의 한계(ARD-Gen의 `bootstrap/` 실험이 이미 겪은 "검증 MSE는 완벽한데 폐루프에서 완전히 실패" 사례, PIPELINE.md/README.md 참고)와 같은 종류 |
| CPU로 작은 diffusion/MLP 정책을 학습해 복구 시나리오에서 성공률 측정(ARD-Gen `bootstrap/` 패턴 재사용) | "이 정도 규모 모델로 이 복구 동작이 학습 가능한지"의 아주 거친 신호 | **SmolVLA(VLM 기반)와 작은 MLP는 용량·입력(비전+언어 포함)이 전혀 다르다** — 이 결과를 SmolVLA 성능으로 외삽하는 건 근거가 약함 |

**정직한 결론**: 지금 환경(GPU 없음, Isaac 없음)에서 할 수 있는 오프라인
검증은 **"데이터가 스키마적으로 올바르고, 복구 전문가 동작이 시뮬레이션
안에서 실제로 문제를 고친다"는 것까지다.** "이 데이터로 학습한 SmolVLA가
RoCo 평가에서 더 높은 점수를 받는가"는 **실제 학습(GPU)과 RoCo 평가 환경
접근 없이는 검증 불가능**하다 — 이 한계는 숨기지 않고 마일스톤(9절)에
명시적으로 반영한다(7월 이후로 분리).

### 6.3 ARD-VLA 학습 코드의 프레임별 가중치 지원 확인 (확인됨 — 코드로 직접 추적)

**결론: 지금 지원하지 않는다.** `modeling_smolvla.py`/`scripts/train_ard.py`
전체에서 `loss_weight`/`frame_weight`/`sample_weight` 류 키를 읽는 코드가
0건이다. 다만 **거의 같은 자리에 이미 비슷한 메커니즘이 있다**:

```python
# modeling_smolvla.py:566, 572-574 (SmolVLAPolicy.forward())
actions_is_pad = batch.get("actions_id_pad")
...
if actions_is_pad is not None:
    in_episode_bound = ~actions_is_pad
    losses = losses * in_episode_bound.unsqueeze(-1)   # (B, chunk, action_dim)
```

이게 정확히 "프레임별(사실은 청크 내 타임스텝별) 가중치를 곱해서 손실을
죽인다"는, 우리가 `loss_weight`에 원하는 것과 **동일한 shape·동일한
주입 지점**이다. 그런데 **이 메커니즘은 지금 항상 비활성이다** —
`LeRobotDataset`이 윈도우 쿼리마다 자동 생성하는 패딩 마스크 키는
`f"{key}_is_pad"`이므로 action 키가 `"action"`이면 실제로 생성되는 키는
`"action_is_pad"`인데(확인됨, `third_party/lerobot/.../lerobot_dataset.py:994`),
모델은 `"actions_id_pad"`(복수형 + `is`→`id` 오타로 보임)를 찾는다 —
**이름이 일치하지 않아 `batch.get(...)`이 항상 `None`을 반환하고,
`actions_is_pad` 분기 전체가 죽어 있다**(확인됨, `scripts/train_ard.py`에도
이 키를 채우는 코드가 없음). 이건 이번 작업이 만든 문제가 아니라 기존
vendored 코드의 선재 결함이고, 수정은 이 설계 문서의 범위 밖이다(별도
이슈로 분리 제안) — 다만 **새 `loss_weight` 메커니즘을 설계할 때 이 오타를
반복하지 않는 것**이 중요한 교훈이다.

**제안(구현은 범위 밖, 설계만)**:
1. LeRobot v3.0 writer가 `loss_weight`를 `action`과 같은 fps로 저장(스칼라
   컬럼, (T,) 형태).
2. `resolve_delta_timestamps()`(확인됨, `third_party/lerobot/.../datasets/factory.py:38`)가
   `ACTION`/`OBS_PREFIX`/`REWARD` 세 종류만 윈도우 처리하므로, `loss_weight`도
   같은 `action_delta_indices`로 윈도우되도록 이 함수(또는 그 호출부)를
   확장해야 한다 — 그래야 배치에서 `(B, chunk_size)` shape으로 들어온다.
3. `modeling_smolvla.py`의 `losses = losses * in_episode_bound.unsqueeze(-1)`
   바로 다음 줄에 `losses = losses * batch["loss_weight"].unsqueeze(-1)`
   같은 형태로 곱한다(기존 패딩 마스크와 완전히 같은 패턴 — 새 메커니즘이
   아니라 **이미 검증된 메커니즘의 두 번째 사용처**).
4. 기본값(`loss_weight` 없음/전부 1.0)이면 기존과 bit-exact — 이 프로젝트의
   기존 설계 원칙(`docs/configurable_ard.md`의 모든 신규 필드가 지키는
   "꺼져 있으면 기존과 동일")과 일치시킨다.

## 7. 시뮬 간 격차: MuJoCo vs Isaac Sim

| 격차 | 학습에 미치는 영향(추정) | 완화책(제안) |
|---|---|---|
| 렌더링(그림자/반사/재질) | 비전 인코더가 "MuJoCo스러운" 저사실감 텍스처에 과적합할 위험 — RoCo 공식 평가가 Isaac 기반이라면(**미확인**, RoCo 공식 환경 자체가 미확인) 실제 평가 영상과 분포가 다를 수 있음 | 5절의 색/텍스처/조명 랜덤화를 **의도적으로 과하게** 줘서(domain randomization의 일반적 목적) 비전 인코더가 특정 렌더 스타일에 덜 민감하게 만든다 |
| 접촉/마찰 물리 모델 | 같은 `friction` 수치라도 솔버가 다르면 실제 거동이 다르다(이미 ARD-Gen 안에서도 VX300s→OpenArm 전환 때 게인 스케일 자체가 달라진 사례가 있음, `PIPELINE.md` "근본적 차이" 절) — 복구 전문가 동작(3절)의 정확한 파라미터(오프셋 배율, weld 강성 등)가 Isaac에서는 재보정 필요 | 복구 전문가의 파라미터를 하드코딩하지 않고 `tasks/*.yaml`에 두어(이미 그런 패턴, `gain_bounds`/`x0` 등) 백엔드별로 다른 값을 쓸 수 있게 한다 |
| 로봇 모델(기구학/관성) 자체가 RoCo 공식 로봇과 다를 가능성 | ARD-Gen의 OpenArm 모델이 RoCo 공식 시뮬이 쓰는 로봇과 치수/DOF가 동일한지 **미확인** — 다르면 state/action 수치 분포 자체가 애초에 안 맞을 수 있음 | 생성 데이터 통계를 RoCo 실측 통계(6.1절)와 비교할 때 이 가능성을 항상 같이 체크 |
| 제어 주파수/타임스텝 | MuJoCo 100Hz(확인됨) vs Isaac의 기본 물리 스텝이 다를 수 있음(미확인) | LeRobot 쪽 fps=10은 "저장 fps"라 제어 주파수와 분리돼 있으므로(1.2절), 제어 주파수 차이 자체는 최종 데이터 스키마에 드러나지 않음 — 다만 컨트롤러 게인 재튜닝은 필요 |

**정직한 한계**: 이 격차들이 실제로 학습 성능에 얼마나 영향을 주는지는
**Isaac Sim을 돌려보기 전에는 추정만 가능하다**(전역 전까지 미확인으로
남는다 — 9절 마일스톤 참고).

## 8. 규모와 비용

### 8.1 실측 처리량 (확인됨 — 이 샌드박스에서 직접 측정)

4코어 CPU 샌드박스에서 `pipeline/bootstrap.py`와 같은 방식으로 기존 태스크의
`run_episode()`를 반복 호출해 직접 측정했다(이미지 렌더링 없이, 순수 물리
스텝만):

| 태스크 | 측정값 (N=10) |
|---|---|
| `peg_in_hole`(OpenArm 7-DOF 양팔, computed-torque) | **1.28 s/에피소드** |
| `cap_twist`(단순 힌지) | 0.057 s/에피소드 |
| `tacker`(OpenArm 7-DOF 양팔) | 0.201 s/에피소드 |

RoCo 스키마와 가장 가까운 "OpenArm 7-DOF 양팔" 계열(peg_in_hole/tacker)
기준으로 **에피소드당 0.2~1.3초**(물리 스텝만, 이미지 렌더링 제외)가
확인된 수치다. **이미지 렌더링(3카메라×10fps×에피소드 길이)은 아직
파이프라인에 연결돼 있지 않아 실측하지 못했다** — `mujoco.Renderer`
자체의 프레임당 비용(일반적으로 카메라 1개·해상도 640×480 기준 수~수십ms
대라고 알려져 있음, **이 레포에서 240×320 3카메라로 실측한 적은 없어
미검증**)이 더해지면 물리 스텝보다 렌더링이 더 비쌀 가능성이 높다 —
반드시 실제 구현 1차 버전에서 재측정해야 한다(9절 마일스톤에 포함).

Colab(특히 무료 티어, 통상 2 vCPU 수준)은 이 4코어 샌드박스보다 느릴
가능성이 높지만, GPU가 필요 없는 순수 CPU MuJoCo 연산이라 **느려도
"안 되는" 수준은 아니다**(추정 — Colab에서 직접 측정하기 전까지 정확한
배율은 미확인).

### 8.2 목표 에피소드 수 (제안, 근거와 함께)

RoCo 공식 데이터는 200 에피소드다(확인됨, `docs/configurable_ard.md`가
이미 확인한 사실). 채점의 40%가 복구라는 비중을 그대로 데이터 비중에
반영한다는 **단순한 원칙**을 적용하면:

| 구성 | 제안 수량 | 근거 |
|---|---|---|
| 정상 시연(실패 없음) | 기존 유지(필요시 ARD-Gen으로 추가 생성 가능) | 이미 RoCo 공식 데이터 200개가 있음 |
| 복구 시연(3개 실패 유형 × 각 50~100개) | **150~300개** 1차 목표 | 정상:복구 비율을 대략 4:3~1:1.5 범위로 둬서 "복구가 희귀 이벤트로 과소표현되지 않게" 하되, 처음부터 과하게 키우지 않음(제안값일 뿐 — 실제 필요 비율은 **미확인**, 학습해보기 전에는 알 수 없음) |

이 규모는 8.1절 실측 기준(물리 스텝만 0.2~1.3s/에피소드)으로는 **CPU
수 분~수십 분**이면 생성 가능한 양이다(렌더링 비용은 미반영 — 실제로는
렌더링이 지배적 비용이 될 가능성이 높으므로 이 추정은 낙관적 하한으로
읽어야 한다).

## 9. 마일스톤 (2026-12 ~ 2027-04)

전역(2027-07-11) 전이므로 전부 **MuJoCo + Colab/Claude Code** 범위다.
Isaac Sim 관련 작업은 설계/인터페이스 정의까지만 하고 실행은 포함하지
않는다.

| 기간 | 작업 | 완료 기준 |
|---|---|---|
| 2026-12 | `SimBackend` 인터페이스 정의 + `MuJoCoBackend` 구현(기존 `sim/*_sim.py`의 물리 호출을 이 인터페이스로 래핑, 리팩토링만, 동작 변경 없음) | 기존 회귀 테스트(`tests/test_*.py`)가 전부 그대로 통과(동작이 안 바뀌었다는 증거) |
| 2026-12 ~ 2027-01 | 3카메라(head/left_hand/right_hand) XML 정의 + 렌더링을 `run_episode()` 루프에 연결, 10fps 다운샘플 | 생성된 에피소드 하나를 로드해 3카메라 각각 (T,240,320,3) 배열이 나오고, 프레임 수가 `action` 프레임 수의 1/10(10fps 다운샘플 확인)과 일치 |
| 2027-01 | state 44차원/action 14차원 스키마로 `to_lerobot.py`를 재작성(LeRobot `LeRobotDataset.create`/`add_frame` API 직접 사용, npz 근사 폐기) | RoCo 실측 통계(`docs/configurable_ard.md`)와 채널별 범위를 나란히 비교한 표가 나오고, `LeRobotDataset(repo_id, ...)`로 실제 로드가 성공 |
| 2027-01 ~ 02 | `FaultInjector` + 3개 실패 유형(3절) 구현, `loss_weight`/`failure_type`/`phase`/`source` 프레임 필드 추가 | 각 실패 유형당 N=20 소규모 시범 생성에서 "주입→복구 전문가→성공" 전체 흐름이 6.1절 리플레이 검증을 통과 |
| 2027-02 | ARD-VLA 쪽 `loss_weight` 배치 키 지원(6.3절 제안 반영, `actions_id_pad` 오타 재발 방지) | 작은 합성 모델로 `loss_weight=0` 구간의 gradient가 실제로 0이 되는 단위 테스트 통과, 기존 전체 테스트 스위트 회귀 없음 |
| 2027-02 ~ 03 | 시각 도메인 랜덤화(조명/색) + 1차 규모(150~300 에피소드) 생성 | 8.1절 실측 처리량을 렌더링 포함해서 재측정하고, 목표 수량 생성에 걸리는 실제 wall-clock 시간을 보고 |
| 2027-03 ~ 04 | 생성 데이터 전체 검증(6.1절 전부) + RoCo 통계 대조 보고서 | 모든 에피소드가 스키마 검증 통과, 통계 비교 결과(일치/불일치 항목)를 문서화 |
| (범위 밖, 7월 이후) | Isaac Sim `SimBackend` 구현, 실제 학습/평가로 복구 데이터 효과 검증 | 이 문서의 범위가 아님 — 전역 후 별도 계획 |

## 10. 미해결 질문 목록

### 내가(사용자가) 결정해야 하는 것

1. 복구 시연의 정상:복구 비율(8.2절 제안값 150~300개를 그대로 쓸지, RoCo
   실제 채점 가중치 40%를 더 직접적으로 반영한 다른 비율을 쓸지).
2. "잘못된 부품 삽입" 실패 유형(3절)에 들어갈 새 씬 지오메트리(두 번째
   가짜 목표)를 어느 태스크에 먼저 추가할지 — peg_in_hole 확장이 자연스러운
   후보지만, 새 태스크를 따로 만들 수도 있음.
3. 실패 주입 타이밍의 진행률 경계값(0.2~0.8, 3절 제안)과 위치 어긋남의
   오프셋 배율(2~3배, 제안)을 이 값으로 확정할지, 실측 후 조정할지.
4. `SimBackend` 리팩토링(9절 2026-12 작업)을 **먼저** 하고 그 위에 나머지를
   쌓을지, 아니면 지금 구조에서 복구 파이프라인을 먼저 만들고 나중에
   `SimBackend`로 재래핑할지(후자가 더 빠르지만 2절의 "백엔드 교체 시
   안 바뀌는 범위" 보장이 약해짐).

### 7월(전역) 이후에야 알 수 있는 것

1. RoCo 공식 환경의 "정상 시연 생성기"(스크립트 전문가)가 실제로 존재하는지,
   존재한다면 API가 뭔지 — 존재하면 이 문서의 "정상 시연은 이미 있다"는
   전제를 재검토해야 할 수도 있다(**미확인**, 가정하지 않음).
2. RoCo 공식 평가 환경이 Isaac Sim 기반인지, 그 시뮬레이터의 접촉/마찰
   모델이 MuJoCo와 구체적으로 얼마나 다른지(7절).
3. MuJoCo로 생성한 복구 데이터로 학습한 정책이 실제로 RoCo 평가에서
   복구 성능이 올라가는지(6.2절의 근본적 한계 — GPU 학습 + 실제 평가
   환경 접근 없이는 검증 불가능).
4. ARD-Gen의 OpenArm 모델이 RoCo 공식 로봇과 치수/DOF/관성이 실제로
   일치하는지(1.3절/7절에서 반복적으로 등장한 미확인 사항 — 공식
   info.json/URDF를 직접 받아보기 전까지는 알 수 없음).
