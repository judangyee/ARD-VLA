# RoCo 평가용 정책 실행 인터페이스

[← README로 돌아가기](../README.md)

구현: `scripts/policy_interface.py`(`RoCoPolicyAdapter`, `replay_episode`). 테스트:
`tests/test_policy_interface.py`.

**전제**: 공식 RoCo 평가 환경의 실제 API(관측을 주는 함수 시그니처, 액션을 받는 함수 시그니처,
호출 빈도, 카메라 관측 키 이름 등)는 이 레포 어디에도 없고, 이 작업을 한 세션에서 HF Hub에
접근할 수 없어 열어볼 수도 없었다. 아래 인터페이스는 **공식 환경과 대조해서 맞춘 것이 아니라**,
이 레포가 이미 가진 `SmolVLAPolicy.select_action()`/`make_smolvla_pre_post_processors()`의 실제
입출력을 그대로 감싼 것이다 — "이 레포 쪽에서 요구하는 모양"은 확정이고, "공식 환경이 실제로
그 모양을 주는지"는 맨 아래 "미확인 항목"에 전부 모아뒀다.

## 1. 입출력 인터페이스

```python
class RoCoPolicyAdapter:
    def reset(self) -> None: ...
    def act(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> torch.Tensor: ...

def replay_episode(adapter: RoCoPolicyAdapter, frames: list[dict]) -> dict: ...
```

**입력**(`act()`):
- `images`: `{"head": (240,320,3) uint8, "left_hand": (240,320,3) uint8, "right_hand": (240,320,3) uint8}`.
  실제 observation 피처 키 이름(`observation.images.*`)으로의 매핑은 생성자의 `camera_keys`
  인자로 바꿀 수 있다(기본값은 `DEFAULT_CAMERA_KEYS`, **추정값 — 미확인**).
- `state`: `(44,)` float32. 순서는 `configs/presets/roco_board.yaml` 상단 주석대로
  EE 위치+쿼터니언 14 + 관절 위치 14 + 관절 속도 14 + 그리퍼 2(**이 순서 자체는 사용자가 RoCo
  공식 데이터셋 info.json을 직접 읽어 확인한 값** — 레포가 환경과 통신해서 확인한 값은 아니다).
  `state_select`(관절 속도 28-41 제외, 30채널만 모델에 들어감)는 **이 어댑터가 아니라 정책
  내부**(`SmolVLAPolicy.prepare_state()`)에서 적용되므로, 여기서는 항상 44차원 그대로 넘긴다.
- `task`: 자연어 지시문 문자열(예: `"insert the bolt into the board"`).

**출력**: `(14,)` float32 — EE 절대 목표(각 팔 [xyz 3 + rx,ry,rz 3 + 그리퍼 1] = 7, 왼팔 먼저).
연속 각도(값 자체가 ±π를 넘어갈 수 있음, RoCo 데이터가 이미 "unwrap된" 형태로 기록돼 있다는
실측 근거는 `docs/configurable_ard.md`의 "RoCo 시뮬 실측" 절 참고) — `angle_unwrap_dims=None`
(RoCo 프리셋 기본값)이라 출력에 wrap(`[-π,π)`로 다시 가두기)을 적용하지 않는다. wrap이
필요하면 호출자가 직접 해야 한다(이 어댑터는 하지 않는다).

## 2. 내부 처리

`act()` 한 번은 다음을 순서대로 한다(전부 이 레포 기존 코드를 그대로 재사용 — 새로 구현한
부분은 "이미지 dtype/레이아웃 변환"과 "조각들을 하나로 묶기"뿐이다):

1. 이미지 uint8 HWC → float32 CHW [0,1] 변환(`_image_hwc_uint8_to_chw_float`). 리사이즈/패딩/
   [-1,1] 변환은 하지 않는다 — `SmolVLAPolicy.prepare_images()`(모델 내부, `select_action()`이
   호출)가 담당.
2. `preprocessor(obs)`(`make_smolvla_pre_post_processors()`가 만든 파이프라인) — 배치 차원
   추가, task 문자열 토큰화, **state 정규화(여기서 `state_min_std`가 적용됨, 44차원 전체에
   대해 — `state_select`보다 먼저)**, device 이동.
3. `policy.select_action(batch)` — **state_select**(여기서 처음 적용, 44→30채널), ARD 비대칭
   헤드, **n_action_steps 단위 액션 큐**(5스텝 계획하면 5번의 `act()` 호출 중 1번만 실제
   forward, 나머지는 큐에서 pop), LAC(`lac_mode != "none"`일 때)까지 전부 이 한 호출 안에서
   일어난다. RoCo 프리셋은 `lac_mode: none`이라 LAC는 기본적으로 꺼져 있다.
4. `postprocessor(action)` — **역정규화**(`action_min_std` 반영), CPU 이동.

## 3. `reset()`

`policy.reset()`을 그대로 부른다 — 액션 큐(`n_action_steps` 길이 deque)와 LAC 필터 상태를
전부 새로 만든다(`modeling_smolvla.py:270-293`). **task-graph 수준의 "폐루프 상태"**(서브태스크
진행 상황, 재시도 횟수 — `closed_loop.py`의 `EpisodeState`)는 이 어댑터의 책임이 아니다. 이
어댑터는 "관측 한 번 → 액션 한 번"의 최하위 스텝 단위 인터페이스이고, Judge/Selector로 여러
서브태스크를 오가는 평가가 필요하면 `lerobot.policies.smolvla.closed_loop.run_closed_loop()`를
(이 어댑터 대신, 또는 이 어댑터의 `preprocessor`/`postprocessor`를 그대로 넘겨서) 쓴다 — 그
러너가 이미 need_judgment/재시도/복구/n_action_steps 재계획 간격(이번 세션에 `closed_loop.py`에
새로 연결함, 아래 참고)을 전부 구현하고 있다.

## 4. 오프라인 리플레이 하네스

`replay_episode(adapter, frames)` — LeRobot 포맷 프레임(dict) 리스트를 시간 순서대로
`adapter.act()`에 넣고, 예측 액션과 `frame["action"]`(정답) 사이의 **차원별 L1**
(`|pred - true|`, 프레임 평균)을 계산한다. 환경과 통신하지 않는다 — "정답 액션을 실행했으면
실제로 어떻게 됐을지"가 아니라 "그 상황에서 정책이 무엇을 예측했는지"만 본다(표준적인 오프라인
모방학습 지표의 한계 — RoCo 실제 채점 방식과 다를 수 있음, **미확인**).

```python
frames = [
    {"observation.images.head": img_hwc_uint8, "observation.images.left_hand": ...,
     "observation.images.right_hand": ..., "observation.state": state_44d,
     "action": action_14d_ground_truth, "task": "insert the bolt into the board"},
    ...  # 시간 순서
]
result = replay_episode(adapter, frames)
result["per_dim_l1"]   # (14,) ndarray
result["mean_l1"]      # float
```

## 5. 공식 환경과 연결할 때 확인해야 할 미확인 항목

| # | 항목 | 왜 미확인인가 |
|---|---|---|
| 1 | 카메라 관측 피처의 실제 키 이름 | `DEFAULT_CAMERA_KEYS`는 추정값. RoCo 공식 데이터셋 info.json(HF Hub)을 열어봐야 확정된다. |
| 2 | 카메라가 정말 3대(head/left_hand/right_hand)인지, 역할 이름이 맞는지 | 사용자가 이번 요청에서 준 스펙을 그대로 코드화했을 뿐, 공식 스펙 문서를 대조하지 않았다. |
| 3 | state 44차원의 정확한 채널 순서 | `roco_board.yaml` 상단 주석(사용자가 info.json 채널 이름으로 확인했다고 기록)에 의존 — 이 세션이 직접 재확인하지 않았다. |
| 4 | 평가 하네스가 `act()`를 몇 Hz로 부르는지, 한 에피소드가 몇 스텝인지 | 전혀 모름 — `n_action_steps`(재계획 간격)를 실제 호출 빈도에 맞게 튜닝해야 할 수도 있다. |
| 5 | 액션을 "절대 목표"로 그대로 로봇에 보내는지, 추가 변환(좌표계/단위)이 있는지 | `action_space="ee"`로 설정만 해뒀을 뿐, 로봇 API가 실제로 EE 절대 pose를 받는지 검증 안 됨. |
| 6 | 에피소드/서브태스크 종료를 평가 하네스가 어떻게 알리는지 | `ClosedLoopEnv.subtask_finished()`에 대응하는 공식 개념이 있는지 모름. |
| 7 | 정규화 통계(`dataset_stats`) | 이 어댑터는 `preprocessor`/`postprocessor`를 호출자가 만들어서 넘기게 돼 있다 — 실제 RoCo 학습 통계로 만든 `dataset_stats`가 필요하다(`make_smolvla_pre_post_processors(config, dataset_stats=...)`). 이 세션은 실제 통계에 접근할 수 없어 테스트에서는 `dataset_stats=None`(사실상 미정규화)으로만 확인했다. |
| 8 | `per_timestep_norm`의 a/s 계수 | `roco_board.yaml`이 항등값(a=1, s=0)으로 둔 채다 — `scripts/fit_timestep_norm.py`를 실제 데이터로 돌려야 한다(`docs/model_completion_audit.md` 7절 #4와 동일 항목). |
| 9 | `replay_episode`의 L1 지표가 RoCo 공식 채점 지표와 같은 개념인지 | 모름 — 공식 리더보드가 성공률/삽입 깊이 등 다른 지표를 쓸 수도 있다. |

## 관련 변경 (이번 세션)

- `closed_loop.py`가 이제 `config.n_action_steps`를 재계획 간격으로 실제로 쓴다(이전에는
  전혀 참조하지 않았다 — `docs/model_completion_audit.md` 6절/7절 #5에서 발견한 문제).
  자세한 설명은 `closed_loop.py`의 "재계획 간격" 절 docstring 참고.
- `ard.py`에 `apply_ard_residual()`을 추가해서, 학습 forward와 `denoise_step`에 중복돼 있던
  "ARD 잔차를 action_out_proj 출력에 더하는" 코드를 하나로 합쳤다(동작 변화 없음 — 직접 비교
  테스트로 두 경로가 bit-for-bit 동일한 v_t를 내는지 확인함).
