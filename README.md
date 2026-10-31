# ARD-VLA

Research environment for experimenting with [SmolVLA](https://huggingface.co/docs/lerobot/smolvla), the small vision-language-action policy from Hugging Face's [LeRobot](https://github.com/huggingface/lerobot) framework.

**ARD-VLA가 무엇인가.** 양손 도구 조작(한 팔은 작업물을 고정하는 Stabilizer, 다른 팔은
정밀 조작을 수행하는 Actuator) 파인튜닝을 위해 SmolVLA의 액션 전문가에 선택적인 비대칭
듀얼 헤드 구조(ARD, Asymmetric Role Decomposition)를 추가한 연구용 포크입니다. 모든 ARD
관련 기능은 기본적으로 꺼져 있고, 켜지 않으면 업스트림 SmolVLA와 동일하게 동작합니다.

**연구 가설.** 양손 조작에서 두 팔은 역할이 본질적으로 다르다 — Stabilizer는 "흔들리지
않고 가만히 있는 것"이, Actuator는 "정밀하게 힘/궤적을 제어하는 것"이 중요하다. 이
역할 비대칭을 모델 구조(헤드 분리)와 손실 함수(팔별로 다른 정규화 항)에 명시적으로
반영하면, 양팔에 동일한 처리를 적용하는 대칭(symmetric) 접근보다 더 나은 정책을 학습할
수 있을 것이다. 이 레포는 이 가설을 검증하기 위한 ARD 구조/손실과, 공정 비교를 위한
대칭 대조군(`ard_symmetric`)을 함께 제공한다 — 아직 실제 데이터로 가설 자체를 검증하지는
못했다(아래 각 기능 문서의 "검증된 것 vs 아닌 것" 참고).

## Setup

```bash
./scripts/install.sh
```

This will:

1. Create a virtualenv at `.venv` (pass `--no-venv` to install into the current interpreter instead).
2. Detect whether an NVIDIA GPU is present (`nvidia-smi`) and install a matching `torch`/`torchvision` build — CPU-only wheels when no GPU is found, so machines without a GPU don't pay for a CUDA download. If the CPU wheel index (`download.pytorch.org`) isn't reachable on your network, it falls back to the default PyPI build.
3. Install `lerobot` **editable** from the vendored source at `third_party/lerobot` with the `smolvla` extra.
4. Install research tooling from `requirements.txt`.
5. Run `scripts/check_env.py` to confirm everything imports cleanly.

Activate the environment afterwards with:

```bash
source .venv/bin/activate
```

## Verifying the environment

`scripts/check_env.py` is a standalone, offline smoke test — it imports `torch`, `transformers`, `lerobot`, and the SmolVLA policy/config classes, and confirms device selection falls back to CPU when no GPU/accelerator is available. It downloads no model weights, so it's safe to run on any machine, GPU or not:

```bash
python scripts/check_env.py
```

## 테스트 (`pytest`)

단위/오프라인 테스트는 전부 `tests/`에 있고, 레포 루트의 `pytest.ini`가 `testpaths = tests`로
설정돼 있어서 아무 인자 없이 `pytest`만 치면 전부 돈다:

```bash
pytest
```

GPU도, Hugging Face Hub 접근도 필요 없다 — `tests/conftest.py`가 `AutoConfig.from_pretrained`/
`AutoProcessor.from_pretrained`를 아주 작은 합성 SmolVLM 설정으로 몽키패치해서, 실제
`SmolVLAPolicy` 전체(forward/backward/추론 경로 포함)를 검증하는 테스트(`tests/
test_ard_integration.py`)까지도 오프라인으로 돈다. 개별 파일만 돌리려면
`pytest tests/test_ard.py`처럼 평소 pytest 쓰듯 하면 된다. CI(`.github/workflows/tests.yml`)도
같은 명령(`pytest -m "not requires_hub"`)으로 매 push/PR마다 돈다.

## Modifying SmolVLA's model code

`lerobot` is installed in **editable mode** from the source vendored at `third_party/lerobot` (trimmed from [huggingface/lerobot](https://github.com/huggingface/lerobot) `v0.4.4`, Apache-2.0), not from PyPI. This means the SmolVLA implementation lives inside this repo and is tracked by git:

- Model/config code: `third_party/lerobot/src/lerobot/policies/smolvla/`
- Edit those files directly — changes take effect immediately in the active venv (no reinstall needed) and can be committed like any other file in this repo.
- `third_party/lerobot` has no nested `.git`; it's plain vendored source, so `git status`/`git diff` at the repo root see it normally.

If you need to pull in upstream lerobot changes later, re-clone the desired tag/commit into `third_party/lerobot` and re-apply any local modifications (there's no submodule link to fast-forward).

### Why the vendored tree is smaller than upstream

`third_party/lerobot` only keeps what SmolVLA's own code actually imports at runtime — verified by tracing `sys.modules` after importing the SmolVLA config/policy/processor classes plus dataset loading. Everything else was deleted:

- Other policy families (ACT, Diffusion, GR00T, pi0/pi0.5, SAC, SARM, TD-MPC, VQ-BeT, Wall-X, XVLA) — `lerobot/policies/__init__.py` used to eagerly import all of their config classes; it's trimmed to just SmolVLA + its RTC dependency now.
- Physical hardware drivers (`robots/*`, `teleoperators/*`, `motors/*` vendor subdirs, `cameras/`) — only the base config/abstract classes remain, since we're not driving real robots.
- `scripts/` (lerobot's CLI tools: calibration, teleop, the generic multi-policy train/eval commands), `rl/`, `async_inference/`, `transport/`, `data_processing/`, `model/`, plus `docs/`, `examples/`, `tests/`, `benchmarks/`, `media/`, `.github/` — none of it is needed to import or run SmolVLA.

What's left (`configs/`, `datasets/`, `envs/configs.py` only, `optim/`, `policies/{smolvla,rtc,pretrained.py,utils.py}`, `processor/`, `robots/`+`teleoperators/` base classes, `motors/motors_bus.py`, `utils/`) is the actual dependency closure for building/training/running the SmolVLA policy against a `LeRobotDataset`. `scripts/check_env.py` and the trace above are re-run after any change to confirm nothing extra crept back in.

## ARD: 비대칭 역할 분리 (Asymmetric Role Decomposition)

SmolVLA의 액션 전문가에 Stabilizer/Actuator 전용 residual head를 추가하는 핵심 기능입니다.
`use_ard=False`(기본)면 업스트림 SmolVLA와 완전히 동일하게 동작합니다. 타겟 로봇은
OpenArm + 공식 그리퍼로 확정됐습니다(팔당 `ard_arm_dim` 기본값 **8** = 관절7+그리퍼1).

- **핵심 구조/손실**: `AsymmetricResidualHeads`, `compute_ard_losses`(`L_pos`/`L_smooth`/
  `L_force`/`L_traj`를 `alpha`/`beta`로 결합), 고정 왼팔/오른팔 라우팅.
- **`ard_reg_time_weighting`**("none" 기본 | "one_minus_t") — smooth/traj 손실이 노이즈가
  많이 섞인(`t`가 큰) 샘플에서 과도하게 시끄러워지는 것을 `(1-t)` 가중으로 완화하는 옵션.
- **`ard_use_force_head`**(기본 False) — 접촉력/토크를 actuator 채널 재활용 대신 별도
  `ForceHead` MLP로 예측.
- **`ard_gripper_dim`**(기본 1) — 각 팔의 마지막 N개 채널(그리퍼)을 smooth/traj 벌점에서
  제외(그리퍼는 `L_pos`만 받음) — bang-bang성 신호에 매끄러움을 강제하지 않기 위함.
- **`ard_use_joint_torque`**(기본 False) — 관절 토크 관측값을 `observation.state`에 concat
  해서 모델 입력으로 사용(새 파라미터 없음).
- **`ard_symmetric`**(기본 False) — 양팔에 동일한 정규화를 거는 공정 비교용 대조군 모드.
- **GradNorm** (`--use-gradnorm`) — smooth/force/traj lambda를 그래디언트 norm 기준으로
  자동 조정.
- **Wrist 카메라**: `config.image_features`를 그냥 순회하는 기존 메커니즘이 이름/개수에
  무관하게 처리하므로 코드 변경 없이 추가 이미지 입력으로 들어갑니다.

자세한 구조 설명, 각 옵션 사용법, 그리고 지금까지의 검증된 것/측정된 것/아직 확인되지 않은 것
기록은 **[docs/ard.md](docs/ard.md)** 에 있습니다.

```bash
pytest tests/test_ard.py tests/test_ard_integration.py
```

## Bridge Attention (`--use-bridge-attention`, VLA-Adapter식)

ARD head가 action expert의 마지막 지점 출력(`suffix_out`) 하나만 보는 대신, SmolLM2 백본의
여러 중간 레이어를 cross-attention으로 동시에 조건받게 하는 옵션입니다. zero-init gate라
꺼져 있으면(기본) 기존과 100% 동일합니다. 파라미터 수 증가, 레이어 인덱스 자동/수동 선택,
`scripts/compare_bridge_attention.py`를 통한 loss 비교 방법은 **[docs/bridge_attention.md](docs/bridge_attention.md)**
에 있습니다.

## FreqPolicy: 주파수 영역 일관성 손실 (`--use-freq-policy`)

액션 청크를 DCT로 분해해서 저주파(궤적 형태)를 고주파(디테일)보다 우선하는 추가 손실입니다.
`use_ard`와 독립적으로 켤 수 있고, 새 학습 파라미터가 없습니다. 수학적 등가성(Parseval)
검증과 사용법은 **[docs/freq_policy.md](docs/freq_policy.md)** 에 있습니다.

## 비전 토큰 프루닝 (`--use-token-pruning`, EfficientVLA식)

프레임당 고정 64개 비전 토큰을 태스크 관련성 + 다양성 기준으로 줄이는 옵션입니다. 선택
로직, gradient 흐름, K_final 스윕 검증은 **[docs/token_pruning.md](docs/token_pruning.md)**
에 있습니다.

## 폐루프 구조: Judge/Selector + 결정 로직 (`--use-judge`/`--use-selector`)

SmolVLA(+ARD)를 "한 번 추론" 대신, 서브태스크마다 성공/실패를 스스로 판정(Judge)하고 다음
행동(Selector + 순수 Python 결정 로직)을 고르는 폐루프 시스템으로 확장하는 구조입니다. 매
control cycle마다 SmolVLM2 prefix forward를 Judge/Selector/action expert가 한 번만 공유하고
(청크당 1회 — 노드/복구 전이처럼 지시문이 바뀌는 경계에서만 다음 cycle에 한 번 더 forward),
판정은 항상 그 cycle에서 실행한 결과를 반영한 **실행 후** 관측으로 이뤄집니다. 전부 기본
꺼짐이고, 꺼져 있으면 기존과 bit-for-bit 동일합니다. 구조도, 플래그 표, 배치 키, 판정
시점/forward 횟수 규칙, 검증된 것/아직 확인되지 않은 것은
**[docs/closed_loop.md](docs/closed_loop.md)** 에 있습니다 — 이번 작업은 구조 구현만이고,
실제 Judge/Selector 학습 데이터 생성·학습·임계값 캘리브레이션은 범위 밖입니다.

## LAC: 추론 시점 EMA 필터 (`--lac-mode covla|role_aware`, Co-VLA arXiv 2606.20285)

학습 없이 추론 시점에서만 동작하는 떨림 억제 필터입니다. ARD의 "공유(action_out_proj)+
잔차(ARD 헤드)" 가산 구조에서 팔별 공유/잔차 에너지 비율(ρ)과 협동 방향(ω)을 보고 매
실행 스텝마다 EMA 필터 강도 α를 조절합니다 — `"covla"`는 Co-VLA 원본(두 팔에 같은 α),
`"role_aware"`는 ARD의 Stabilizer/Actuator 역할별로 α를 따로 둡니다(Stabilizer는 항상
강하게 평활화, Actuator는 반대 방향 협동일 때만 빠르게 통과). `select_action()`과
ARD-VLA 2.0 폐루프 경로(`predict_action_chunk_with_decisions`+`closed_loop.py`) 둘 다
같은 LAC 인스턴스를 공유하며 연결돼 있고, 폐루프 쪽은 서브태스크 전환/재시도·복구 진입
시점마다 LAC를 리셋합니다(이전 서브태스크의 궤적에 안 끌려가도록). 기본 `lac_mode="none"`
이면 기존과 bit-for-bit 동일합니다. 수식, 설정 플래그, 리셋 규칙, 오른팔 관절 미러링 보정
(mirror_sign, 미검증), 검증된 것/아직 확인되지 않은 것은 **[docs/lac.md](docs/lac.md)**
에 있습니다.

```bash
pytest tests/test_lac.py tests/test_ard_integration.py tests/test_lac_closed_loop.py
python scripts/compare_lac_modes.py             # select_action() 경로, 합성 백본
python scripts/compare_lac_closed_loop_modes.py # 폐루프 경로, 합성 백본
```

```bash
pytest tests/test_judge.py tests/test_ard_integration.py tests/test_decision_logic.py tests/test_closed_loop.py
```

## 설정 가능한 ARD (Configurable ARD) + RoCo Challenge 프리셋

ARD-VLA는 원래 OpenArm 16채널/왼팔=Stabilizer/chunk 50/L2 손실 전제였는데, 대회마다 과제가
바뀌는 RoCo Challenge 같은 쓰임을 위해 이 전제들을 전부 config로 뺐습니다: `role_map`(역할-팔
매핑), `action_space`("joint"\|"ee", "ee"면 `lac_mirror_sign` 자동 비활성), `pos_loss_type`
("l2"\|"l1"\|"smooth_l1", L2의 목표 근처 기울기 소실 문제 회피), `per_timestep_norm`(청크 내
위치별 정규화 스케일). 손실과는 완전히 독립적인 데이터 전처리 옵션 `angle_unwrap_dims`
(회전 채널을 학습 전 시간축으로 펴고 추론 후 로봇에 보내기 직전 다시 `[-π,π)`로 감싸는
전처리, `processor_smolvla.py`)도 추가했습니다 — pos_loss(u_t/v_t는 raw 액션이 아니라
flow-matching 속도장이라 거기서 각도 래핑을 하면 틀린 설계였습니다, docs 참고)는 건드리지
않습니다. `state_select`(`observation.state`가 `max_state_dim`보다 넓을 때 필요한 채널만
골라 패딩 에러를 피함)와 `action_min_std`(거의 상수인 채널의 정규화 노이즈 증폭을 막는
차원별 std 하한, 정규화·역정규화 양쪽에 동일 적용)도 추가했습니다. 전부 기본값이 기존과
bit-for-bit 동일합니다. `configs/presets/roco_board.yaml`이 RoCo용 조합을 담고 있습니다 —
action 14차원 스키마(팔당 [xyz+rxryrz+그리퍼]=7채널)와 state 44차원 구성은 **사용자가
공식 시뮬 데이터셋 200 에피소드/121,454프레임 전체를 집계해 확인**했습니다. 그 실측
결과 각도가 이미 연속(±π 경계 점프 0건)이라 `angle_unwrap_dims`는 꺼둔 채이고(켜면 오히려
불연속을 만듦), `observation.state`는 관절 속도(14차원)를 뺀 30차원만 골라 쓰고
(`state_select`), 사실상 상수인 오른팔 action 채널엔 `action_min_std`로 정규화 바닥을
깔았습니다. 조사 결과("뭐가 진짜 하드코딩이었나"), 각 필드의 수식적 근거, 검증된 것/
미확인 사항은 **[docs/configurable_ard.md](docs/configurable_ard.md)** 에 있습니다.

```bash
pytest tests/test_configurable_ard.py
python scripts/fit_timestep_norm.py --dataset-repo-id <HF_USER>/<DATASET> --chunk-size 20
python scripts/check_rotation_range.py --dataset-repo-id <HF_USER>/<DATASET>
```

## 학습 (로컬 GPU 환경)

lerobot의 범용 학습 CLI(`lerobot_train.py`)는 모든 정책을 알아야 하는 `policies/factory.py`에
의존해서 트림할 때 같이 지웠습니다. 대신 `scripts/train_ard.py`가 SmolVLA 하나만 아는 최소
학습 루프입니다: `LeRobotDataset` 로드 → `SmolVLAConfig`/`SmolVLAPolicy` 생성 → optimizer/
scheduler 빌드 → 학습 루프 → 주기적 체크포인트 저장. `use_ard=True`일 때는 `ard_stabilizer_loss`
등 손실 breakdown도 함께 로깅됩니다(모드별 `loss_dict["ard_mode"]`도 포함).

```bash
python scripts/train_ard.py \
    --dataset-repo-id <HF_USER>/<DATASET> \
    --output-dir outputs/ard_run1 \
    --steps 20000 \
    --batch-size 32
```

`LeRobotDataset`은 `config.chunk_size`(flow-matching이 한 번에 예측하는 액션 시퀀스 길이)
기준으로 `delta_timestamps`를 만들어서 로드합니다 — 이게 없으면 액션이 단일 프레임(한 시점)
으로만 나와서 `embed_suffix()`가 기대하는 `(B, chunk_size, action_dim)` 형태가 깨지고,
`make_att_2d_masks`에서 `RuntimeError: size ... must match ... at non-singleton dimension 2`로
학습이 크래시합니다. 실제로 `lerobot/aloha_mobile_cabinet`으로 첫 real-dataset 학습을 시도하다
발견한 버그이며, `lerobot.datasets.factory.resolve_delta_timestamps`(공식 `lerobot_train.py`가
쓰는 것과 동일한 유틸)로 고쳤습니다.

`--no-use-ard`를 주면 ARD 없이 베이스라인 SmolVLA만 학습합니다. `--ard-symmetric`을 주면
ARD 대신 [대칭 대조군 모드](docs/ard.md#대칭-대조군-모드-ablation-ard_symmetric)로 학습합니다.
`--ard-gripper-dim`(기본 1)/`--ard-use-joint-torque`로
[그리퍼 손실 분리/관절 토크 입력](docs/ard.md#그리퍼-손실-분리-ard_gripper_dim)을 켜고 끌 수
있습니다. 시작할 때 액션 채널의 왼팔/오른팔 예상 순서를 출력해주니, 실제 로봇 배선과 맞는지
눈으로 한 번 확인하세요 (ARD는 "앞 `ard_arm_dim`개=왼팔, 다음 `ard_arm_dim`개=오른팔"이라는
관례를 가정할 뿐, 데이터셋이 실제로 그 순서인지는 검증하지 않습니다).

`--vlm-layer-indices`로 VLM 레이어를 "앞쪽 N개"가 아니라 특정 인덱스 조합으로 구성해서 학습할
수도 있습니다 — `scripts/layer_importance.py`가 코사인 유사도 기준으로 골라준 레이어들을 그대로
넣는 식입니다(자세한 건 [docs/profiling_tools.md](docs/profiling_tools.md) 참고):

```bash
python scripts/train_ard.py --dataset-repo-id <...> \
    --vlm-layer-indices 2 3 4 7 8 9 10 11 12 14 15 16 20 21 25 26
```

이 스크립트도 이 샌드박스에서는 end-to-end로 돌려보지 못했습니다(Hub 접근 차단) — 대신
헬퍼 함수들(`split_policy_features`, `warn_if_action_layout_looks_wrong`)은 합성 데이터로
직접 검증했고, import/인자 파싱도 확인했습니다. 실제 학습 루프 자체는 로컬 GPU 환경에서
처음 돌려보실 때 검증해주세요.

## 파라미터/메모리 프로파일링 도구

`scripts/count_params.py`(구성 요소별 파라미터 집계), `gradient_checkpointing_enable()`,
`scripts/profile_memory.py`(LoRA+bf16+grad checkpoint 조합 메모리 측정),
`scripts/compare_smolvla_ard.py`(base/ard/symmetric 변형 간 파라미터·메모리·속도 비교),
`scripts/layer_importance.py`(SmolLM2 레이어 중요도 분석) — 전부
**[docs/profiling_tools.md](docs/profiling_tools.md)** 에 사용법과 검증 기록이 있습니다.

## Layout

- `requirements.txt` — research tooling installed on top of lerobot (notebook/plotting deps, including `pytest`). torch and lerobot itself are installed by `scripts/install.sh`, not listed here.
- `scripts/install.sh` — environment setup: CPU/GPU-aware torch install, editable `lerobot[smolvla]` install from `third_party/lerobot`, then `requirements.txt`.
- `scripts/check_env.py` — import + CPU-fallback smoke test (standalone; not under `tests/`, see [docs/ard.md](docs/ard.md) note below).
- `scripts/train_ard.py` — SmolVLA(+ARD) 전용 최소 학습 스크립트 (lerobot의 범용 학습 CLI 대체).
- `scripts/count_params.py` — 구성 요소별 파라미터 집계 + 해상도별 이미지 토큰 수 실측.
- `scripts/profile_memory.py` — LoRA/bf16/gradient-checkpointing 조합 메모리 프로파일링.
- `scripts/compare_smolvla_ard.py` — base/ard/symmetric 변형 간 파라미터·메모리·속도 비교.
- `scripts/compare_bridge_attention.py` — Bridge Attention on/off loss 곡선 비교.
- `scripts/layer_importance.py` — SmolLM2 백본 레이어 중요도(코사인 유사도 기반) 분석.
- `scripts/compare_lac_modes.py` — LAC none/covla/role_aware 모드 비교(`select_action()` 경로, 합성 백본, jerk/ρ/ω/α).
- `scripts/compare_lac_closed_loop_modes.py` — 같은 비교를 ARD-VLA 2.0 폐루프 경로(`run_closed_loop`+Judge)로 끝까지.
- `third_party/lerobot/` — vendored, editable LeRobot/SmolVLA source.
  - `src/lerobot/policies/smolvla/ard.py` — ARD heads/losses, `ForceHead`, GradNorm, Bridge Attention, 대칭 대조군 손실.
  - `src/lerobot/policies/smolvla/freq_policy.py` — FreqPolicy 주파수 일관성 손실.
  - `src/lerobot/policies/smolvla/token_pruning.py` — 비전 토큰 프루닝.
  - `src/lerobot/policies/smolvla/judge.py` — 폐루프 구조의 Judge/Selector head.
  - `src/lerobot/policies/smolvla/decision_logic.py` — 폐루프 구조의 결정 로직(torch 의존성 없음).
  - `src/lerobot/policies/smolvla/closed_loop.py` — 폐루프 구조의 환경-무관 러너.
  - `src/lerobot/policies/smolvla/lac.py` — LAC(Latent-Aware Controller) 추론 시점 EMA 필터.
  - `src/lerobot/policies/smolvla/configuration_smolvla.py` / `modeling_smolvla.py` — 위 기능들을 켜고 끄는 config 플래그와 forward/추론 연결 지점.
- `tests/` — pytest 테스트 전부(`test_ard.py`, `test_ard_integration.py`, `test_freq_policy.py`, `test_token_pruning.py`, `test_judge.py`, `test_decision_logic.py`, `test_closed_loop.py`, `test_lac.py`, `test_lac_closed_loop.py`, `test_env.py`, `conftest.py`의 합성 SmolVLM fixture 포함).
- `pytest.ini` — `pytest` 루트 실행 설정(`testpaths=tests`, `requires_hub` 마커).
- `.github/workflows/tests.yml` — CPU 전용 pytest CI.
- `docs/` — 기능별 상세 설계/사용법/검증 기록(`ard.md`, `bridge_attention.md`, `freq_policy.md`, `token_pruning.md`, `profiling_tools.md`, `closed_loop.md`, `lac.md`).
- `PROGRESS.md` — 멀티세션 작업 진행 기록(사용량 한도로 세션이 끊겨도 다음 세션이 이어갈 수 있도록).
