# ARD-VLA

Research environment for experimenting with [SmolVLA](https://huggingface.co/docs/lerobot/smolvla), the small vision-language-action policy from Hugging Face's [LeRobot](https://github.com/huggingface/lerobot) framework.

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

ARD-VLA 연구계획서(양손 도구 조작 파인튜닝: 한 팔은 작업물을 고정하고, 다른 팔이 정밀한
조작을 수행)에 따라, SmolVLA의 액션 전문가(action expert)가 선택적인 비대칭 듀얼 헤드 모드를
지원하도록 확장했습니다. **Actuator 팔은 config로 고정되며, 항상 오른팔입니다** — 지시문마다
역할이 바뀌지 않습니다. `ard_default_actuator_arm`이 그 역할을 맡을 팔을 지정하고, 모든
샘플에 동일하게 적용됩니다.

- `third_party/lerobot/src/lerobot/policies/smolvla/ard.py` (신규 파일) — `AsymmetricResidualHeads`
  (공유 flow-matching 출력 위에서 Stabilizer/Actuator 채널을 특화시키는 zero-init residual
  MLP), `resolve_actuator_is_first`(고정된 왼팔/오른팔 라우팅), `compute_ard_losses`
  (`alpha * L_stab + beta * L_act`, `L_stab = L_pos + λ·L_smooth`, `L_act = L_pos + λ·L_force +
  λ·L_traj` — 연구계획서의 손실 설계를 그대로 따름. `L_pos`는 별도의 L1 항을 다시 계산하지
  않고 베이스 모델 자체의 flow-matching 회귀 손실을 재사용합니다).
- `configuration_smolvla.py` — 새 `use_ard`, `ard_arm_dim`, `ard_alpha`/`ard_beta`,
  `ard_lambda_{smooth,force,traj}`, `ard_default_actuator_arm`(기본값 `"right"`) 필드 추가,
  기본값은 전부 꺼짐 (`use_ard=False`이면 업스트림 SmolVLA와 완전히 동일하게 동작 — 플래그를
  켜지 않는 한 코드 경로가 전혀 바뀌지 않음을 검증함).
- `modeling_smolvla.py` — `VLAFlowMatching.forward`(학습 손실), `.sample_actions`/
  `.denoise_step`(추론, RTC 포함) 모두 동일한 residual head 보정과 고정 역할 라우팅을
  적용해서 학습과 추론이 서로 어긋나지 않도록 함.
- 선택적 샘플별 배치 키, 없으면 아무 영향 없음: `ard_force_target` (접촉력/토크 supervision —
  현재 이 레포의 어떤 데이터셋도 이 값을 제공하지 않아서, 데이터셋이 생기기 전까지 `L_force`는
  항상 0).

**검증.** 이 샌드박스의 네트워크 정책이 Hugging Face Hub를 막고 있고, `SmolVLAPolicy`는
`load_vlm_weights=False`여도 SmolVLM2 백본 config를 항상 다운로드해야 해서 — 여기서는
end-to-end로 생성해볼 수 없었습니다. 대신 `scripts/test_ard.py`에서 오프라인으로 검증한
내용: `SmolVLAConfig`의 새 검증 로직, 고정 역할 라우팅이 항상 오른팔 채널을 Actuator head로
보내는지, 역할 split/combine 라운드트립, residual head의 zero-init/gradient 흐름,
`compute_ard_losses`의 수치 계산(이 과정에서 실제 버그도 하나 잡았습니다: `L_force`가
샘플별 타겟을 타임스텝별 예측값과 브로드캐스팅하려던 문제). `scripts/check_env.py`로는
기본(`use_ard=False`) 경로가 여전히 그대로 import/실행되는 것도 확인했습니다. 둘 다 아래로
실행할 수 있습니다:

```bash
python scripts/check_env.py
python scripts/test_ard.py
```

`SmolVLAPolicy` 자체로 실제 forward/backward pass를 돌려보는 것(`use_ard=True`, 작은 VLM
차원으로)이 이 환경에 Hub 접근이 가능해지면 진행할 다음 검증 단계입니다.

## 학습 (로컬 GPU 환경)

lerobot의 범용 학습 CLI(`lerobot_train.py`)는 모든 정책을 알아야 하는 `policies/factory.py`에
의존해서 트림할 때 같이 지웠습니다. 대신 `scripts/train_ard.py`가 SmolVLA 하나만 아는 최소
학습 루프입니다: `LeRobotDataset` 로드 → `SmolVLAConfig`/`SmolVLAPolicy` 생성 → optimizer/
scheduler 빌드 → 학습 루프 → 주기적 체크포인트 저장. `use_ard=True`일 때는 `ard_stabilizer_loss`
등 손실 breakdown도 함께 로깅됩니다.

```bash
python scripts/train_ard.py \
    --dataset-repo-id <HF_USER>/<DATASET> \
    --output-dir outputs/ard_run1 \
    --steps 20000 \
    --batch-size 32
```

`--no-use-ard`를 주면 ARD 없이 베이스라인 SmolVLA만 학습합니다. 시작할 때 액션 채널의
왼팔/오른팔 예상 순서를 출력해주니, 실제 로봇 배선과 맞는지 눈으로 한 번 확인하세요 (ARD는
"앞 `ard_arm_dim`개=왼팔, 다음 `ard_arm_dim`개=오른팔"이라는 관례를 가정할 뿐, 데이터셋이
실제로 그 순서인지는 검증하지 않습니다).

이 스크립트도 이 샌드박스에서는 end-to-end로 돌려보지 못했습니다(Hub 접근 차단) — 대신
헬퍼 함수들(`split_policy_features`, `warn_if_action_layout_looks_wrong`)은 합성 데이터로
직접 검증했고, import/인자 파싱도 확인했습니다. 실제 학습 루프 자체는 로컬 GPU 환경에서
처음 돌려보실 때 검증해주세요.

## 파라미터 구성 확인 (`scripts/count_params.py`)

SmolVLA(+ARD) 전체 파라미터를 비전 인코더 / LLM(SmolLM2) / Action Expert / ARD head /
나머지 shim 레이어로 나눠서 개수와 비중을 보여주고, 지정한 해상도별로 이미지가 실제로 몇 개의
토큰이 되는지도 출력합니다. `load_vlm_weights=False`라 전체 가중치를 받지는 않지만, config는
Hugging Face Hub에서 받아야 해서 이 샌드박스에서는 실행이 안 됩니다 (compile/import/CLI 파싱은
확인했고, 실행하면 예상대로 네트워크 호출 단계에서 막히는 것까지 확인했습니다).

```bash
python scripts/count_params.py --resolutions 384 512 768
```

중요: SmolVLA는 SmolLM2-360M의 레이어를 전부 쓰지 않고 `config.num_vlm_layers`(기본값 16)개만
물리적으로 잘라서 씁니다(`smolvlm_with_expert.py`의 `text_model.layers = ...[:num_vlm_layers]`).
이 스크립트는 원본 레이어 수와 실제 사용하는 레이어 수를 둘 다 보여줘서 이 부분을 헷갈리지
않게 합니다.

## Gradient checkpointing (`SmolVLMWithExpertModel.gradient_checkpointing_enable()`)

`SmolVLMWithExpertModel.forward()`는 VLM과 action expert를 레이어 단위로 번갈아 호출하는
커스텀 루프라서, transformers의 표준 `model.gradient_checkpointing_enable()` 훅이 걸리지
않습니다 (그 훅은 서브모듈의 표준 `forward()` 호출 경로를 가로채는데, 이 루프는 그 경로를
쓰지 않습니다). 그래서 이번에 별도로 추가했습니다:

```python
vlm_expert = policy.model.vlm_with_expert
vlm_expert.gradient_checkpointing_enable()   # 켜기
vlm_expert.gradient_checkpointing_disable()  # 끄기
```

내부적으로 레이어 하나의 본문을 `_run_layer()`로 뽑아내고, 학습 중(`self.training`)이면서
KV 캐시를 안 쓰는 forward 경로(`use_cache=False`, `fill_kv_cache=False` — 즉 학습 시
`VLAFlowMatching.forward`가 실제로 쓰는 경로)에서만 `torch.utils.checkpoint.checkpoint(
self._run_layer, ..., use_reentrant=False)`로 감쌉니다. 추론(`sample_actions`/`denoise_step`,
KV 캐시 사용)에는 영향이 없습니다.

이 기능은 실제 SmolVLM2 모델로 검증하지 못했습니다(Hub 접근 차단, GPU 없음) — 대신 동일한
호출 시그니처(텐서 리스트 + `None` + non-tensor 인자가 섞인 형태)를 흉내 낸 가짜 레이어로
체크포인팅 유무에 따라 forward 출력과 gradient가 정확히 일치하는지 별도로 검증했습니다.
`check_env.py`/`test_ard.py`는 이 플래그가 기본 `False`라 회귀 없이 통과합니다.

## 메모리 프로파일링 (`scripts/profile_memory.py`)

LoRA + bf16 autocast + gradient checkpointing을 모두 켠 상태에서, bimanual 액션
(14 DoF, `chunk_size=50`) 더미 배치로 forward+backward를 한 번 돌려 배치 사이즈별
(`1, 2, 4, 8, 16, 32` 기본값) `torch.cuda.max_memory_allocated()` 최대 메모리를 표로 출력하는
스크립트입니다. Colab/Kaggle 노트북에서 GPU 런타임으로 바로 돌릴 수 있게 단일 파일로
작성했습니다:

```bash
!pip install -e "third_party/lerobot[smolvla,peft]"
!python scripts/profile_memory.py
```

`--no-lora`, `--no-bf16`, `--no-grad-checkpoint`, `--no-ard`로 각 기법을 개별적으로
끄고 비교할 수 있고, 배치 사이즈 도중 OOM이 나도 스크립트가 죽지 않고 해당 칸을 "OOM"으로
표시한 뒤 나머지 배치 사이즈를 계속 시도합니다.

LoRA는 기존에 있던 `PreTrainedPolicy.wrap_with_peft()`를 그대로 사용합니다 — 다만
SmolVLA의 기본 LoRA 타겟(`lm_expert`의 attention projection들)에는 ARD의
`stabilizer_head`/`actuator_head`가 포함되지 않아서, `wrap_with_peft()`가 나머지 전부를
얼린 뒤 ARD head를 명시적으로 다시 `requires_grad_(True)`로 풀어줍니다 (그렇지 않으면
ARD head가 통째로 학습에서 빠집니다).

기본 실행은 SmolVLA의 레이어 프루닝(`num_vlm_layers`로 SmolLM2를 앞쪽 몇 개 레이어만 쓰도록
자르는 것) 적용 여부를 **둘 다** 프로파일링해서 표를 두 개 냅니다 — "적용 O"는
`--num-vlm-layers`(기본 16)로 자른 기본 SmolVLA 설정, "적용 X"는 원본 SmolLM2 레이어 수를
그대로 쓰는 모델입니다. 원본 레이어 수 쪽은 `num_expert_layers`가 기본 `-1`이라 action
expert도 VLM 레이어 수를 따라가며 같이 커지므로 메모리를 훨씬 많이 쓰고 더 빨리 OOM이 날 수
있습니다. `--layer-pruning-mode pruned` 또는 `unpruned`를 주면 그중 하나만 돌려서 시간을
아낄 수 있습니다.

이 스크립트는 실제 Colab GPU 런타임에서 검증했습니다. 처음 실행할 때 두 가지 문제가
나올 수 있는데(둘 다 이 레포/스크립트 버그는 아니고 Colab 환경 특성입니다):
- `pip install -e ...`가 torch/torchvision을 재설치하면서 "You must restart the runtime"
  경고가 뜨면, 재시작 후 새 셀에서 `%cd`부터 다시 하고 스크립트만 실행하세요 (설치를 다시 할
  필요는 없습니다).
- `wrap_with_peft()`가 peft의 `dispatch_torchao` 단계에서 `ImportError: Found an
  incompatible version of torchao`를 던지면, Colab에 미리 깔린 `torchao`가 peft 요구
  버전과 안 맞아서입니다 — 저희는 양자화를 안 쓰니 `!pip uninstall -y torchao`로 지우고
  다시 실행하면 됩니다.

## 기본 SmolVLA vs ARD-VLA 비교 (`scripts/compare_smolvla_ard.py`)

`profile_memory.py`와 같은 패턴(LoRA + bf16 + gradient checkpointing 기본 켜짐)으로, 이번엔
"레이어 프루닝 O/X"가 아니라 **모델 두 개**를 같은 배치 사이즈(`1, 4, 16, 32` 기본값)로 비교합니다:

- **기본 SmolVLA**: 단일팔 7 DoF, `use_ard=False` — 원본 액션 헤드만 사용
- **ARD-VLA**: bimanual 14 DoF, `use_ard=True` — `AsymmetricResidualHeads` 적용

```bash
!python scripts/compare_smolvla_ard.py
```

출력은 표 두 개입니다: (1) 변형별 전체 파라미터 수 / LoRA 학습 대상 파라미터 수, (2) 배치
사이즈별 두 변형의 peak memory(GB)와 forward pass 시간(ms)을 나란히 놓은 비교표. 메모리는
`profile_memory.py`와 동일하게 forward+backward 기준(실제 학습 스텝의 메모리 최고점을 반영),
시간은 backward를 뺀 forward 단독 기준입니다 — 이 둘을 같은 배치에서 한 번에 재느라, 시간
쪽엔 별도 warmup이 없어서 첫 호출(특히 batch_size=1)은 CUDA 커널 초기화 비용이 섞여 다소
부풀려질 수 있습니다.

`--variants base` 또는 `--variants ard`로 한쪽만 돌릴 수 있고, 나머지 옵션(`--no-lora`,
`--no-bf16`, `--no-grad-checkpoint`, `--lora-r`/`--lora-alpha`, `--batch-sizes` 등)은
`profile_memory.py`와 동일하게 동작합니다.

이 스크립트는 `profile_memory.py`와 같은 검증된 패턴을 그대로 재사용했지만, 스크립트 자체를
실제 GPU에서 돌려보지는 못했습니다 — `py_compile`/CLI 파싱 확인 외에, 표 출력 로직(OOM 셀
처리 포함)은 가짜 데이터로 직접 검증했습니다.

## Layout

- `requirements.txt` — research tooling installed on top of lerobot (notebook/plotting deps). torch and lerobot itself are installed by `scripts/install.sh`, not listed here.
- `scripts/install.sh` — environment setup: CPU/GPU-aware torch install, editable `lerobot[smolvla]` install from `third_party/lerobot`, then `requirements.txt`.
- `scripts/check_env.py` — import + CPU-fallback smoke test.
- `scripts/test_ard.py` — offline unit tests for the ARD modification.
- `scripts/train_ard.py` — SmolVLA(+ARD) 전용 최소 학습 스크립트 (lerobot의 범용 학습 CLI 대체).
- `scripts/count_params.py` — 구성 요소별 파라미터 집계 + 해상도별 이미지 토큰 수 실측.
- `third_party/lerobot/` — vendored, editable LeRobot/SmolVLA source.
