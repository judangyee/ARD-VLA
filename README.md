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

## Layout

- `requirements.txt` — research tooling installed on top of lerobot (notebook/plotting deps). torch and lerobot itself are installed by `scripts/install.sh`, not listed here.
- `scripts/install.sh` — environment setup: CPU/GPU-aware torch install, editable `lerobot[smolvla]` install from `third_party/lerobot`, then `requirements.txt`.
- `scripts/check_env.py` — import + CPU-fallback smoke test.
- `third_party/lerobot/` — vendored, editable LeRobot/SmolVLA source.
