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

## ARD: Asymmetric Role Decomposition

Per the ARD-VLA research plan (bimanual tool-use fine-tuning: one arm stabilizes the workpiece,
the other performs the precise manipulation), SmolVLA's action expert now supports an optional
asymmetric dual-head mode:

- `third_party/lerobot/src/lerobot/policies/smolvla/ard.py` (new file) — `RoleClassifier` (predicts
  which arm is the Actuator from the pooled language instruction), `AsymmetricResidualHeads`
  (zero-init residual MLPs that specialize the Stabilizer/Actuator channels on top of the shared
  flow-matching output), and `compute_ard_losses` (`alpha * L_stab + beta * L_act`, with
  `L_stab = L_pos + λ·L_smooth` and `L_act = L_pos + λ·L_force + λ·L_traj`, matching the plan's
  loss design — `L_pos` reuses the base model's own flow-matching regression loss rather than
  recomputing a separate L1 term).
- `configuration_smolvla.py` — new `use_ard`, `ard_arm_dim`, `ard_alpha`/`ard_beta`,
  `ard_lambda_{smooth,force,traj}`, `ard_default_actuator_arm`, `ard_use_role_classifier`,
  `ard_role_loss_weight` fields, off by default (`use_ard=False` reproduces upstream SmolVLA
  exactly — verified no code path changes unless the flag is set).
- `modeling_smolvla.py` — `VLAFlowMatching.forward` (training loss), `.sample_actions`/
  `.denoise_step` (inference, including under RTC) all apply the same residual-head correction and
  role resolution, so training and inference stay consistent.
- Optional per-sample batch keys, both no-ops if absent: `ard_actuator_is_first` (ground-truth role
  label, e.g. from a scripted Isaac Sim episode) and `ard_force_target` (contact-force/torque
  supervision — no dataset in this repo provides it yet, so `L_force` is 0 until one does).

**Verification.** This sandbox's network policy blocks the Hugging Face Hub, and `SmolVLAPolicy`
always needs to download the SmolVLM2 backbone's config even with `load_vlm_weights=False` — so it
can't be instantiated end-to-end here. What *is* verified, offline, in `scripts/test_ard.py`:
`SmolVLAConfig`'s new validation, the role split/combine round-trip, role-resolution priority
(label > classifier > static default), zero-init/gradient-flow of the residual heads, and
`compute_ard_losses`'s numerics (including a real bug it caught: `L_force` broadcasting a
per-sample target against a per-timestep prediction). `scripts/check_env.py` confirms the default
(`use_ard=False`) path still imports and runs unchanged. Run both with:

```bash
python scripts/check_env.py
python scripts/test_ard.py
```

Exercising a real forward/backward pass through `SmolVLAPolicy` itself (with `use_ard=True`, tiny
VLM dims) is the natural next verification step once this environment has Hub access.

## Layout

- `requirements.txt` — research tooling installed on top of lerobot (notebook/plotting deps). torch and lerobot itself are installed by `scripts/install.sh`, not listed here.
- `scripts/install.sh` — environment setup: CPU/GPU-aware torch install, editable `lerobot[smolvla]` install from `third_party/lerobot`, then `requirements.txt`.
- `scripts/check_env.py` — import + CPU-fallback smoke test.
- `scripts/test_ard.py` — offline unit tests for the ARD modification.
- `third_party/lerobot/` — vendored, editable LeRobot/SmolVLA source.
