#!/usr/bin/env bash
# Install the SmolVLA research environment.
#
# - Creates/uses a virtualenv at .venv (skip with --no-venv to install into
#   the current interpreter, e.g. inside a container that already isolates
#   the environment).
# - Detects whether a CUDA GPU is available and installs the matching torch
#   build (CPU-only wheels when no GPU is present, avoiding a multi-GB CUDA
#   download on machines that can't use it).
# - Installs the rest of requirements.txt (lerobot[smolvla] + research
#   tooling) on top.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
USE_VENV=1
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"

for arg in "$@"; do
  case "$arg" in
    --no-venv) USE_VENV=0 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--no-venv]" >&2
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "$USE_VENV" -eq 1 ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
fi

echo "==> Using interpreter: $("$PYTHON_BIN" --version)"
"$PYTHON_BIN" -m pip install --upgrade pip

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
fi

if [ "$HAS_GPU" -eq 1 ]; then
  echo "==> NVIDIA GPU detected: installing CUDA-enabled torch ${TORCH_VERSION}"
  "$PYTHON_BIN" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
else
  echo "==> No GPU detected: installing CPU-only torch ${TORCH_VERSION}"
  if ! "$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; then
    echo "==> download.pytorch.org unreachable (blocked network?); falling back to PyPI torch build" >&2
    "$PYTHON_BIN" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
  fi
fi

echo "==> Installing lerobot[smolvla] and research dependencies"
"$PYTHON_BIN" -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "==> Verifying the environment imports cleanly"
"$PYTHON_BIN" "${REPO_ROOT}/scripts/check_env.py"

echo "==> Done. Activate with: source ${VENV_DIR}/bin/activate"
