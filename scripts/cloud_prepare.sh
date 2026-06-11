#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CERA_ENV_NAME:-cera-iad}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-cu121}"
IAD_ROOT="${IAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "[CERA-IAD] root: ${IAD_ROOT}"
echo "[CERA-IAD] env: ${ENV_NAME}, python: ${PYTHON_VERSION}, cuda: ${CUDA_VERSION}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda on Ubuntu first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

case "${CUDA_VERSION}" in
  cpu)
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu118)
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ;;
  cu121)
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ;;
  cu124)
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ;;
  *)
    python -m pip install torch torchvision torchaudio
    ;;
esac

python -m pip install -r "${IAD_ROOT}/requirements-cloud.txt"

export PYTHONPATH="${IAD_ROOT}:${PYTHONPATH:-}"
echo "[CERA-IAD] PYTHONPATH=${PYTHONPATH}"
echo "[CERA-IAD] environment ready. Activate with: conda activate ${ENV_NAME}"
