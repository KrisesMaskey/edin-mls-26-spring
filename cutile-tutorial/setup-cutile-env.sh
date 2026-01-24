#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
ENV_NAME="cutile"
PYTHON_VERSION="3.11"
CUDA_TAG="cuda13x"

# =========================
# Sanity hints (non-fatal)
# =========================
echo ">>> Assumptions:"
echo "    - NVIDIA driver >= r580"
echo "    - CUDA Toolkit >= 13.1"
echo "    - Blackwell GPU (CC 10.x / 12.x)"
echo

# =========================
# Check uv
# =========================
if ! command -v uv >/dev/null 2>&1; then
	echo "❌ uv not found."
	echo "Install with:"
	echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi

# =========================
# Create virtual env
# =========================
if [ ! -d ".venv" ]; then
	echo ">>> Creating virtual environment (.venv)"
	uv venv .venv --python ${PYTHON_VERSION}
else
	echo ">>> Reusing existing .venv"
fi

source .venv/bin/activate

# =========================
# Upgrade base tooling
# =========================
echo ">>> Upgrading pip / setuptools / wheel"
uv pip install --upgrade pip setuptools wheel

# =========================
# Core CUDA Python stack
# =========================
echo ">>> Installing CUDA Python stack (CUDA 13)"

# CuPy for CUDA 13
uv pip install \
	"cupy-${CUDA_TAG}"

# NVIDIA CUDA Python bindings (driver/runtime API)
uv pip install \
	cuda-python

# cuTile Python
uv pip install \
	cuda-tile

# =========================
# Optional but recommended
# =========================
echo ">>> Installing optional tooling"

# NVML access (driver introspection, useful for debugging)
uv pip install pynvml

# NumPy (used by almost all examples)
uv pip install numpy

# =========================
# Freeze snapshot
# =========================
echo ">>> Writing lock snapshot (requirements.lock)"
uv pip freeze >requirements.lock

# =========================
# Done
# =========================
echo
echo "✅ cuTile Python environment is ready."
echo
echo "Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Installed key packages:"
echo "  - cupy-${CUDA_TAG}"
echo "  - cuda-python"
echo "  - cuda-tile"
echo
