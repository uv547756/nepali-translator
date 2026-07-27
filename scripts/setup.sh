#!/usr/bin/env bash
# Bootstrap the Nepali Translator server environment.
# Supports Arch Linux (pacman) and Ubuntu/Debian (apt).
# Run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

echo "==> Nepali Translator Setup"
echo "    Repo: $REPO_ROOT"

# ── 1. Detect distro and install system packages ───────────────────────────
if command -v pacman &>/dev/null; then
    echo "==> Installing system packages (Arch Linux)"
    sudo pacman -S --needed --noconfirm \
        cuda cudnn \
        python python-pip python-venv \
        espeak-ng \
        libsndfile \
        ffmpeg \
        curl wget git \
        2>/dev/null || true
elif command -v apt-get &>/dev/null; then
    echo "==> Installing system packages (Ubuntu/Debian)"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        espeak-ng libespeak-ng1 \
        libsndfile1 \
        ffmpeg \
        curl wget git \
        2>/dev/null || true
else
    echo "==> Unsupported package manager — skipping system packages"
fi

# ── 2. Python virtual environment ──────────────────────────────────────────
echo "==> Creating Python virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools --quiet

# ── 3. PyTorch — detect CUDA version and pick matching wheel ───────────────
_cuda_ver=""
if command -v nvcc &>/dev/null; then
    _cuda_ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
elif command -v nvidia-smi &>/dev/null; then
    _cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
fi

# Map major.minor → PyTorch wheel tag (newest stable that supports it)
_pt_version="2.5.1"
_pt_index="https://download.pytorch.org/whl/cu124"   # default: CUDA 12.4
case "${_cuda_ver%%.*}" in   # match on major version only
    12)
        case "$_cuda_ver" in
            12.1) _pt_index="https://download.pytorch.org/whl/cu121" ;;
            12.4) _pt_index="https://download.pytorch.org/whl/cu124" ;;
            12.6|12.7|12.8|12.9)
                  _pt_version="2.6.0"
                  _pt_index="https://download.pytorch.org/whl/cu126" ;;
            *)    _pt_index="https://download.pytorch.org/whl/cu124" ;;
        esac ;;
    11)   _pt_index="https://download.pytorch.org/whl/cu118" ;;
esac

echo "==> Installing PyTorch ${_pt_version} (CUDA ${_cuda_ver:-unknown}, index: $_pt_index)"
pip install --quiet \
    "torch==${_pt_version}" \
    "torchaudio==${_pt_version}" \
    --index-url "$_pt_index"

# ── 4. Python dependencies ─────────────────────────────────────────────────
echo "==> Installing Python dependencies"
pip install --quiet -r "$REPO_ROOT/server/requirements.txt"

# ── 5. Node.js / npm for the web client ────────────────────────────────────
if command -v npm &>/dev/null; then
    echo "==> Installing Node dependencies"
    cd "$REPO_ROOT/client"
    npm install --silent
    cd "$REPO_ROOT"
else
    echo "==> npm not found — skipping client build (install Node.js >= 20 manually)"
fi

# ── 6. Models directory ────────────────────────────────────────────────────
mkdir -p "$REPO_ROOT/models"
echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download models:"
echo "       bash scripts/download_models.sh"
echo "  2. Start the server:"
echo "       source .venv/bin/activate"
echo "       python -m server.main --config configs/default.yaml"
echo "  3. (Optional) Build the web client:"
echo "       cd client && npm run build"
echo ""
