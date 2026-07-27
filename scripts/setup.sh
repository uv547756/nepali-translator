#!/usr/bin/env bash
# Bootstrap the Nepali Translator server environment.
# Supports Arch Linux (pacman) and Ubuntu/Debian (apt).
# Run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

echo "==> Nepali Translator Setup"
echo "    Repo: $REPO_ROOT"

# ── 0. Redirect pip cache + tmp to the largest available partition ──────────
# PyTorch wheels are ~2.5 GB; pip's cache + unpack temp can exceed user quotas
# on the root partition. Use /mnt/media if it has more free space than /tmp.
_root_free=$(df --output=avail / 2>/dev/null | tail -1 | tr -d ' ')
_media_free=0
if mountpoint -q /mnt/media 2>/dev/null; then
    _media_free=$(df --output=avail /mnt/media 2>/dev/null | tail -1 | tr -d ' ')
fi

if [ "${_media_free}" -gt "${_root_free}" ]; then
    _pip_tmp="/mnt/media/.pip-tmp-$(id -un)"
    mkdir -p "$_pip_tmp"
    export PIP_CACHE_DIR="$_pip_tmp/cache"
    export TMPDIR="$_pip_tmp/tmp"
    mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
    echo "    pip cache → $_pip_tmp (more space than /)"
else
    # Still avoid polluting ~/.cache with multi-GB wheels; use a local cache
    export PIP_CACHE_DIR="$REPO_ROOT/.pip-cache"
    mkdir -p "$PIP_CACHE_DIR"
fi

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

# ── 3. PyTorch — detect CUDA major version and pick matching wheel index ───
_cuda_major=""
if command -v nvcc &>/dev/null; then
    _cuda_major=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
elif command -v nvidia-smi &>/dev/null; then
    _cuda_major=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
fi

# cu126 is the newest stable index; works on CUDA 12.6+ and 13.x (backward compat)
case "$_cuda_major" in
    11)   _pt_index="https://download.pytorch.org/whl/cu118" ;;
    12)
        # Detect minor to pick cu121 vs cu124 vs cu126
        _cuda_minor=""
        if command -v nvcc &>/dev/null; then
            _cuda_minor=$(nvcc --version 2>/dev/null | grep -oP 'release [0-9]+\.\K[0-9]+' | head -1)
        elif command -v nvidia-smi &>/dev/null; then
            _cuda_minor=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: [0-9]+\.\K[0-9]+' | head -1)
        fi
        if   [ "${_cuda_minor:-99}" -le 1 ]; then _pt_index="https://download.pytorch.org/whl/cu121"
        elif [ "${_cuda_minor:-99}" -le 4 ]; then _pt_index="https://download.pytorch.org/whl/cu124"
        else                                       _pt_index="https://download.pytorch.org/whl/cu126"
        fi ;;
    *)    _pt_index="https://download.pytorch.org/whl/cu126" ;;   # 13.x and future
esac

echo "==> Installing PyTorch (latest stable, index: $_pt_index)"
pip install torch torchaudio --index-url "$_pt_index" || {
    echo "    Retrying with --no-cache-dir (quota fallback)..."
    pip install --no-cache-dir torch torchaudio --index-url "$_pt_index"
}

# ── 4. Python dependencies ─────────────────────────────────────────────────
echo "==> Installing Python dependencies"
pip install -r "$REPO_ROOT/server/requirements.txt" || {
    echo "    Retrying with --no-cache-dir (quota fallback)..."
    pip install --no-cache-dir -r "$REPO_ROOT/server/requirements.txt"
}

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
