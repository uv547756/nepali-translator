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

# ── 3. PyTorch with CUDA 12.1 ──────────────────────────────────────────────
echo "==> Installing PyTorch (CUDA 12.1)"
pip install --quiet \
    torch==2.3.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

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
