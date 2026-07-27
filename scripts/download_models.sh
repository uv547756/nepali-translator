#!/usr/bin/env bash
# Download all model weights required by the translator.
# Models are saved to the models/ directory (git-ignored).
# Requires: Python venv activated, huggingface_hub installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: .venv not found. Run scripts/setup.sh first."
    exit 1
fi

echo "==> Downloading models to $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# ── Faster-Whisper large-v3 (INT8 CTranslate2 format) ─────────────────────
WHISPER_DIR="$MODELS_DIR/faster-whisper-large-v3"
if [[ ! -d "$WHISPER_DIR" ]]; then
    echo "--> Downloading Faster-Whisper large-v3 (INT8)..."
    "$VENV_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Systran/faster-whisper-large-v3',
    local_dir='$WHISPER_DIR',
    local_dir_use_symlinks=False,
)
print('Faster-Whisper large-v3 downloaded.')
"
else
    echo "--> Faster-Whisper large-v3 already exists, skipping."
fi

# ── SeamlessM4T v2 large ───────────────────────────────────────────────────
SEAMLESS_DIR="$MODELS_DIR/seamless-m4t-v2-large"
if [[ ! -d "$SEAMLESS_DIR" ]]; then
    echo "--> Downloading SeamlessM4T v2 large (this is ~4.5 GB — may take a while)..."
    "$VENV_PYTHON" -c "
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
import torch
print('Downloading processor...')
AutoProcessor.from_pretrained(
    'facebook/seamless-m4t-v2-large'
).save_pretrained('$SEAMLESS_DIR')
print('Downloading model weights (FP16)...')
SeamlessM4Tv2ForTextToText.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    torch_dtype=torch.float16,
).save_pretrained('$SEAMLESS_DIR')
print('SeamlessM4T v2 large downloaded.')
"
else
    echo "--> SeamlessM4T v2 large already exists, skipping."
fi

# ── Silero VAD ONNX ────────────────────────────────────────────────────────
SILERO_FILE="$MODELS_DIR/silero_vad.onnx"
if [[ ! -f "$SILERO_FILE" ]]; then
    echo "--> Downloading Silero VAD v5 ONNX..."
    curl -fsSL \
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
        -o "$SILERO_FILE"
    echo "Silero VAD downloaded."
else
    echo "--> Silero VAD already exists, skipping."
fi

# ── Piper TTS — en_US lessac medium ────────────────────────────────────────
PIPER_DIR="$MODELS_DIR/piper"
mkdir -p "$PIPER_DIR"

PIPER_ONNX="$PIPER_DIR/en_US-lessac-medium.onnx"
if [[ ! -f "$PIPER_ONNX" ]]; then
    echo "--> Downloading Piper TTS (en_US-lessac-medium)..."
    BASE_URL="https://github.com/rhasspy/piper/releases/download/v1.2.0"
    curl -fsSL "$BASE_URL/en_US-lessac-medium.onnx"      -o "$PIPER_ONNX"
    curl -fsSL "$BASE_URL/en_US-lessac-medium.onnx.json" -o "$PIPER_ONNX.json"
    echo "Piper TTS downloaded."
else
    echo "--> Piper TTS already exists, skipping."
fi

# ── DeepFilterNet3 (optional noise reduction) ──────────────────────────────
DF_DIR="$MODELS_DIR/DeepFilterNet3"
if [[ ! -d "$DF_DIR" ]]; then
    echo "--> Downloading DeepFilterNet3..."
    "$VENV_PYTHON" -c "
from df import init_df
# init_df downloads the model to the default cache, then we copy
import shutil, os
model, state, info = init_df()
# deepfilternet stores models in its own cache; we symlink for clarity
print('DeepFilterNet3 ready (cached by deepfilternet library).')
" 2>/dev/null || echo "    (DeepFilterNet download skipped — install deepfilternet first)"
else
    echo "--> DeepFilterNet3 already exists, skipping."
fi

echo ""
echo "==> All models downloaded to $MODELS_DIR"
echo ""
echo "Model sizes:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
