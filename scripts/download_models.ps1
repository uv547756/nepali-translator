# Download all model weights required by the translator (Windows).
# Run from the repository root.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$ModelsDir = Join-Path $RepoRoot "models"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "ERROR: .venv not found. Run scripts\setup.ps1 first."
}

Write-Host "==> Downloading models to $ModelsDir"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# --- Faster-Whisper large-v3 --------------------------------------------------
$WhisperDir = Join-Path $ModelsDir "faster-whisper-large-v3"
if (-not (Test-Path $WhisperDir)) {
    Write-Host "--> Downloading Faster-Whisper large-v3 (INT8)..."
    & $PythonExe -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Systran/faster-whisper-large-v3',
    local_dir=r'$WhisperDir',
    local_dir_use_symlinks=False,
)
print('Faster-Whisper large-v3 downloaded.')
"
} else {
    Write-Host "--> Faster-Whisper large-v3 already exists, skipping."
}

# --- SeamlessM4T v2 large -----------------------------------------------------
$SeamlessDir = Join-Path $ModelsDir "seamless-m4t-v2-large"
$hasWeights  = (Test-Path (Join-Path $SeamlessDir "model.safetensors")) -or
               (Test-Path (Join-Path $SeamlessDir "pytorch_model.bin")) -or
               (@(Get-ChildItem (Join-Path $SeamlessDir "model-*.safetensors") -ErrorAction SilentlyContinue)).Count -gt 0

if (-not $hasWeights) {
    Write-Host "--> Downloading SeamlessM4T v2 large (~4.5 GB, this will take a while)..."
    New-Item -ItemType Directory -Force -Path $SeamlessDir | Out-Null
    & $PythonExe -c "
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
import torch
print('Downloading processor...')
AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large').save_pretrained(r'$SeamlessDir')
print('Downloading model weights (FP16)...')
SeamlessM4Tv2ForTextToText.from_pretrained(
    'facebook/seamless-m4t-v2-large', torch_dtype=torch.float16
).save_pretrained(r'$SeamlessDir')
print('SeamlessM4T v2 large downloaded.')
"
} else {
    Write-Host "--> SeamlessM4T v2 large already exists (weights present), skipping."
}

# --- Silero VAD ONNX ----------------------------------------------------------
$SileroFile = Join-Path $ModelsDir "silero_vad.onnx"
if (-not (Test-Path $SileroFile)) {
    Write-Host "--> Downloading Silero VAD v5 ONNX..."
    $url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    Invoke-WebRequest -Uri $url -OutFile $SileroFile
    Write-Host "    Silero VAD downloaded."
} else {
    Write-Host "--> Silero VAD already exists, skipping."
}

# --- Piper TTS (en_US lessac medium) ------------------------------------------
$PiperDir  = Join-Path $ModelsDir "piper"
$PiperOnnx = Join-Path $PiperDir "en_US-lessac-medium.onnx"
New-Item -ItemType Directory -Force -Path $PiperDir | Out-Null

if (-not (Test-Path $PiperOnnx)) {
    Write-Host "--> Downloading Piper TTS (en_US-lessac-medium)..."
    $base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
    Invoke-WebRequest -Uri "$base/en_US-lessac-medium.onnx"      -OutFile $PiperOnnx
    Invoke-WebRequest -Uri "$base/en_US-lessac-medium.onnx.json" -OutFile "$PiperOnnx.json"
    Write-Host "    Piper TTS downloaded."
} else {
    Write-Host "--> Piper TTS already exists, skipping."
}

Write-Host ""
Write-Host "==> All models downloaded."
Write-Host ""
Write-Host "Model sizes:"
Get-ChildItem $ModelsDir | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    "{0,8:N0} MB  {1}" -f ($size / 1MB), $_.Name
}
