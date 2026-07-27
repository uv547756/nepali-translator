# Bootstrap the Nepali Translator server environment on Windows.
# Run from the repository root in PowerShell (as normal user, NOT admin).
# Requires: Python 3.10+, Git, NVIDIA drivers with nvidia-smi in PATH.

param(
    [switch]$SkipNode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$VenvDir   = Join-Path $RepoRoot ".venv"
$PtIndex   = "https://download.pytorch.org/whl/cu128"

Write-Host "==> Nepali Translator Setup (Windows)"
Write-Host "    Repo: $RepoRoot"

# --- Set KMP_DUPLICATE_LIB_OK to silence OpenMP duplicate DLL warning --------
[System.Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "    KMP_DUPLICATE_LIB_OK=TRUE written to user environment"

# --- Check prerequisites ------------------------------------------------------
foreach ($cmd in @("python", "git")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "ERROR: '$cmd' not found in PATH. Install it and re-run."
    }
}

if (-not (Get-Command "espeak-ng" -ErrorAction SilentlyContinue)) {
    Write-Warning "espeak-ng not found. Piper TTS requires it."
    Write-Host "  Install: winget install --id espeak-ng.espeak-ng"
    Write-Host "  Then re-run this script."
}

# --- Python virtual environment -----------------------------------------------
Write-Host "==> Creating Python virtual environment: $VenvDir"
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"

& $PipExe install --upgrade pip wheel setuptools --quiet

# --- PyTorch: detect CUDA version from nvidia-smi and pick wheel index --------
$smiOut = ""
try { $smiOut = & nvidia-smi 2>$null } catch {}

if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    Write-Host "    Detected CUDA $major.$minor"

    if ($major -eq 11) {
        $PtIndex = "https://download.pytorch.org/whl/cu118"
    } elseif ($major -eq 12 -and $minor -le 1) {
        $PtIndex = "https://download.pytorch.org/whl/cu121"
    } elseif ($major -eq 12 -and $minor -le 4) {
        $PtIndex = "https://download.pytorch.org/whl/cu124"
    } elseif ($major -eq 12) {
        $PtIndex = "https://download.pytorch.org/whl/cu126"
    }
    # CUDA 13+ keeps cu128 default
} else {
    Write-Host "    Could not detect CUDA version -- using cu128 index (RTX 5090 default)"
}

Write-Host "==> Installing PyTorch (latest stable, index: $PtIndex)"
& $PipExe install torch torchaudio --index-url $PtIndex

# --- Python dependencies ------------------------------------------------------
Write-Host "==> Installing Python dependencies"
& $PipExe install -r (Join-Path $RepoRoot "server\requirements.txt")

# --- Node.js / npm for the web client -----------------------------------------
if (-not $SkipNode) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "==> Installing Node dependencies"
        Push-Location (Join-Path $RepoRoot "client")
        npm install --silent
        Pop-Location
    } else {
        Write-Host "==> npm not found -- skipping client build (install Node.js >= 20)"
    }
}

# --- Models directory ---------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "models") | Out-Null

Write-Host ""
Write-Host "==> Setup complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download models:   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1"
Write-Host "  2. Run offline test:  .\translator.ps1 --mode offline --input test.wav --output out.wav"
Write-Host "  3. Run server:        .\translator.ps1 --mode server --config configs\default.yaml"
Write-Host ""
