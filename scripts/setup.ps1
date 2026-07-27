# Bootstrap the Nepali Translator server environment on Windows.
# Run from the repository root in PowerShell (as normal user, NOT admin).
# Requires: Python 3.10+, Git, NVIDIA drivers with nvidia-smi in PATH.

param(
    [switch]$SkipNode  # pass -SkipNode to skip npm install
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir  = Join-Path $RepoRoot ".venv"

Write-Host "==> Nepali Translator Setup (Windows)" -ForegroundColor Cyan
Write-Host "    Repo: $RepoRoot"

# ── 0. Set KMP_DUPLICATE_LIB_OK to avoid OpenMP conflicts on Windows ────────
[System.Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "    KMP_DUPLICATE_LIB_OK=TRUE set (user env)"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
foreach ($cmd in @("python", "git", "curl")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "ERROR: '$cmd' not found in PATH. Install it and re-run."
    }
}

# espeak-ng (required by Piper phonemizer)
if (-not (Get-Command "espeak-ng" -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Warning "espeak-ng not found. Piper TTS needs it for phonemization."
    Write-Host "  Install via winget:  winget install --id espeak-ng.espeak-ng"
    Write-Host "  Or download from:    https://github.com/espeak-ng/espeak-ng/releases"
    Write-Host "  Then re-run this script."
    Write-Host ""
}

# ── 2. Python virtual environment ─────────────────────────────────────────────
Write-Host "==> Creating Python virtual environment: $VenvDir"
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"

& $PipExe install --upgrade pip wheel setuptools --quiet

# ── 3. PyTorch — detect CUDA version from nvidia-smi ─────────────────────────
$PtIndex = "https://download.pytorch.org/whl/cu128"  # default: newest stable

try {
    $smiOut = & nvidia-smi 2>$null
    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        $PtIndex = switch ($major) {
            11 { "https://download.pytorch.org/whl/cu118" }
            12 {
                if     ($minor -le 1) { "https://download.pytorch.org/whl/cu121" }
                elseif ($minor -le 4) { "https://download.pytorch.org/whl/cu124" }
                else                  { "https://download.pytorch.org/whl/cu126" }
            }
            default { "https://download.pytorch.org/whl/cu128" }  # 13.x / 5090
        }
        Write-Host "    Detected CUDA $major.$minor"
    }
} catch {
    Write-Host "    nvidia-smi not found — defaulting to cu128 index"
}

Write-Host "==> Installing PyTorch (latest stable, index: $PtIndex)"
& $PipExe install torch torchaudio --index-url $PtIndex

# ── 4. Python dependencies ────────────────────────────────────────────────────
Write-Host "==> Installing Python dependencies"
& $PipExe install -r (Join-Path $RepoRoot "server\requirements.txt")

# ── 5. Node.js / npm for the web client ──────────────────────────────────────
if (-not $SkipNode) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "==> Installing Node dependencies"
        Push-Location (Join-Path $RepoRoot "client")
        npm install --silent
        Pop-Location
    } else {
        Write-Host "==> npm not found — skipping client build (install Node.js >= 20)"
    }
}

# ── 6. Models directory ───────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "models") | Out-Null

Write-Host ""
Write-Host "==> Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download models:"
Write-Host "       powershell scripts\download_models.ps1"
Write-Host "  2. Start the server:"
Write-Host "       .\translator.ps1 --mode server --config configs\default.yaml"
Write-Host "  3. Offline test:"
Write-Host "       .\translator.ps1 --mode offline --input test.wav --output out.wav"
Write-Host ""
