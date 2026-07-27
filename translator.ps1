# Windows launcher - activates the venv and runs the server.
# Usage: .\translator.ps1 --mode server --config configs\default.yaml
# From cmd.exe use translator.bat instead.

$RepoRoot   = $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: .venv not found at $RepoRoot\.venv" -ForegroundColor Red
    Write-Host "Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1"
    exit 1
}

# Silence the harmless duplicate-OpenMP-DLL abort on Windows
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

& $VenvPython -m server.main @args
exit $LASTEXITCODE
