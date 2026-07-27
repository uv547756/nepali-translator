# Windows launcher — activates the venv and runs the server.
# Usage: .\translator.ps1 [--mode server|offline] [options]

$RepoRoot  = $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "ERROR: .venv not found at $RepoRoot\.venv — run: powershell scripts\setup.ps1"
}

# Suppress the OpenMP duplicate DLL warning (harmless on Windows)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

& $VenvPython -m server.main @args
