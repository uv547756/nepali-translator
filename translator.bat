@echo off
REM Launcher for cmd.exe (use translator.ps1 from PowerShell instead).
REM Usage: translator.bat --mode server --config configs\default.yaml

setlocal
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: .venv not found at %REPO_ROOT%.venv
    echo Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    exit /b 1
)

REM Silence the harmless duplicate-OpenMP-DLL abort on Windows
set "KMP_DUPLICATE_LIB_OK=TRUE"

"%VENV_PYTHON%" -m server.main %*
