@echo off
setlocal enabledelayedexpansion
title SEAF-SDA: Desktop GUI Launcher

:: Set UTF-8 encoding
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

:: Check Ollama
echo [*] Checking Ollama daemon...
curl.exe -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting Ollama in background...
    start "" /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
)

:START_GUI
:: Locate Python
set "VENV_PYTHON=%~dp0Agentic AI\micro_sda\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto RUN_VENV
goto RUN_SYS

:RUN_VENV
echo [*] Launching SEAF Desktop GUI (Venv)...
"%VENV_PYTHON%" "%~dp0seaf_agent\app.py"
goto FINISH

:RUN_SYS
echo [*] Launching SEAF Desktop GUI (Python)...
python "%~dp0seaf_agent\app.py"
goto FINISH

:FINISH
echo.
set /p RESTART="GUI closed. Restart? (y/n): "
if /i "%RESTART%"=="y" goto START_GUI
exit /b 0
