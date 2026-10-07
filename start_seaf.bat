@echo off
setlocal enabledelayedexpansion
title SEAF-SDA: Sovereign-Edge Agentic Framework

:: Set UTF-8 encoding for Windows terminal
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Move to current script directory
cd /d "%~dp0"

echo ====================================================================
echo    SEAF-SDA: Sovereign-Edge Agentic Framework (Edge AI PC Edition)
echo    Hardware Target: NVIDIA RTX 3050/2050 + Intel Core i5 + 16GB RAM
echo ====================================================================
echo.

:: 1. Check & Start Ollama daemon
echo [*] Checking local Ollama service (http://localhost:11434)...
curl.exe -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 goto START_OLLAMA
echo [+] Ollama daemon is running.
goto DETECT_PYTHON

:START_OLLAMA
echo [!] Ollama is not currently responding. Starting Ollama daemon in background...
start "" /B ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul
curl.exe -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Notice: Could not connect to Ollama. Local generation will use edge fallback.
) else (
    echo [+] Ollama daemon connected successfully.
)

:: 2. Locate Python executable (using labels to avoid Windows path parenthesis parsing bugs)
:DETECT_PYTHON
echo.
set "VENV_PYTHON=%~dp0Agentic AI\micro_sda\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto USE_VENV
goto CHECK_SYSTEM_PY

:USE_VENV
set "PY_CMD=%VENV_PYTHON%"
echo [+] Virtual Environment Python detected.
goto MENU

:CHECK_SYSTEM_PY
where python >nul 2>&1
if %errorlevel% neq 0 goto NO_PY
set "PY_CMD=python"
echo [!] Using System Python: python
goto MENU

:NO_PY
echo [ERROR] Python not found. Please ensure Python is installed or virtual environment exists.
pause
exit /b 1

:: 3. Interactive Launch Menu
:MENU
echo.
echo ====================================================================
echo                     SELECT EXECUTION MODE
echo ====================================================================
echo   [1] Launch SEAF Desktop GUI (CustomTkinter App) [DEFAULT]
echo   [2] Run Research Verification Demo (run_seaf_demo.py)
echo   [3] Run Autonomous Coding Agent Test (test_coding_run.py)
echo   [4] Run Full Unit Test Suite (tests)
echo   [5] Exit
echo ====================================================================
set /p CHOICE="Enter choice [1-5] (Press Enter for 1): "

if "%CHOICE%"=="" set CHOICE=1
if "%CHOICE%"=="1" goto LAUNCH_GUI
if "%CHOICE%"=="2" goto RUN_DEMO
if "%CHOICE%"=="3" goto RUN_TEST
if "%CHOICE%"=="4" goto RUN_UNIT_TESTS
if "%CHOICE%"=="5" goto EXIT_SCRIPT

echo [!] Invalid selection. Please enter 1, 2, 3, 4, or 5.
goto MENU

:LAUNCH_GUI
echo.
echo [*] Starting SEAF-SDA Desktop GUI...
"%PY_CMD%" "%~dp0seaf_agent\app.py"
echo.
echo [*] GUI closed. Returning to menu...
goto MENU

:RUN_DEMO
echo.
echo [*] Executing SEAF 3-Pillar Research Verification Suite...
"%PY_CMD%" "%~dp0seaf_agent\run_seaf_demo.py"
echo.
pause
goto MENU

:RUN_TEST
echo.
echo [*] Executing Autonomous Coding Agent CLI Test...
"%PY_CMD%" "%~dp0seaf_agent\test_coding_run.py"
echo.
pause
goto MENU

:RUN_UNIT_TESTS
echo.
echo [*] Running All SEAF Unit Tests...
"%PY_CMD%" -c "import subprocess, sys, glob; tests = glob.glob('seaf_agent/tests/test_*.py'); [print('PASS: ' + t) if subprocess.run([sys.executable, t], capture_output=True).returncode == 0 else print('FAIL: ' + t) for t in tests]; print('\nAll test checks completed.')"
echo.
pause
goto MENU

:EXIT_SCRIPT
echo Exiting SEAF-SDA.
exit /b 0

:END
echo.
echo [*] Process finished.
pause
