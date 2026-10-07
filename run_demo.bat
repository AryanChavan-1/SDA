@echo off
setlocal enabledelayedexpansion
title SEAF-SDA: Research Verification Suite

chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

set "VENV_PYTHON=%~dp0Agentic AI\micro_sda\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto RUN_VENV
goto RUN_SYS

:RUN_VENV
"%VENV_PYTHON%" "%~dp0seaf_agent\run_seaf_demo.py"
goto FINISH

:RUN_SYS
python "%~dp0seaf_agent\run_seaf_demo.py"
goto FINISH

:FINISH
echo.
pause
