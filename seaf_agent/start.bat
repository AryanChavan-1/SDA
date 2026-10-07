@echo off
setlocal enabledelayedexpansion
title SEAF-SDA: Sovereign-Edge Agentic Framework

:: Set UTF-8 encoding for Windows terminal
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Move to parent directory so relative package paths resolve
cd /d "%~dp0.."

call "%~dp0..\start_seaf.bat"
