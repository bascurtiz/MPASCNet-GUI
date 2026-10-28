@echo off
title MPASCNet Web Studio
cd /d "%~dp0"
echo =======================================================
echo   Starting MPASCNet Web GUI (Spek-Alternative Analyzer)
echo =======================================================

if exist "..\..\vanilla\MPASCNet-main\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\vanilla\MPASCNet-main\.venv\Scripts\python.exe"
) else if exist "D:\vanilla\MPASCNet-main\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=D:\vanilla\MPASCNet-main\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

start http://localhost:7860
"%PYTHON_EXE%" web_gui.py %*
pause
