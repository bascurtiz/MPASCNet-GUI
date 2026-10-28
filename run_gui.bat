@echo off
title MPASCNet Web Studio
cd /d "%~dp0"
echo =======================================================
echo   Starting MPASCNet Web GUI (Spek-Alternative Analyzer)
echo =======================================================
start http://localhost:7860
python web_gui.py
pause
