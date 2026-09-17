@echo off
cd /d "%~dp0..\.."
python tools\build_activity_raw_windows.py
if errorlevel 1 exit /b 1
python tools\train_activity_dl_v1.py
