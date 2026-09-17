@echo off
setlocal
cd /d "%~dp0.."
python tools\build_common_acc_features.py
if errorlevel 1 exit /b 1
python tools\cross_validate_forth_trace.py
if errorlevel 1 exit /b 1
python tools\cross_dataset_activity.py
if errorlevel 1 exit /b 1
