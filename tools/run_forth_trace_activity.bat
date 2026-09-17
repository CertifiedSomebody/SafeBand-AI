@echo off
setlocal
cd /d "%~dp0.."

echo [1/3] Preparing FORTH-TRACE left-wrist features...
python tools\prepare_forth_trace.py --data-dir datasets\raw\FORTH_TRACE\FORTH_TRACE_DATASET-master --out-dir datasets\processed\forth_trace
if errorlevel 1 exit /b 1

echo [2/3] Training core activity model...
python tools\train_forth_trace_activity.py --input datasets\processed\forth_trace\core_activity_features.csv --out-dir models\forth_trace_activity_core
if errorlevel 1 exit /b 1

echo [3/3] Training transition-aware model...
python tools\train_forth_trace_activity.py --input datasets\processed\forth_trace\transition_aware_features.csv --out-dir models\forth_trace_activity_transition
if errorlevel 1 exit /b 1

echo.
echo FORTH-TRACE Activity V2 pipeline completed.
endlocal
