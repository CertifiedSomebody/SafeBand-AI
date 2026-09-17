@echo off
setlocal
python tools\activity_hierarchical_v3.py
if errorlevel 1 exit /b %errorlevel%
echo.
echo Activity Hierarchical V3 complete.
endlocal
