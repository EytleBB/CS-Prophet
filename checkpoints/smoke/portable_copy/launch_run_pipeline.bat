@echo off
echo === CS Prophet Portable (pipeline) ===
set "PYTHONNOUSERSITE=1"
"%~dp0dist\python\python.exe" "%~dp0run_pipeline.py" %*
if errorlevel 1 pause
