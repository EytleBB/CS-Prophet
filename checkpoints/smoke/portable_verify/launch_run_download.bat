@echo off
echo === CS Prophet Portable (download-only) ===
set "PYTHONNOUSERSITE=1"
"%~dp0dist\python\python.exe" "%~dp0run_download.py" %*
if errorlevel 1 pause
