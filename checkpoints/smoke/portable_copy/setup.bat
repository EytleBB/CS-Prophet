@echo off
echo === CS Prophet Portable Setup ===
echo.

REM Prefer bundled Python, fall back to system Python
set "PY="
if exist "%~dp0dist\python\python.exe" (
    set "PY=%~dp0dist\python\python.exe"
    echo [OK] Using bundled Python
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No bundled or system Python found.
        echo         Run prepare_usb.py on your own machine first,
        echo         or install Python 3.10+ on this machine.
        pause
        exit /b 1
    )
    set "PY=python"
    echo [OK] Using system Python
)

set "PYTHONNOUSERSITE=1"

%PY% --version
echo.

REM Check 7-Zip
if exist "C:\Program Files\7-Zip\7z.exe" (
    echo [OK] 7-Zip found at C:\Program Files\7-Zip\7z.exe
) else if exist "%~dp07z.exe" (
    echo [OK] Portable 7z.exe found
) else if exist "%~dp0bin\7z.exe" (
    echo [OK] Portable bin\7z.exe found
) else (
    echo [WARN] 7-Zip not found. .rar demos will fail to extract.
    echo        Install 7-Zip or place 7z.exe + 7z.dll next to this script.
)

echo.
echo Choose install mode:
echo   1) Download-only (minimal deps)
echo   2) Full pipeline  (includes parsing)
echo   0) Skip install (deps already installed)
echo.
set /p choice="Enter 0/1/2: "

if "%choice%"=="1" (
    echo Installing download-only dependencies...
    if exist "%~dp0dist\wheels" (
        %PY% -m pip install -r "%~dp0requirements.txt" --no-index --find-links "%~dp0dist\wheels" --no-warn-script-location
    ) else (
        %PY% -m pip install -r "%~dp0requirements.txt" --no-warn-script-location
    )
) else if "%choice%"=="2" (
    echo Installing full pipeline dependencies...
    if exist "%~dp0dist\wheels" (
        %PY% -m pip install -r "%~dp0requirements_pipeline.txt" --no-index --find-links "%~dp0dist\wheels" --no-warn-script-location
    ) else (
        %PY% -m pip install -r "%~dp0requirements_pipeline.txt" --no-warn-script-location
    )
) else (
    echo Skipping dependency install.
)

echo.
echo === Setup complete ===
echo.
echo Usage:
echo   Download only:  %PY% run_download.py
echo   Full pipeline:  %PY% run_pipeline.py
echo   Dry run:        %PY% run_download.py --dry-run
echo.
echo Or use the launcher scripts:
echo   launch_run_download.bat
echo   launch_run_pipeline.bat
echo.
pause
