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

REM Check 7-Zip (need both 7z.exe AND 7z.dll for .rar support)
if exist "C:\Program Files\7-Zip\7z.exe" (
    if exist "C:\Program Files\7-Zip\7z.dll" (
        echo [OK] 7-Zip found at C:\Program Files\7-Zip\ (with .rar support)
    ) else (
        echo [WARN] 7-Zip found but 7z.dll missing — .rar may not work
    )
) else if exist "%~dp07z.exe" (
    if exist "%~dp07z.dll" (
        echo [OK] Portable 7z.exe + 7z.dll found (with .rar support)
    ) else (
        echo [WARN] Portable 7z.exe found but 7z.dll missing — .rar will NOT work
        echo        Place 7z.dll next to 7z.exe, or install full 7-Zip
    )
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
