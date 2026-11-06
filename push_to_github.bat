@echo off
title Push to GitHub
cd /d "%~dp0"

set "GIT_CMD="
if exist "C:\Users\hanwi\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe" (
    set "GIT_CMD=C:\Users\hanwi\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
) else (
    where git >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "GIT_CMD=git"
    )
)

if "%GIT_CMD%"=="" (
    echo [ERROR] Git was not found. Please install Git or GitHub Desktop.
    pause
    exit /b 1
)

echo ========================================================
echo Staging and Committing Changes...
echo ========================================================
"%GIT_CMD%" add .
"%GIT_CMD%" commit -m "Update research benchmark: remove non-academic terms and finalize manuscript structure"

echo.
echo ========================================================
echo Pushing Microgrid Digital Twin to GitHub...
echo ========================================================
"%GIT_CMD%" push -u origin main

echo.
echo ========================================================
echo Finished! Check your GitHub repository:
echo https://github.com/hanwithabandi-droid/resilient-microgrid-digital-twin
echo ========================================================
pause
