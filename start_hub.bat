@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Physical AI Workshop - Start Hub
:: Activates the virtual environment, starts the server,
:: waits until it is healthy, then opens the browser.
:: Run this on workshop day instead of setup.bat.
:: ============================================================

cd /d "%~dp0"
set "VENV_DIR=%~dp0.venv"

echo ============================================================
echo   Physical AI Workshop - Starting Hub
echo ============================================================
echo.

:: Check venv exists
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
    echo No virtual environment found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:: Kill any existing server on port 8000
netstat -ano 2>nul | findstr ":8000 " | findstr "LISTENING" > nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Stopping existing process on port 8000...
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
        taskkill /PID %%P /F >nul 2>&1
    )
    timeout /t 1 /nobreak >nul
)

:: Delegate everything to start_hub.py — it kills port 8000, launches the
:: server in a new window using the venv Python directly (no temp files),
:: polls until healthy, then opens the browser.
echo Starting hub via start_hub.py...
"%VENV_DIR%\Scripts\python.exe" "%~dp0start_hub.py"
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo start_hub.py reported an error. Check the hub window for details.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Hub is running. Use Chrome or Edge.
echo   Close the "Physical AI Workshop Hub" window to stop.
echo ============================================================
echo.
pause
exit /b 0
