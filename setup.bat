@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Physical AI Workshop - Windows Setup Script
:: Created by Jim Seelan
::
:: Requires nothing pre-installed. Downloads Python 3.12
:: if needed, creates a .venv, installs all dependencies.
:: Safe to re-run. Works from any terminal or double-click.
:: ============================================================

cd /d "%~dp0"

:: Free port 8000 if something is already using it
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr /r "\b8000\b"') do taskkill /F /PID %%P >nul 2>&1

echo ============================================================
echo   Physical AI Workshop - Setup
echo   Created by Jim Seelan
echo ============================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE="

:: ============================================================
:: STEP 1 - Find Python 3.12
::
:: Uses for/f to expand env vars via a subprocess, which bypasses
:: both the Windows App Execution Alias (Store python stub) and
:: the for-list variable expansion limitation in batch.
:: ============================================================
echo [1/9] Locating Python 3.12...

:: 1a - per-user install (most common on Windows 11)
for /f "delims=" %%D in ('cmd /c "echo %LOCALAPPDATA%\Programs\Python\Python312"') do (
    IF EXIST "%%D\python.exe" (
        "%%D\python.exe" --version 2>&1 | findstr /r "3\.12\." >nul 2>&1
        IF !ERRORLEVEL! EQU 0 (
            set "PYTHON_EXE=%%D\python.exe"
            echo [OK] Found: !PYTHON_EXE!
            goto :check_venv
        )
    )
)

:: 1b - py launcher (registered to C:\Windows\py.exe by official installer)
py -3.12 --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
    echo [OK] Found via py launcher: !PYTHON_EXE!
    goto :check_venv
)

:: 1c - system-wide installs
IF EXIST "C:\Program Files\Python312\python.exe" (
    set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
    echo [OK] Found: !PYTHON_EXE!
    goto :check_venv
)
IF EXIST "C:\Python312\python.exe" (
    set "PYTHON_EXE=C:\Python312\python.exe"
    echo [OK] Found: !PYTHON_EXE!
    goto :check_venv
)

:: 1d - not found - download and install
echo Python 3.12 not found. Downloading installer (~25 MB)...
echo.
set "PY_INSTALLER=%TEMP%\python312_setup.exe"
powershell -NoProfile -Command ^
    "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '%PY_INSTALLER%' -UseBasicParsing"
IF %ERRORLEVEL% NEQ 0 (
    echo Download failed. Check your internet connection.
    echo Install Python 3.12 manually: https://www.python.org/downloads/
    goto :fail
)
echo [OK] Installer downloaded.
echo Installing Python 3.12 (a progress window will appear briefly)...
start /wait "" "%PY_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
:: Exit codes: 0=success  1=cancelled  3010=success+reboot pending
IF %ERRORLEVEL% EQU 1 (
    echo Installation was cancelled.
    goto :fail
)
IF %ERRORLEVEL% GTR 3010 (
    echo Python installation failed. Install manually: https://www.python.org/downloads/
    goto :fail
)
echo [OK] Python 3.12 installed.
for /f "delims=" %%D in ('cmd /c "echo %LOCALAPPDATA%\Programs\Python\Python312"') do (
    IF EXIST "%%D\python.exe" (
        set "PYTHON_EXE=%%D\python.exe"
        echo [OK] Python ready: !PYTHON_EXE!
        goto :check_venv
    )
)
echo Could not locate python.exe after install. Please restart and re-run setup.bat.
goto :fail

:: ============================================================
:: STEP 2 - Create virtual environment
:: ============================================================
:check_venv
echo.
echo [2/9] Setting up virtual environment...

IF EXIST "%VENV_DIR%\Scripts\python.exe" (
    :: Verify the venv is healthy - its base Python may have been deleted
    "%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
    IF !ERRORLEVEL! EQU 0 (
        echo [OK] Virtual environment healthy.
        goto :activate
    ) ELSE (
        echo [!] Broken .venv detected - base Python was deleted. Recreating...
        rmdir /s /q "%VENV_DIR%"
        echo [OK] Removed broken .venv.
    )
)

"!PYTHON_EXE!" -m venv "%VENV_DIR%"
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
    echo Failed to create virtual environment.
    goto :fail
)
echo [OK] Virtual environment created.

:: ============================================================
:: STEP 3 - Activate
:: ============================================================
:activate
echo.
echo [3/9] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
IF "%VIRTUAL_ENV%"=="" (
    echo Activation failed - VIRTUAL_ENV not set.
    goto :fail
)
echo [OK] Active: %VIRTUAL_ENV%
echo.

:: Upgrade pip (non-fatal)
python -m pip install --upgrade pip --quiet 2>nul

:: ============================================================
:: STEP 4 - PyTorch CPU-only
:: ============================================================
echo [4/9] Installing PyTorch CPU-only (~200 MB, may take a few minutes)...
:: Try 2.14.0 first (current stable as of Sep 2026).
:: If the wheel isn't on the index yet, fall back to whatever is latest stable.
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [!] torch==2.14.0 not found on stable index - trying latest available...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    IF %ERRORLEVEL% NEQ 0 (
        echo PyTorch install failed. Check internet and re-run setup.bat.
        goto :fail
    )
)
echo [OK] PyTorch installed.
echo.

:: ============================================================
:: STEP 5 - Workshop dependencies
:: ============================================================
echo [5/9] Installing workshop dependencies...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo Dependency install failed. Check internet and re-run setup.bat.
    goto :fail
)
echo [OK] Dependencies installed.
echo.

:: ============================================================
:: STEP 6 - Jupyter kernel
:: ============================================================
echo [6/9] Registering Jupyter kernel...
python -m ipykernel install --user --name physical-ai --display-name "Physical AI Workshop"
IF %ERRORLEVEL% NEQ 0 (
    echo Warning: Kernel registration failed. Select it manually in VS Code.
) ELSE (
    echo [OK] Kernel "Physical AI Workshop" registered.
)
echo.

:: ============================================================
:: STEP 7 - hand_landmarker.task model
:: ============================================================
echo [7/9] Checking hand_landmarker.task model...
IF EXIST "assets\hand_landmarker.task" (
    echo [OK] Already present.
    goto :step8
)
echo Downloading hand_landmarker.task (~8 MB^)...
python "%~dp0download_model.py"
IF %ERRORLEVEL% NEQ 0 (
    echo Warning: Download failed. Place the file in assets\ manually.
)
:step8
echo.

:: ============================================================
:: STEP 8 - Verify installation
:: ============================================================
echo [8/9] Verifying installation...
echo.
python verify_install.py
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Completed with warnings - see above.
    echo [WARN] = fallback active   [FAIL] = needs attention
) ELSE (
    echo [OK] All checks passed.
)
echo.

:: ============================================================
:: STEP 9 - Launch hub and open browser
:: ============================================================
echo [9/9] Launching Physical AI Workshop hub...
echo.

python start_hub.py
IF %ERRORLEVEL% NEQ 0 (
    echo Warning: Hub launcher failed. Start it manually:
    echo   python start_hub.py
)

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Hub:     http://localhost:8000  (new window)
echo   Browser: opening now
echo.
echo   Use Chrome or Edge - Firefox is not supported.
echo.
echo   Tip: run  python start_hub.py  any time to restart the hub.
echo        Open exercise.py files in VS Code alongside the browser.
echo ============================================================
echo.
pause
exit /b 0

:: ============================================================
:fail
echo.
echo ============================================================
echo   Setup did not complete. See error above.
echo   Fix the issue and re-run setup.bat - it is safe to retry.
echo ============================================================
echo.
pause
exit /b 1

