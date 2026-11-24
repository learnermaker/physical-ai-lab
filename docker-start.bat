@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Physical AI Lab - Docker quick start (Windows)
:: Checks Docker, builds if needed, starts the lab, waits for
:: health, opens the browser. Double-click friendly.
:: ============================================================

cd /d "%~dp0"

:: HOST_PORT may be set in the environment before calling this script:
::   set HOST_PORT=8001 && docker-start.bat
:: (compose.yaml honours the same variable, so both stay in sync.)
if "%HOST_PORT%"=="" set "HOST_PORT=8000"
if not "%HOST_PORT_OVERRIDE%"=="" set "HOST_PORT=%HOST_PORT_OVERRIDE%"

set "HEALTH_URL=http://localhost:%HOST_PORT%/api/config"

echo ============================================================
echo   Physical AI Lab - Docker start
echo ============================================================
echo.

:: --- 1. Docker installed? ---
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo [FAIL] Docker not found. Install Docker Desktop:
  echo        https://www.docker.com/products/docker-desktop/
  echo        Then re-run docker-start.bat.
  goto :fail
)
for /f "delims=" %%v in ('docker --version 2^>^&1') do echo [OK] %%v

:: --- 2. Docker daemon running? ---
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo [FAIL] Docker daemon is not running. Start Docker Desktop and retry.
  goto :fail
)
echo [OK] Docker daemon running.

:: --- 3. Compose v2? ---
docker compose version >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo [FAIL] 'docker compose' (v2) not found. Update Docker Desktop.
  goto :fail
)
for /f "delims=" %%v in ('docker compose version 2^>^&1') do echo [OK] %%v

:: --- 4. Port warning (non-fatal) ---
netstat -ano ^| findstr /r ":%HOST_PORT% .*LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
  echo [WARN] Port %HOST_PORT% already in use - container may fail to bind.
  echo        Stop the conflicting process or set HOST_PORT=8001 and re-run.
)

:: --- 5. Model assets (idempotent; uses host python if available) ---
if exist download_model.py (
  echo.
  echo Checking model assets (skips what already exists)...
  where python >nul 2>&1
  if %ERRORLEVEL% equ 0 (
    python download_model.py
    if %ERRORLEVEL% neq 0 echo [WARN] Asset download had issues - continuing.
  ) else (
    echo [WARN] python not on PATH - assets will be checked inside the container.
  )
)

:: --- 6. Build + start ---
echo.
echo Building image if needed and starting container...
docker compose up --build -d
if %ERRORLEVEL% neq 0 (
  echo [FAIL] 'docker compose up' failed. See output above.
  goto :fail
)

:: --- 7. Wait for backend health (up to 90 s) ---
echo Waiting for backend at %HEALTH_URL% ...
set "READY=0"
for /l %%i in (1,1,90) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%HEALTH_URL%' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
  if !ERRORLEVEL! equ 0 (
    set "READY=1"
    goto :healthy_done
  )
  timeout /t 1 /nobreak >nul
)
:healthy_done

echo.
if "%READY%"=="1" (
  echo [OK] Lab is healthy at http://localhost:%HOST_PORT%
) else (
  echo [WARN] Backend did not respond within 90 s.
  echo        Check logs: docker compose logs --tail=100
  echo        Opening browser anyway - it may need a manual refresh.
)

:: --- 8. Open browser ---
start "" "http://localhost:%HOST_PORT%"

echo.
echo   Hub:     http://localhost:%HOST_PORT%
echo   Logs:    docker compose logs -f
echo   Verify:  docker compose exec lab python verify_docker.py
echo.
echo   Stop with: docker-stop.bat  (files are kept)
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   Docker start did not complete. Fix the issue above and retry.
echo ============================================================
echo.
pause
exit /b 1
