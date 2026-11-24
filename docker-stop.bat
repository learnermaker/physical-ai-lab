@echo off
setlocal

:: ============================================================
:: Physical AI Lab - Docker stop (Windows)
:: Stops the container WITHOUT deleting participant files,
:: models or caches.
:: ============================================================

cd /d "%~dp0"

echo Stopping Physical AI Lab container (files are kept)...
docker compose down
if %ERRORLEVEL% neq 0 (
  echo [WARN] 'docker compose down' reported an issue - see above.
  pause
  exit /b 1
)
echo [OK] Stopped. Your files, models and caches are preserved.
echo      Restart with: docker-start.bat
echo.
pause
exit /b 0
