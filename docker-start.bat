@echo off
setlocal
cd /d "%~dp0"
echo Physical AI Lab - Docker quick start
docker version >nul 2>&1
if errorlevel 1 (
  echo Docker is unavailable. Install and start Docker Desktop:
  echo https://docs.docker.com/get-started/get-docker/
  pause
  exit /b 1
)
docker compose version >nul 2>&1
if errorlevel 1 (
  echo Docker Compose v2 is unavailable. Update Docker Desktop.
  pause
  exit /b 1
)
echo Building and starting the lab. The first run downloads dependencies.
docker compose up --build -d
if errorlevel 1 goto :fail
echo Waiting for the hub...
for /l %%i in (1,1,60) do (
  curl -fsS http://localhost:8000/api/config >nul 2>&1 && goto :ready
  timeout /t 2 /nobreak >nul
)
echo Health check timed out. Inspect: docker compose logs --tail=100
start "" http://localhost:8000
goto :end
:ready
echo Lab is ready: http://localhost:8000
start "" http://localhost:8000
goto :end
:fail
echo Startup failed. Inspect the error above or run docker compose logs.
pause
exit /b 1
:end
echo.
echo To stop the lab: docker compose down
pause
