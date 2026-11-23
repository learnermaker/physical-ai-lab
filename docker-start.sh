#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null 2>&1 || { echo "Install Docker first: https://docs.docker.com/get-started/get-docker/"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required."; exit 1; }
echo "Building and starting Physical AI Lab. First run downloads dependencies."
docker compose up --build -d
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/api/config >/dev/null 2>&1; then
    echo "Lab ready: http://localhost:8000"
    if command -v open >/dev/null 2>&1; then open http://localhost:8000
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:8000 >/dev/null 2>&1 &
    fi
    exit 0
  fi
  sleep 2
done
echo "Health check timed out. Inspect: docker compose logs --tail=100"
exit 1
