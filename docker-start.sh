#!/usr/bin/env bash
# Physical AI Lab — Docker quick start (macOS / Linux).
# Checks Docker, builds if needed, starts the lab, waits for health, opens browser.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST_PORT="${HOST_PORT:-8000}"
HEALTH_URL="http://localhost:${HOST_PORT}/api/config"
COMPOSE="docker compose"

echo "============================================================"
echo "  Physical AI Lab — Docker start"
echo "============================================================"

# --- 1. Docker installed? ---
if ! command -v docker &>/dev/null; then
  echo "[FAIL] Docker not found. Install Docker Desktop:"
  echo "       https://www.docker.com/products/docker-desktop/"
  exit 1
fi
echo "[OK] Docker: $(docker --version)"

# --- 2. Docker daemon running? ---
if ! docker info &>/dev/null; then
  echo "[FAIL] Docker daemon is not running. Start Docker Desktop and retry."
  exit 1
fi
echo "[OK] Docker daemon running."

# --- 3. Compose v2 available? ---
if ! docker compose version &>/dev/null; then
  echo "[FAIL] 'docker compose' (v2) not found. Update Docker Desktop to a recent version."
  exit 1
fi
echo "[OK] Compose: $(docker compose version --short 2>/dev/null || docker compose version)"

# --- 4. Free port is checked by compose; warn if something already listens ---
if command -v lsof &>/dev/null && lsof -ti "tcp:${HOST_PORT}" &>/dev/null; then
  echo "[WARN] Port ${HOST_PORT} already in use — the container may fail to bind."
  echo "       Stop the conflicting process or run: HOST_PORT=8001 ./docker-start.sh"
fi

# --- 5. Download model assets to host-mounted dirs (idempotent) ---
if [ -f "download_model.py" ]; then
  echo ""
  echo "Checking model assets (idempotent, skips what exists)..."
  if command -v python3 &>/dev/null; then
    python3 download_model.py || echo "[WARN] Asset download had issues — continuing; re-run ./docker-start.sh with network."
  else
    echo "[WARN] python3 not found on host — assets will download inside the container on first verify."
  fi
fi

# --- 6. Build + start ---
echo ""
echo "Building image if needed and starting container..."
$COMPOSE up --build -d

# --- 7. Wait for backend health (up to 90 s; first import is slow) ---
echo "Waiting for backend at ${HEALTH_URL} ..."
READY=0
for i in $(seq 1 90); do
  if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

echo ""
if [ "$READY" -eq 1 ]; then
  echo "[OK] Lab is healthy at http://localhost:${HOST_PORT}"
  echo "     Run './docker-stop.sh' to stop (files are kept)."
else
  echo "[WARN] Backend did not respond within 90 s."
  echo "       Check logs: docker compose logs --tail=100"
  echo "       Opening browser anyway — it may need a manual refresh."
fi

# --- 8. Open browser ---
if command -v open &>/dev/null; then
  open "http://localhost:${HOST_PORT}" || true
elif command -v xdg-open &>/dev/null; then
  xdg-open "http://localhost:${HOST_PORT}" >/dev/null 2>&1 &
fi

echo ""
echo "  Hub:     http://localhost:${HOST_PORT}"
echo "  Logs:    docker compose logs -f"
echo "  Verify:  docker compose exec lab python verify_docker.py"
echo ""
