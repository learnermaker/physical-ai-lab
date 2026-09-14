#!/usr/bin/env bash
# =============================================================================
# Physical AI Lab — Hub Launcher (Mac/Linux)
# Created by Jim Seelan
#
# Starts the API server and opens the browser.
# Run this on any subsequent session — much faster than re-running setup.sh.
#
# Usage:
#   bash start_hub.sh
#
# Requirements: setup.sh must have been run at least once (.venv must exist).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PORT=8000
HEALTH_URL="http://localhost:${PORT}/api/config"

echo ""
echo "  Physical AI Lab — starting hub..."
echo ""

# Sanity check: venv must exist
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "  [FAIL] .venv not found. Run setup.sh first:"
    echo "         bash setup.sh"
    exit 1
fi

# Activate the virtual environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Free port 8000 if something is already listening
if command -v lsof &>/dev/null; then
    PIDS=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "  Freeing port ${PORT}..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
fi

# Start the server in the background
nohup python api/server.py > /tmp/physical-ai-server.log 2>&1 &
SERVER_PID=$!
echo "  Server started (PID $SERVER_PID). Log: /tmp/physical-ai-server.log"

# Poll until healthy (up to 20 seconds)
echo "  Waiting for server to become healthy..."
HEALTHY=0
for i in $(seq 1 40); do
    if curl -s --max-time 1 "$HEALTH_URL" &>/dev/null; then
        HEALTHY=1
        break
    fi
    sleep 0.5
done

if [ "$HEALTHY" -eq 1 ]; then
    echo "  [OK] Server healthy at http://localhost:${PORT}"
else
    echo "  [WARN] Server did not respond in 20 s — check /tmp/physical-ai-server.log"
    echo "         Opening browser anyway; it may need a manual refresh."
fi

# Open browser
if command -v open &>/dev/null; then
    open "http://localhost:${PORT}"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:${PORT}" &>/dev/null &
fi

echo ""
echo "  Hub:     http://localhost:${PORT}"
echo "  Browser: opening now"
echo ""
echo "  Use Chrome or Edge — Firefox is not supported (MediaPipe)."
echo "  To stop the server:  kill $SERVER_PID"
echo ""
