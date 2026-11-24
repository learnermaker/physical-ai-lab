#!/usr/bin/env bash
# Physical AI Lab — Docker stop (macOS / Linux).
# Stops the container WITHOUT deleting participant files, models or caches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Physical AI Lab container (files are kept)..."
docker compose down
echo "[OK] Stopped. Your files, models and caches are preserved."
echo "     Restart with: ./docker-start.sh"
