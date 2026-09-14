#!/usr/bin/env bash
# =============================================================================
# Physical AI Lab — Mac/Linux Setup Script
# Created by Jim Seelan
#
# Mirrors setup.bat: creates a .venv, installs all dependencies, downloads
# models, verifies, and launches the hub.
# Safe to re-run — detects what's already done and skips those steps.
#
# Usage:
#   bash setup.sh
#
# Requirements:
#   - Python 3.12 installed (see below if missing)
#   - Internet connection (~700 MB of downloads)
#   - Chrome or Edge browser (Firefox is not supported — MediaPipe needs SharedArrayBuffer)
#
# Python 3.12 quick install:
#   macOS:          brew install python@3.12
#   Ubuntu/Debian:  sudo add-apt-repository ppa:deadsnakes/ppa && \
#                   sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

echo "============================================================"
echo "  Physical AI Lab - Setup"
echo "  Created by Jim Seelan"
echo "============================================================"
echo ""

# =============================================================================
# Free port 8000 if anything is already listening on it
# =============================================================================
if command -v lsof &>/dev/null; then
    PIDS=$(lsof -ti tcp:8000 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "Freeing port 8000..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
fi

# =============================================================================
# STEP 1 — Locate Python 3.12
# =============================================================================
echo "[1/9] Locating Python 3.12..."

PYTHON_EXE=""

# Try explicit python3.12 first, then python3, then python
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oP '3\.12\.\d+' || true)
        if [ -n "$ver" ]; then
            PYTHON_EXE="$candidate"
            echo "[OK] Found: $($PYTHON_EXE --version) at $(command -v $PYTHON_EXE)"
            break
        fi
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    echo ""
    echo "[FAIL] Python 3.12 not found."
    echo ""
    echo "  macOS:  brew install python@3.12"
    echo "  Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "          sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev"
    echo ""
    exit 1
fi

# =============================================================================
# STEP 2 — Create virtual environment
# =============================================================================
echo ""
echo "[2/9] Setting up virtual environment..."

if [ -f "$VENV_DIR/bin/python" ]; then
    # Verify the venv is healthy — its base Python may have been deleted
    if "$VENV_DIR/bin/python" --version &>/dev/null; then
        echo "[OK] Virtual environment healthy."
    else
        echo "[!] Broken .venv detected — recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON_EXE" -m venv "$VENV_DIR"
        echo "[OK] Virtual environment created."
    fi
else
    "$PYTHON_EXE" -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created."
fi

# =============================================================================
# STEP 3 — Activate
# =============================================================================
echo ""
echo "[3/9] Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "[OK] Active: $VIRTUAL_ENV"
echo ""

# Upgrade pip (non-fatal)
python -m pip install --upgrade pip --quiet 2>/dev/null || true

# =============================================================================
# STEP 4 — PyTorch CPU-only
# =============================================================================
echo "[4/9] Installing PyTorch CPU-only (~200 MB, may take a few minutes)..."
# Try pinned version first; fall back to latest if the wheel doesn't exist yet.
if ! pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null; then
    echo "[!] torch==2.14.0+cpu not found — trying latest stable CPU build..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
echo "[OK] PyTorch installed."
echo ""

# =============================================================================
# STEP 5 — Lab dependencies
# =============================================================================
echo "[5/9] Installing lab dependencies..."
pip install -r requirements.txt
echo "[OK] Dependencies installed."
echo ""

# =============================================================================
# STEP 6 — Jupyter kernel
# =============================================================================
echo "[6/9] Registering Jupyter kernel..."
if python -m ipykernel install --user --name physical-ai --display-name "Physical AI Lab"; then
    echo "[OK] Kernel 'Physical AI Lab' registered."
else
    echo "[WARN] Kernel registration failed — select it manually in VS Code."
fi
echo ""

# =============================================================================
# STEP 7 — Download models and demo assets
#   - assets/hand_landmarker.task    (~8 MB, MediaPipe)
#   - models/sac-HalfCheetah-v5.zip  (~3 MB, HuggingFace farama-minari)
#   - assets/fallback_hand_demo.mp4   (copied from frontend/assets/)
# =============================================================================
echo "[7/9] Downloading models and demo assets..."
echo "       (hand_landmarker.task, sac-HalfCheetah-v5.zip, fallback demo video)"
python download_model.py || echo "[WARN] One or more assets failed to download — re-run setup.sh when connectivity is restored."
echo ""

# =============================================================================
# STEP 8 — Verify installation
# =============================================================================
echo "[8/9] Verifying installation..."
echo ""
python verify_install.py && echo "[OK] All checks passed." || {
    echo ""
    echo "Completed with warnings — see above."
    echo "[WARN] = fallback active   [FAIL] = needs attention"
}
echo ""

# =============================================================================
# STEP 9 — Launch hub and open browser
# =============================================================================
echo "[9/9] Launching Physical AI Lab hub..."
echo ""

# Start server in background; save PID so participants can stop it
nohup python api/server.py > /tmp/physical-ai-server.log 2>&1 &
SERVER_PID=$!
echo "Server started (PID $SERVER_PID). Log: /tmp/physical-ai-server.log"

# Wait for the server to become healthy (up to 20 seconds)
echo "Waiting for server to become healthy..."
HEALTH_URL="http://localhost:8000/api/config"
HEALTHY=0
for i in $(seq 1 40); do
    if curl -s --max-time 1 "$HEALTH_URL" &>/dev/null; then
        HEALTHY=1
        break
    fi
    sleep 0.5
done

if [ "$HEALTHY" -eq 1 ]; then
    echo "[OK] Server healthy at http://localhost:8000"
else
    echo "[WARN] Server did not respond in 20 s — check /tmp/physical-ai-server.log"
    echo "       Opening browser anyway; it may need a manual refresh."
fi

# Open browser (macOS: open, Linux: xdg-open)
if command -v open &>/dev/null; then
    open "http://localhost:8000"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8000" &>/dev/null &
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Hub:     http://localhost:8000"
echo "  Browser: opening now"
echo ""
echo "  Use Chrome or Edge — Firefox is not supported (MediaPipe)."
echo ""
echo "  Next time: source .venv/bin/activate && python api/server.py"
echo "  Or just:   bash setup.sh   (it skips steps already done)"
echo "============================================================"
echo ""
