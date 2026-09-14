#!/usr/bin/env bash
# setup.sh — Physical AI Workshop environment setup (Mac/Linux)
# Usage: bash setup.sh
# Note: This is a best-effort fallback; the workshop is validated on Windows 11 only.
# See README.md for ARM64 Linux/Mac notes.

set -e

# ---------------------------------------------------------------------------
# Step 1: Verify conda is available and source conda shell integration
# ---------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html and restart your terminal."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
. "${CONDA_BASE}/etc/profile.d/conda.sh" || {
    echo "Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html and restart your terminal."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 2: Create the conda environment (120-second timeout)
# ---------------------------------------------------------------------------
echo "Creating conda environment 'physical-ai' with Python 3.12..."
timeout 120 conda create -n physical-ai python=3.12 -y --no-default-packages || {
    echo "Conda solver is slow. Re-run with: conda install -n base conda-libmamba-solver && conda config --set solver libmamba"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 3: Activate the environment
# ---------------------------------------------------------------------------
echo "Activating environment..."
conda activate physical-ai || {
    echo "Environment activation failed. Run 'conda init bash' (or your shell name) and restart your terminal, then re-run setup.sh."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 4: Install PyTorch CPU-only (must come before requirements.txt)
# ---------------------------------------------------------------------------
echo "Installing PyTorch (CPU-only)..."
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu || {
    echo "Pip install failed. Check your internet connection and re-run setup.sh"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 5: Install remaining dependencies
# ---------------------------------------------------------------------------
echo "Installing workshop dependencies..."
pip install -r requirements.txt || {
    echo "Pip install failed. Check your internet connection and re-run setup.sh"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 6: Register the Jupyter kernel
# ---------------------------------------------------------------------------
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name physical-ai --display-name "Physical AI Workshop"

# ---------------------------------------------------------------------------
# Step 7: Download models and demo assets (idempotent — safe to re-run)
#   - assets/hand_landmarker.task   (~8 MB, MediaPipe)
#   - models/sac-HalfCheetah-v5.zip (~3 MB, HuggingFace)
#   - assets/fallback_hand_demo.mp4  (copied from frontend/assets/)
# ---------------------------------------------------------------------------
echo "Downloading models and demo assets..."
python download_model.py || echo "Warning: one or more assets failed to download — re-run setup.sh when connectivity is restored."

# ---------------------------------------------------------------------------
# Step 8: Run the verification script (non-fatal)
# ---------------------------------------------------------------------------
echo "Running verification..."
python verify_install.py || echo "Setup completed with warnings — see output above"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Done! Open VS Code and select the 'Physical AI Workshop' kernel."
echo "Then start the workshop hub with: python api/server.py"
echo "Open http://localhost:8000 in your browser."
