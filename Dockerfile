# Physical AI Lab — CPU-first workshop container.
# Base: official Python 3.12 slim (Debian bookworm). Single service, no GPU required.
# Browser runs on the host; backend runs here and is reached via http://localhost:8000.

FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# Environment — headless-safe defaults for MuJoCo / matplotlib / OpenCV.
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

# ---------------------------------------------------------------------------
# System libraries.
#   libgl1, libglib2.0-0, libsm6, libxext6, libxrender1 — OpenCV runtime.
#   libosmesa6, libegl1, libglvnd0, libglu1-mesa        — MuJoCo headless (osmesa).
#   ffmpeg                                             — imageio[ffmpeg] video encoding.
#   curl                                               — manual health probing / debugging.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libosmesa6 \
        libegl1 \
        libglvnd0 \
        libglu1-mesa \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Non-root runtime user. UID 1000 matches the typical host user so bind-mounted
# files created inside the container remain editable on the host.
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 -s /bin/bash lab \
    && mkdir -p /workspace /home/lab/.cache/pip /home/lab/.cache/huggingface /home/lab/.cache/matplotlib \
    && chown -R lab:lab /workspace /home/lab/.cache

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Python dependencies — copy requirements first for layer caching.
# Torch CPU is installed first via the PyTorch CPU index; the pinned version
# is tried first with a fallback to latest stable CPU if the pin disappears.
# requirements.txt intentionally does NOT list torch (see its header comment).
# requirements.txt pins mediapipe==0.10.21, which retains both the Tasks API
# and mp.solutions used by modules/01_perception and 04_perception_to_action.
# ---------------------------------------------------------------------------
COPY --chown=lab:lab requirements.txt ./

# torch 2.14.0+cpu verified on the CPU index for cp312: linux x86_64, linux
# aarch64, win_amd64, win_arm64. The fallback below is for future-proofing only
# (e.g. if the pin is yanked); it prints a loud warning so a silent environment
# change can never go unnoticed in build logs.
RUN pip install --upgrade pip \
    && (pip install "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
        || (echo "WARNING: pinned torch==2.14.0+cpu not found - falling back to latest stable CPU build" >&2 \
            && pip install torch --index-url https://download.pytorch.org/whl/cpu)) \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Application code. .dockerignore keeps secrets, venvs, git metadata and
# downloaded assets out of the image. Models are downloaded at runtime into
# the bind-mounted assets/ and models/ directories (see download_model.py).
# ---------------------------------------------------------------------------
COPY --chown=lab:lab api/ ./api/
COPY --chown=lab:lab frontend/ ./frontend/
COPY --chown=lab:lab modules/ ./modules/
COPY --chown=lab:lab utils/ ./utils/
COPY --chown=lab:lab tests/ ./tests/
COPY --chown=lab:lab download_model.py verify_docker.py start_hub.py ./
COPY --chown=lab:lab assets/.gitkeep assets/
COPY --chown=lab:lab models/.gitkeep models/

USER lab

EXPOSE 8000

# FastAPI health endpoint. Interval/start-period generous: first import of
# torch/mediapipe/mujoco can take 10-20 s on a cold laptop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/config', timeout=4)"

CMD ["python", "api/server.py"]
