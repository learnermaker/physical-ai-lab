# syntax=docker/dockerfile:1

# CPU-first workshop image. Browser camera access remains on the host.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MUJOCO_GL=osmesa

WORKDIR /workspace

# Runtime libraries for OpenCV, MuJoCo's headless software rendering,
# and common scientific Python wheels. No desktop/X11 stack is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libosmesa6 \
        libglfw3 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# Explicit CPU PyTorch index prevents accidentally pulling a CUDA build.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-build-isolation -r /tmp/requirements.txt

COPY . /workspace

# Model/assets live in the mounted workspace and survive container recreation.
RUN mkdir -p /workspace/assets /workspace/models

EXPOSE 8000 8888

# Run as a non-root user, while keeping mounted workspace writable in
# the default Docker Desktop mapping. The repo is mounted at runtime.
RUN useradd --create-home --uid 10001 labuser
USER labuser

CMD ["python", "api/server.py"]
