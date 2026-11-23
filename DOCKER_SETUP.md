# Docker Setup — Physical AI Lab

Docker is the recommended quick-start path for participants who want a consistent Python environment without installing Python, PyTorch, MuJoCo, MediaPipe or workshop packages on the host.

## What Docker does (and does not do)

The container runs the lab's Python environment, FastAPI hub, reinforcement-learning workloads and simulation code. The participant's normal Chrome/Edge browser stays on the host and opens `http://localhost:8000`.

The browser requests webcam permission from the host operating system. The hub's camera path uses browser `getUserMedia`, not container-side `cv2.VideoCapture(0)`. Do not pass the webcam into the container for the hub workflow.

The current MediaPipe pin (`<0.10.22`) has a CPython 3.12 Linux x86-64 wheel but no Linux ARM64 wheel. Docker on ARM-based hosts therefore needs an x86-64 emulation path or a future MediaPipe Tasks API migration in the lab. The current setup is intended for x86-64 Linux containers; Apple Silicon may build/run under emulation, with a performance penalty, and is not yet validated. See the architecture note below.

The default image is CPU-only. No CUDA toolkit or GPU is required. MuJoCo is configured for headless software rendering. Physics, observations, actions and RL training work without a desktop display. This does not promise an interactive native MuJoCo desktop window inside Docker.

## Requirements

- Docker Desktop on Windows/macOS, or Docker Engine plus the Compose plugin on Linux.
- Docker Compose v2.24 or later (`docker compose`); optional `.env` loading in the supplied Compose file uses the `required: false` property added in Compose 2.24.
- Supported 64-bit host and internet access for the first image, package and model downloads.
- Chrome or Edge recommended for the workshop browser hub.

Install Docker: https://docs.docker.com/get-started/get-docker/

On Windows, Docker Desktop with the WSL 2 backend is recommended. Docker Desktop includes Docker Engine, CLI and Compose. Participants do not need a separate Linux distribution just to run this Compose workflow.

## First-time setup

1. Install and start Docker Desktop (or start Docker Engine on Linux).
2. Clone the repository:

   ```bash
   git clone https://github.com/learnermaker/physical-ai-lab.git
   cd physical-ai-lab
   ```

3. Optional: configure Gemini. Copy `.env.example` to `.env` and add `GEMINI_API_KEY`. The key stays on the host in `.env`; do not commit it.
4. Start the lab:

   **Windows:** double-click `docker-start.bat`, or run `docker compose up --build -d` in PowerShell.

   **macOS/Linux:** run `./docker-start.sh`, or run `docker compose up --build -d`.

5. Open `http://localhost:8000` in the host browser. Start a perception module and allow camera access when prompted.

The first build downloads the Python base image, CPU PyTorch and workshop dependencies. It can take several minutes depending on the network. The repository is bind-mounted into the container, so model downloads and participant edits remain on the host.

## Download workshop model assets

The repository excludes generated model binaries. Once the container is running, execute:

```bash
docker compose exec lab python download_model.py
```

This downloads the MediaPipe hand-landmarker task and the pretrained SAC policy into the host-mounted `assets/` and `models/` directories. The fallback demo clip is copied from `frontend/assets/` into `assets/`.

## Daily use

Start an existing build:

```bash
docker compose up -d
```

Follow logs:

```bash
docker compose logs -f lab
```

Stop without deleting participant files or downloaded assets:

```bash
docker compose down
```

Avoid `docker compose down -v` unless you intentionally want to delete the named dependency/cache volume. Files mounted from the repository remain on the host either way.

## Editing exercises and notebooks

The repository is bind-mounted at `/workspace`. Changes to Python exercises, notebooks and source files are written directly to the host checkout and survive container recreation.

For an optional Jupyter session, run:

```bash
docker compose exec lab jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=physical-ai
```

Jupyter is not published by default. To use it from the host, add `- "8888:8888"` under the service's `ports` in `compose.yaml`, recreate the container, then open `http://localhost:8888/lab?token=physical-ai`. This fixed token is for a local workshop machine only; do not expose Jupyter to an untrusted network. For many participants, prefer VS Code Dev Containers or native VS Code with a registered host kernel after validating that workflow.

## Camera and direct Python scripts

There are two separate access paths:

- **Hub modules:** browser camera permission is handled by Chrome/Edge on the host. This is the recommended workshop path and requires no container device mapping.
- **Standalone scripts using `cv2.VideoCapture(0)`:** these attempt to open a camera from Python inside the container. A standard Linux container does not automatically see the host webcam. On native Linux, a specific `/dev/videoN` device can be passed with `--device=/dev/videoN`; on Docker Desktop for Windows/macOS, USB/IP or another device integration may be required. This default Compose setup deliberately does not configure those platform-specific mappings. Use the native Python setup for standalone camera scripts unless the facilitator has tested the specific hardware path.

## Graphics and GPU

The default image uses CPU PyTorch and `MUJOCO_GL=osmesa` for headless software rendering. No GPU is needed for the core workshop. Some standalone exercises that explicitly open an interactive MuJoCo window may not display a window from this container; use the host-native setup for those workflows.

**Architecture support:** the pinned MediaPipe release `0.10.21` publishes Linux wheels for x86-64, not Linux ARM64. The current image is therefore an x86-64-compatible build. On Apple Silicon, Docker Desktop can emulate an AMD64 container, but this may be slower and less reliable than native ARM64. The current Docker setup has not been validated on ARM Linux or Apple Silicon. We should not claim native multi-architecture support until the MediaPipe dependency/API is migrated and both architectures are built and tested.

An NVIDIA GPU option is not enabled by default. Docker Desktop GPU passthrough on Windows requires an NVIDIA GPU, compatible current drivers, an up-to-date WSL 2 kernel and the WSL 2 backend. Compose GPU reservations additionally require compatible Docker Engine/NVIDIA Container Toolkit configuration. CPU-only must remain the reliable baseline for the workshop.

## Troubleshooting

| Symptom | Action |
|---|---|
| `docker` command not found | Install Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux), then restart the terminal. |
| Docker daemon unavailable | Start Docker Desktop or the Linux Docker service. |
| First build fails downloading packages | Check network/proxy access and retry `docker compose build --no-cache`. |
| Port 8000 already allocated | Set `PHYSICAL_AI_PORT=8001` in the shell before starting and open `http://localhost:8001` (PowerShell: `$env:PHYSICAL_AI_PORT=8001`; Command Prompt: `set PHYSICAL_AI_PORT=8001`; macOS/Linux: `export PHYSICAL_AI_PORT=8001`). |
| Hub does not become ready | Run `docker compose logs --tail=200 lab`. |
| No camera prompt appears | Open the hub on the host browser, start a perception module, and check OS camera privacy settings plus browser permission for `localhost`. |
| Camera works in hub but not a Python script | Expected: browser camera and container camera are separate access paths. Use host-native Python or tested platform-specific device passthrough. |
| MuJoCo viewer window fails | The default container is headless. Use the hub's software-render fallback or host-native setup for native interactive viewers. |
| Gemini is unavailable | The API key is optional. Check `.env` contains `GEMINI_API_KEY=...`, then run `docker compose up -d`. Cached responses are used when no key is configured or a request fails. |

## Native setup remains available

Docker is the default quick-start route, not a replacement for the native scripts. Participants or facilitators who need direct OpenCV webcam access, a desktop MuJoCo viewer, unusual host hardware or platform-specific debugging can continue using `setup.bat` (Windows) or `setup.sh` (macOS/Linux).
