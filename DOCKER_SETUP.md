# Physical AI Lab — Docker Setup (Recommended)

Run the whole lab in one container. You do **not** need to install Python, PyTorch, MediaPipe, MuJoCo, or any AI library on your laptop. You only need **Docker + a browser**.

```
Your laptop
|-- Docker Desktop / Docker Engine
|    |-- physical-ai-lab container (Python 3.12, PyTorch CPU, MediaPipe,
|         MuJoCo headless, Gymnasium, SB3, OpenCV, FastAPI)
|-- Chrome / Edge browser -> http://localhost:8000
```

The browser runs natively. The Python backend runs in Docker. The repository folder is mounted into the container, so your notebooks, edits, downloaded models and generated files stay on your laptop.

Native setup (`setup.bat` / `setup.sh`) still works and is documented at the bottom for special hardware needs.

---

## A. Prerequisites

| Requirement | Details |
|---|---|
| OS | Windows 10/11, macOS (Intel or Apple Silicon), Ubuntu/Linux |
| Docker | Docker Desktop 4.x+ (includes Compose v2). Check with `docker compose version` |
| Browser | Chrome or Edge (Firefox is not supported — MediaPipe needs SharedArrayBuffer) |
| Network | Internet for the first pull (~1–1.5 GB image + ~11 MB models) |
| Hardware | No NVIDIA GPU required. Webcam recommended but optional (demo video fallback exists) |

Minimum Compose: v2.20+. Check:

```bash
docker --version
docker compose version
```

If `docker compose` fails but `docker-compose` works, update Docker Desktop — this lab uses `docker compose` (v2).

---

## B. First-time installation

### 1. Clone the repository

```bash
git clone https://github.com/learnermaker/physical-ai-lab.git
cd physical-ai-lab
```

### 2. Start Docker Desktop

Make sure the Docker whale icon says “Running”. On Linux make sure `docker info` succeeds.

### 3. Launch the lab

**Windows** — double-click:

```
docker-start.bat
```

**macOS / Linux**:

```bash
chmod +x docker-start.sh docker-stop.sh
./docker-start.sh
```

Or manually (all platforms):

```bash
docker compose up --build -d
```

The launcher will:

1. Check Docker is installed and running.
2. Check Compose v2 is available.
3. Download model assets if missing (idempotent — skips what exists).
4. Build the image if needed and start the container.
5. Wait until `http://localhost:8000/api/config` returns 200 (up to 90 s on first run).
6. Open `http://localhost:8000` in your browser.

`HOST_PORT` override if port 8000 is taken:

```bash
# macOS/Linux
HOST_PORT=8001 ./docker-start.sh
# Windows (PowerShell)
$env:HOST_PORT="8001"; .\docker-start.bat
# Manual
HOST_PORT=8001 docker compose up --build -d
```

### 4. Open the lab

Go to **http://localhost:8000** (or `http://localhost:8001` if you overrode the port) in Chrome or Edge.

### 5. Initialise model assets (usually automatic)

`docker-start` already runs `download_model.py` on the host. If you skipped it, run inside the container:

```bash
docker compose exec lab python download_model.py
docker compose exec lab python verify_docker.py
```

What gets downloaded (all idempotent):

| Asset | Size | Location (host-mounted, persisted) |
|---|---|---|
| `hand_landmarker.task` (MediaPipe) | ~8 MB | `assets/` |
| `sac-HalfCheetah-v5.zip` (pre-trained SAC) | ~3 MB | `models/` |
| `fallback_hand_demo.mp4` (copied from `frontend/assets/`) | small | `assets/` |

Existing files are not re-downloaded. Failures print an actionable error — re-run when network is back. Large models are never baked into the image and never committed to Git.

### 6. (Optional) Gemini key for Module 5

Module 5 works without a key (cached fallback responses). For live inference:

1. Get a free key at https://aistudio.google.com/app/apikey
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY=...`, **or** paste the key in the hub’s browser UI (stored in `sessionStorage` only).

```bash
cp .env.example .env
# edit .env, then:
docker compose restart lab
```

Never commit `.env`. The image never contains secrets.

---

## C. Daily usage

| Task | Windows | macOS/Linux |
|---|---|---|
| Start | `docker-start.bat` | `./docker-start.sh` |
| Stop (keeps files) | `docker-stop.bat` | `./docker-stop.sh` |
| Restart | `docker compose restart lab` | same |
| Logs (follow) | `docker compose logs -f` | same |
| Last 100 lines | `docker compose logs --tail=100` | same |
| Verify inside container | `docker compose exec lab python verify_docker.py` | same |
| Open a shell | `docker compose exec lab bash` | same |
| Rebuild after Dockerfile change | `docker compose up --build -d` | same |

Stopping never deletes participant work. The repo bind-mount (`.:/workspace`) means notebooks, `models/`, `assets/` and generated files live on your host. Named volumes only hold caches (`pip`, `huggingface`, `matplotlib`).

To fully reset (deletes caches but **not** your repo files):

```bash
docker compose down -v
```

---

## D. Camera access — browser vs container (important)

There are **two different** camera paths. Only one is used in Docker workshops:

**A. Browser-based (recommended Docker workflow)**
- Modules use `navigator.mediaDevices.getUserMedia()` in `frontend/js/module-1.js`.
- The browser asks for camera permission natively — Docker is not involved.
- Works with zero container webcam configuration.
- If no camera: click **Demo video** in Module 1 (uses `/assets/fallback_hand_demo.mp4` served by FastAPI).

**B. Direct Python `cv2.VideoCapture(0)` (native scripts only)**
- Scripts like `modules/01_perception/*.py` call `utils/camera.py:open_camera()`.
- Inside Docker there is no `/dev/video0` by default, so `open_camera()` transparently falls back to the looping demo video.
- That fallback is correct behaviour in a container — it is **not** reported as a failed browser setup.
- Platform passthrough (`--device=/dev/video0` on Linux) is intentionally not enabled by default for cross-platform reliability.

If the browser says “permission denied”: allow camera in browser settings and click Start again. On Linux check `Settings → Privacy → Camera`; on Windows check `Settings → Privacy → Camera`.

`verify_docker.py` deliberately does **not** test the camera — test it from the host browser via Module 1.

---

## E. Model downloads

- Handled by `download_model.py` (same script as native setup).
- Idempotent: existing files are skipped.
- Stored in host-mounted `assets/` and `models/` — survives `docker compose down` and rebuilds.
- Do not commit `*.task`, `*.mp4` (in `assets/`), `*.zip`, `*.pt`, `*.onnx` — they are gitignored.
- Do not bake large models into the image — the Dockerfile only keeps `.gitkeep` placeholders.

---

## F. Jupyter and notebooks

Jupyter is installed (`ipykernel`) but **not published** by default (security: no extra ports, no tokens in a shared workshop).

To use notebooks, either:

1. Open `.ipynb` files in VS Code with the container’s Python (via the Docker extension “Attach”), or
2. Temporarily publish Jupyter yourself:

```bash
docker compose exec lab python -m ipykernel install --user --name physical-ai --display-name "Physical AI Lab"
docker compose exec lab python -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Only do this on a trusted local network. Stop it when done. The default `compose.yaml` intentionally exposes only port 8000 on localhost.

---

## G. Troubleshooting

| Problem | Fix |
|---|---|
| `Docker not found` | Install Docker Desktop: https://www.docker.com/products/docker-desktop/ |
| `Docker daemon not running` | Start Docker Desktop (whale icon → Running). Linux: `sudo systemctl start docker` |
| `docker compose` not found | Update Docker Desktop (need Compose v2). Do not use the old `docker-compose` Python package |
| Port 8000 in use | `HOST_PORT=8001 ./docker-start.sh` (or Task Manager → kill PID; `lsof -ti tcp:8000 \| xargs kill -9` on mac) |
| Image build fails on `torch` | Retry with network; the Dockerfile falls back from `torch==2.14.0+cpu` to latest CPU automatically. Check disk space (~2 GB free needed) |
| Model download fails | Re-run `docker compose exec lab python download_model.py` with network. Manual URLs are in that file |
| Browser: camera permission denied | Allow camera in browser site settings, reload, click Start again |
| Browser: no camera found | Use **Demo video** button (Module 1) — no webcam required |
| Simulation shows “software render” | Expected headless mode (`MUJOCO_GL=osmesa`). Physics, observations and RL all run correctly; only the interactive desktop window is replaced by a PIL status frame |
| `MUJOCO_GL` / EGL errors in logs | Default is `osmesa` (software). Do not switch to `egl`/`glfw` unless you have a GPU + NVIDIA Container Toolkit. Native install supports interactive rendering |
| Gemini returns cache/fallback | No key set or key invalid. Set `.env` or browser key (see §B.6). Check quota at https://aistudio.google.com/ |
| Apple Silicon “exec format error” or slow | See §H. `mediapipe` 0.10.21 has no Linux ARM64 wheel — use AMD64 emulation (Docker Desktop → Rosetta) or native macOS setup |
| `verify_docker.py` backend WARN | Server not running — `docker compose up -d`, wait 30 s, re-run |
| Files created as root / permission denied | Image runs as UID 1000 (`lab`). If you ran an old root container, `sudo chown -R $(id -u):$(id -g) .` on the host once |

Logs are your friend:

```bash
docker compose logs --tail=100
docker compose exec lab python verify_docker.py
docker compose exec lab python -c "import torch, mediapipe, mujoco; print(torch.__version__)"
```

---

## H. Platform notes / architecture support

| Platform | Status | Notes |
|---|---|---|
| Windows 10/11 x86-64 | Supported (primary) | Use `docker-start.bat`. WSL2 backend recommended |
| Linux x86-64 | Supported (best-tested container arch) | All wheels native, `osmesa` headless works |
| macOS Intel | Supported | Use `./docker-start.sh` |
| macOS Apple Silicon (ARM64) | Works via AMD64 emulation; native ARM has one known gap | **MediaPipe 0.10.21** (required for `mp.solutions` in `02_hand_tracking.py`) publishes **Linux x86-64 wheels only, no Linux ARM64 wheel**. Docker Desktop on Apple Silicon can run the AMD64 image via Rosetta 2 emulation (enable in Docker Desktop → Settings → General → Rosetta). Expect slower startup. Browser MediaPipe (Module 1, WASM) is unaffected. Do not claim native ARM support until `pip install mediapipe==0.10.21` succeeds on `linux/arm64` |
| ARM64 Windows (Snapdragon) | Not validated | MuJoCo has no ARM64 Windows wheel either (see `instructor/COMMON_ISSUES.md`). Use WSL2 + emulation or native x86-64 laptop |
| NVIDIA GPU | Not required, not used by default | CPU-only PyTorch. GPU is a future optional enhancement — do not install CUDA Toolkit for this workshop |

The Dockerfile is CPU-first by design: no CUDA, no NVIDIA Container Toolkit, no privileged mode, no `--gpus` flag.

---

## I. Native installation (still available)

Docker is the recommended route, but native setup is preserved for interactive MuJoCo windows, direct `cv2.VideoCapture(0)` access, and GPU experiments:

- Windows: `setup.bat`
- macOS/Linux: `bash setup.sh`
- Health check: `python verify_install.py`
- Manual start: `python start_hub.py` or `python api/server.py`

Native and Docker share the same `requirements.txt`, `download_model.py`, `api/server.py` and `frontend/`. No Docker change breaks native setup — CI checks this on every push (see `.github/workflows/docker-build.yml`).

---

## Quick reference

```bash
git clone https://github.com/learnermaker/physical-ai-lab.git
cd physical-ai-lab
# Windows: double-click docker-start.bat
./docker-start.sh            # mac/Linux
# open http://localhost:8000 in Chrome/Edge
docker compose logs -f
docker compose exec lab python verify_docker.py
./docker-stop.sh             # or docker-stop.bat (files are kept)
```
