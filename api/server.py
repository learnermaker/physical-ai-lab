"""
Experience Hub — FastAPI backend for the Physical AI Workshop.

Start with:
    python api/server.py

Serves the frontend at http://localhost:8000 and exposes API endpoints used
by the browser page for simulation streaming, RL training, and Gemini calls.
"""

import asyncio
import base64
import importlib.util as _importlib_util
import json
import os
import queue
import random
import socket
import sys
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Startup: ensure repo root is importable regardless of working directory.
# This lets utils.gym_utils, utils.camera, and modules.* resolve correctly.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

# Load .env from repo root; silently succeeds if file is absent.
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

# Cache key in memory — never exposed to clients.
_GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY") or None

# ---------------------------------------------------------------------------
# Module-level Gemini cache — loaded once at startup via the Module 5 helper.
# load_cache() calls sys.exit(1) if the file is absent, which would crash the
# server at import time.  We therefore catch that and fall back to an empty
# dict so the server still starts; the endpoints degrade gracefully.
# ---------------------------------------------------------------------------
_CACHE: dict = {}
try:
    _cache_init_path = _REPO_ROOT / "modules" / "05_foundation_models" / "__init__.py"
    _cache_spec = _importlib_util.spec_from_file_location("fm_init", _cache_init_path)
    _fm = _importlib_util.module_from_spec(_cache_spec)
    _cache_spec.loader.exec_module(_fm)
    load_cache = _fm.load_cache
    _CACHE = load_cache()
except SystemExit:
    print(
        "[server] WARNING: cached_responses.json not found or empty — "
        "Gemini endpoints will not have a cache fallback.",
        file=sys.stderr,
    )
except Exception as _cache_exc:
    print(f"[server] WARNING: Failed to load Gemini cache: {_cache_exc}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Pydantic request models for Gemini endpoints
# ---------------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402


class GeminiDescribeRequest(BaseModel):
    frame_b64: str
    api_key: str | None = None


class GeminiActionRequest(BaseModel):
    frame_b64: str
    api_key: str | None = None
    system_prompt: str | None = None


# Default structured action prompt — mirrors SYSTEM_PROMPT in 02_gemini_robot_brain.py.
# Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)
_ACTION_PROMPT = (
    "You are a robot controller. Looking at this image, suggest an action. "
    'Respond ONLY in JSON with no markdown: '
    '{"action": "LEFT|RIGHT|FORWARD|BACK|WAIT", "reason": "one sentence"}'
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Physical AI Workshop Hub")


# ---------------------------------------------------------------------------
# Middleware: COOP + COEP headers required for MediaPipe WASM SharedArrayBuffer
# in Chrome/Edge. Must be registered before any routes are added.
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        response: StarletteResponse = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

# In-memory job registry: {job_id: queue.Queue}
# The training thread puts progress dicts; the SSE endpoint consumes them.
_jobs: dict[str, queue.Queue] = {}


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Return whether a Gemini API key is configured server-side.

    The key value itself is never returned — only a boolean presence flag.
    Requirements: 19.3
    """
    return JSONResponse({"gemini_key_configured": _GEMINI_API_KEY is not None})


# --- Stubs for tasks 21.2 – 21.4 (return 501 until implemented) -----------


def _make_status_frame(obs: np.ndarray, action: np.ndarray, step: int) -> str:
    """Generate a synthetic 400x300 JPEG status frame using PIL.
    Used when OpenGL rendering is unavailable (no GPU driver).
    """
    import io  # noqa: PLC0415

    from PIL import Image, ImageDraw  # noqa: PLC0415

    W, H = 400, 300
    img = Image.new("RGB", (W, H), color=(18, 18, 28))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((16, 14), "Reacher-v5  (no-GPU mode)", fill=(180, 160, 120))
    draw.text((16, 32), f"Step {step}", fill=(100, 100, 120))

    # Obs labels
    labels = [
        "cos θ₀", "sin θ₀", "cos θ₁", "sin θ₁",
        "target x", "target y", "ω₀", "ω₁",
        "finger x", "finger y", "dist",
    ]
    draw.text((16, 60), "Observation:", fill=(140, 200, 140))
    for i, (lbl, val) in enumerate(zip(labels, obs)):
        x = 16 + (i % 2) * 196
        y = 78 + (i // 2) * 18
        draw.text((x, y), f"{lbl}: {val:+.3f}", fill=(200, 220, 200))

    # Action
    draw.text((16, 196), "Action:", fill=(200, 160, 100))
    draw.text((16, 214), f"torque 0: {action[0]:+.3f}", fill=(230, 190, 130))
    draw.text((16, 232), f"torque 1: {action[1]:+.3f}", fill=(230, 190, 130))

    # Render bars for action magnitude
    cx, cy = 200, 260
    for i, (val, colour) in enumerate(zip(action, [(100, 200, 100), (100, 160, 220)])):
        bar_w = int(abs(val) * 80)
        bar_x = cx if val >= 0 else cx - bar_w
        bar_y = cy + i * 18
        draw.rectangle([cx, bar_y, cx + 1, bar_y + 14], fill=(80, 80, 80))
        if bar_w > 0:
            draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 14], fill=colour)

    draw.text((16, 280), "OpenGL unavailable — state display mode", fill=(80, 80, 100))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/simulation/stream")
async def simulation_stream(action: str = Query("0.0,0.0")):
    """SSE: stream Reacher-v5 frames as base64 JPEG at ~20 fps.
    Falls back to a text-based status frame if OpenGL is unavailable.

    Requirements: 19.4
    """
    try:
        parts = [float(x) for x in action.split(",")]
        if len(parts) != 2:
            parts = [0.0, 0.0]
    except (ValueError, AttributeError):
        parts = [0.0, 0.0]
    action_array = np.array(parts, dtype=np.float32)

    async def event_generator():
        loop = asyncio.get_running_loop()
        env = None
        step_count = 0
        # Send a heartbeat SSE comment immediately so the browser does not
        # time out while MuJoCo initialises (can take 2–5 s on first load).
        yield ": heartbeat\n\n"
        try:
            from utils.gym_utils import make_env  # noqa: PLC0415

            # Create env in thread pool — gym.make() is synchronous and
            # performs MuJoCo initialisation (file I/O, memory allocation).
            # make_env's fallback chain handles the no-OpenGL case by falling
            # through human → rgb_array → None.  The reset() probe is now
            # inside make_env so we get back a ready-to-use environment.
            env = await loop.run_in_executor(None, make_env, "Reacher-v5", "rgb_array")

            # Detect actual render mode after make_env's fallback chain.
            actual_render_mode = getattr(env.unwrapped, "render_mode", None)
            _use_status_frame = actual_render_mode not in ("rgb_array", "human")

            # env.reset() was already called inside make_env (probe).
            # Call it again to get the initial observation for the first frame.
            obs, _ = await loop.run_in_executor(None, env.reset)

            while True:
                obs, reward, terminated, truncated, info = await loop.run_in_executor(
                    None, env.step, action_array
                )
                step_count += 1
                if terminated or truncated:
                    obs, _ = await loop.run_in_executor(None, env.reset)

                if not _use_status_frame:
                    # Try OpenGL rendering; fall back to status frame if it
                    # raises (e.g. gladLoadGL on machines without GPU drivers).
                    try:
                        rgb_frame = await loop.run_in_executor(None, env.render)
                        if rgb_frame is not None:
                            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                            success, jpg_buf = cv2.imencode(
                                ".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
                            )
                            if success:
                                b64 = base64.b64encode(jpg_buf.tobytes()).decode("ascii")
                                yield f"data: {b64}\n\n"
                                await asyncio.sleep(0.05)
                                continue
                        # render() returned None — switch to status frames.
                        _use_status_frame = True
                    except Exception as render_exc:
                        print(
                            f"[simulation_stream] render() failed: {render_exc} — "
                            "switching to status frame mode",
                            file=sys.stderr,
                        )
                        _use_status_frame = True

                # Status-frame path: PIL-generated JPEG with obs/action text.
                b64 = _make_status_frame(obs, action_array, step_count)
                yield f"data: {b64}\n\n"

                # Target ~20 fps.
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            # Client disconnected — exit generator cleanly.
            pass
        except Exception as exc:
            # Catch-all: send one final error frame instead of silently closing
            # the SSE stream (which would cause the browser to see OPEN→ERROR).
            print(f"[simulation_stream] fatal error: {exc}", file=sys.stderr)
            try:
                import io  # noqa: PLC0415
                from PIL import Image, ImageDraw  # noqa: PLC0415
                W, H = 400, 300
                img = Image.new("RGB", (W, H), color=(80, 20, 20))
                draw = ImageDraw.Draw(img)
                draw.text((16, 14), "Simulation error", fill=(255, 200, 200))
                draw.text((16, 40), str(exc)[:60], fill=(220, 180, 180))
                draw.text((16, 60), "Restart simulation to retry.", fill=(180, 140, 140))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                yield f"data: {b64}\n\n"
            except Exception:
                pass  # If even the error frame fails, close silently.
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/training/start")
async def training_start() -> JSONResponse:
    """Launch PPO CartPole training in a background thread; return a job_id.

    The background thread pushes progress events to a queue.Queue stored in
    _jobs[job_id].  The caller uses GET /api/training/progress/{job_id} to
    consume those events via SSE.
    Requirements: 19.5
    """
    job_id = str(uuid.uuid4())
    job_queue: queue.Queue = queue.Queue()
    _jobs[job_id] = job_queue

    def _run_training(jq: queue.Queue) -> None:
        import gymnasium as gym  # noqa: PLC0415
        from utils.gym_utils import StreamingCallback  # noqa: PLC0415
        from stable_baselines3 import PPO  # noqa: PLC0415

        # Use gym.make directly with render_mode=None for training — no
        # frames are needed and bypassing make_env avoids the fallback
        # chain's GLFW/OpenGL probe entirely on headless machines.
        env = gym.make("CartPole-v1", render_mode=None)
        model = PPO("MlpPolicy", env, verbose=0)
        callback = StreamingCallback(jq)
        try:
            model.learn(total_timesteps=50_000, callback=callback)
        except Exception as train_exc:
            print(f"[training] error: {train_exc}", file=sys.stderr)
            jq.put({"done": True, "error": str(train_exc)})
        finally:
            env.close()

    t = threading.Thread(target=_run_training, args=(job_queue,), daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/training/progress/{job_id}")
async def training_progress(job_id: str):
    """SSE: stream training metrics for a running job.

    Yields JSON-encoded progress events as SSE data frames.  A heartbeat
    event is sent every 30 seconds of queue silence so the connection stays
    alive through long gaps between reward updates.
    Requirements: 19.6
    """
    if job_id not in _jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job_queue = _jobs[job_id]

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(
                    None, lambda: job_queue.get(timeout=30)
                )
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("done"):
                    break
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/gemini/describe")
async def gemini_describe(req: GeminiDescribeRequest) -> JSONResponse:
    """Scene description via Gemini.

    Accepts {"frame_b64": str, "api_key": str | None}.
    Key priority: request body → server-side .env → None (cache fallback).
    Returns {"text": str}.
    Requirements: 19.7
    """
    key = req.api_key or _GEMINI_API_KEY

    # Gather string-valued cache entries for a sensible describe fallback.
    _describe_fallbacks = [v for v in _CACHE.values() if isinstance(v, str)]
    if not _describe_fallbacks:
        _describe_fallbacks = ["A scene captured from the workshop camera."]

    if not key:
        return JSONResponse({"text": random.choice(_describe_fallbacks)})

    try:
        jpg_bytes = base64.b64decode(req.frame_b64)

        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=key)
        image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=["Describe this scene in one sentence.", image_part],
        )
        return JSONResponse({"text": response.text})
    except Exception as exc:
        print(f"[Gemini describe error] {exc}", file=sys.stderr)
        return JSONResponse({"text": random.choice(_describe_fallbacks)})


@app.post("/api/gemini/action")
async def gemini_action(req: GeminiActionRequest) -> JSONResponse:
    """Robot action suggestion via Gemini.

    Accepts {"frame_b64": str, "api_key": str | None, "system_prompt": str | None}.
    Key priority: request body → server-side .env → None (cache fallback).
    Strips markdown fences before JSON parse.
    Returns a parsed action dict or a cached fallback entry.
    Requirements: 19.8
    """
    key = req.api_key or _GEMINI_API_KEY
    system_prompt = req.system_prompt or _ACTION_PROMPT

    action_entries = [v for v in _CACHE.values() if isinstance(v, dict) and "action" in v]
    if not action_entries:
        action_entries = [{"action": "WAIT", "reason": "No action entries in cache."}]

    if not key:
        return JSONResponse(random.choice(action_entries))

    try:
        jpg_bytes = base64.b64decode(req.frame_b64)

        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=key)
        image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[system_prompt, image_part],
        )

        # Strip markdown fences if the model wrapped the JSON (Design § 4.3).
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        if "action" not in result:
            raise ValueError(f"Response missing 'action' key: {result}")

        return JSONResponse(result)
    except Exception as exc:
        print(f"[Gemini action error] {exc}", file=sys.stderr)
        return JSONResponse(random.choice(action_entries))


# ---------------------------------------------------------------------------
# Static files: serve frontend/ at root.
# Must be mounted AFTER API routes so /api/* is not swallowed by StaticFiles.
# ---------------------------------------------------------------------------
_FRONTEND_DIR = _REPO_ROOT / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Check port availability before handing control to uvicorn.
    _port = int(os.environ.get("PORT", 8000))
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _sock.bind(("0.0.0.0", _port))
    except OSError:
        print(
            f"Port {_port} is in use. "
            "Stop the conflicting process or set PORT=<n> env var."
        )
        sys.exit(1)
    finally:
        _sock.close()

    print(f"Workshop hub running at http://localhost:{_port}")
    uvicorn.run("api.server:app", host="0.0.0.0", port=_port, reload=False)

