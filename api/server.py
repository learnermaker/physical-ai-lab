"""
Experience Hub — FastAPI backend for the Physical AI Lab.

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
# ARIA — Assistive Robot for Intelligent Awareness
# Richer JSON schema: confidence, observation, reason, next_step
_ACTION_PROMPT = (
    "You are ARIA, an assistive robot in a workspace. Your job is to observe people "
    "and decide if they need help. Look at the person in this image. "
    "Respond ONLY in JSON with no markdown: "
    '{"action": "APPROACH|WAIT|ALERT|RETREAT", '
    '"confidence": 0.0, '
    '"observation": "one sentence describing what you see", '
    '"reason": "one sentence explaining your action choice", '
    '"next_step": "one sentence describing what ARIA does next"} '
    "— APPROACH if they seem confused, stuck, or are inviting interaction; "
    "WAIT if they are working calmly and do not need help; "
    "ALERT if they appear unwell, distressed, or unresponsive; "
    "RETREAT if they are leaving or clearly want space."
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Physical AI Lab Hub")


# ---------------------------------------------------------------------------
# Middleware: COOP + COEP headers required for MediaPipe WASM SharedArrayBuffer
# in Chrome/Edge. Must be registered before any routes are added.
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        response: StarletteResponse = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

# In-memory job registry: {job_id: queue.Queue}
# The training thread puts progress dicts; the SSE endpoint consumes them.
_jobs: dict[str, queue.Queue] = {}

# Concurrency guard for simulation stream — one active MuJoCo env at a time.
# MuJoCo's GLFW/EGL context is not safe to use from multiple concurrent threads.
_sim_lock   = threading.Lock()    # Lock (not Semaphore) — double-release raises RuntimeError immediately
_sim_stop   = threading.Event()   # Set to signal the worker to exit at any safe point.
                                  # Cleared automatically when a new stream successfully acquires the lock.
_sim_action = np.zeros(2, dtype=np.float32)   # Mutable action updated via POST /api/simulation/action.
                                              # Worker reads this each step — no reconnect needed on move.

# ---------------------------------------------------------------------------
# Gemini response cache — prevents duplicate calls on rapid double-click.
# Each participant runs their own server with their own API key, so this
# cache is per-user and per-process. TTL is kept short (5 s) so that moving
# the camera and clicking again always gets a fresh response.
# ---------------------------------------------------------------------------
import hashlib as _hashlib  # noqa: E402
import time as _time_module  # noqa: E402

_gemini_response_cache: dict = {}          # {hash: (timestamp, response_dict)}
_GEMINI_CACHE_TTL_S: int = 5               # seconds — short enough that a new frame always gets a fresh call


def _gemini_cache_get(cache_key: str) -> dict | None:
    """Return cached Gemini response if still fresh, else None."""
    entry = _gemini_response_cache.get(cache_key)
    if entry is None:
        return None
    ts, payload = entry
    if _time_module.monotonic() - ts > _GEMINI_CACHE_TTL_S:
        del _gemini_response_cache[cache_key]
        return None
    return payload


def _gemini_cache_put(cache_key: str, payload: dict) -> None:
    """Store a Gemini response; evict entries older than 2× TTL to cap memory."""
    now = _time_module.monotonic()
    _gemini_response_cache[cache_key] = (now, payload)
    # Evict stale entries if the cache grows large
    if len(_gemini_response_cache) > 200:
        cutoff = now - 2 * _GEMINI_CACHE_TTL_S
        stale = [k for k, (ts, _) in list(_gemini_response_cache.items()) if ts < cutoff]
        for k in stale:
            _gemini_response_cache.pop(k, None)


def _gemini_cache_key(endpoint: str, api_key: str, frame_bytes: bytes) -> str:
    h = _hashlib.sha256(f"{endpoint}:{api_key}:".encode() + frame_bytes).hexdigest()[:16]
    return h


@app.get("/api/open-file")
async def open_file(path: str = Query(...)) -> JSONResponse:
    """Open a repo file in the OS default handler.

    Accepts a relative path (e.g. ``modules/01_perception/exercise.py``).
    Resolves it against the repo root and calls the OS default file handler —
    whatever the participant has set as their default for ``.py`` / ``.ipynb``
    (VS Code, Cursor, PyCharm, Notepad, etc.).

    Returns JSON with ``opened: true`` on success, ``error: str`` on failure.
    Only paths inside the repo root are permitted (path traversal guard).
    """
    import subprocess  # noqa: PLC0415

    try:
        # Resolve and validate — must stay inside the repo root
        target = (_REPO_ROOT / path).resolve()
        if not str(target).startswith(str(_REPO_ROOT)):
            return JSONResponse({"opened": False, "error": "Path outside repo"}, status_code=400)
        if not target.exists():
            return JSONResponse({"opened": False, "error": f"File not found: {path}"}, status_code=404)

        # Use the OS default handler — respects user's configured IDE
        import sys as _sys  # noqa: PLC0415
        if _sys.platform == "win32":
            import os as _os  # noqa: PLC0415
            _os.startfile(str(target))
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:  # Linux / other
            subprocess.Popen(["xdg-open", str(target)])

        return JSONResponse({"opened": True, "path": str(target)})
    except Exception as exc:
        return JSONResponse({"opened": False, "error": str(exc)}, status_code=500)


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Return whether a Gemini API key is configured server-side."""
    return JSONResponse({"gemini_key_configured": _GEMINI_API_KEY is not None})


@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Return a detailed health snapshot of all services.

    Used by the hub's server-control panel to show per-service status without
    requiring a full page reload.  Never raises — always returns 200 so the
    client can always read the response.
    """
    import importlib  # noqa: PLC0415

    # Check each optional dependency can be imported
    dep_ok: dict[str, bool] = {}
    for pkg in ("gymnasium", "mujoco", "stable_baselines3", "google.genai", "mediapipe"):
        try:
            importlib.import_module(pkg)
            dep_ok[pkg] = True
        except ImportError:
            dep_ok[pkg] = False

    active_jobs = {jid: not q.empty() for jid, q in list(_jobs.items())}
    return JSONResponse({
        "server": "ok",
        "uptime_s": None,          # placeholder — could add a startup timestamp later
        "gemini_key_configured": _GEMINI_API_KEY is not None,
        "gemini_cache_loaded": bool(_CACHE),
        "active_training_jobs": len([v for v in active_jobs.values() if v]),
        "total_training_jobs": len(active_jobs),
        "deps": dep_ok,
    })


# --- Stubs for tasks 21.2 – 21.4 (return 501 until implemented) -----------


def _make_status_frame(obs: np.ndarray, action: np.ndarray, step: int) -> str:
    """Render a 640x480 JPEG frame showing the Reacher-v5 arm geometry.

    Draws the two-link arm, target, fingertip, and torque indicators using
    PIL — no GPU or OpenGL required.

    Reacher-v5 observation layout (10 values, per gymnasium docs):
        obs[0] = cos(θ₁)   obs[1] = cos(θ₂)   — joint angles (shoulder, elbow)
        obs[2] = sin(θ₁)   obs[3] = sin(θ₂)
        obs[4] = target_x  obs[5] = target_y   — target world position
        obs[6] = ω₁        obs[7] = ω₂         — angular velocities
        obs[8] = fingertip_x − target_x        — relative vector (NOT absolute)
        obs[9] = fingertip_y − target_y
    Fingertip absolute position is recovered via forward kinematics.
    Distance to target = hypot(obs[8], obs[9]).
    Each arm link is 0.1 m.
    """
    import io  # noqa: PLC0415
    import math  # noqa: PLC0415

    from PIL import Image, ImageDraw  # noqa: PLC0415

    # ── canvas ─────────────────────────────────────────────────────────────
    W, H = 640, 480
    BG    = (15, 18, 28)
    img  = Image.new("RGB", (W, H), color=BG)
    draw = ImageDraw.Draw(img)

    # ── colour palette ──────────────────────────────────────────────────────
    C_GRID    = (30, 35, 50)
    C_AXIS    = (45, 50, 70)
    C_LINK1   = (80, 160, 230)    # upper arm — blue
    C_LINK2   = (60, 210, 140)    # forearm — teal
    C_JOINT   = (255, 255, 255)
    C_FINGER  = (255, 210, 60)    # fingertip — gold
    C_TARGET  = (230, 80, 80)     # target — red
    C_TORQ1   = (80, 160, 230)
    C_TORQ2   = (60, 210, 140)
    C_TEXT    = (180, 185, 200)
    C_DIM     = (80, 85, 100)
    C_TITLE   = (200, 180, 120)

    # ── world→pixel mapping ─────────────────────────────────────────────────
    # Viewport: left 400 px for the arm diagram; right 240 px for data panel.
    VW, VH = 400, 480          # viewport dimensions
    MARGIN = 60                # pixels of padding inside the viewport
    SCALE  = (VW - 2 * MARGIN) / 0.5   # world range ≈ [-0.25, 0.25] → pixels
    OX, OY = VW // 2, VH // 2  # pixel origin = centre of viewport

    def w2p(wx: float, wy: float):
        """World coords → pixel coords (y flipped: +y is up in MuJoCo)."""
        px = int(OX + wx * SCALE)
        py = int(OY - wy * SCALE)
        return (px, py)

    # ── background grid ─────────────────────────────────────────────────────
    for gv in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        x0, y0 = w2p(gv, -0.3)
        x1, y1 = w2p(gv,  0.3)
        draw.line([(x0, y0), (x1, y1)], fill=C_GRID, width=1)
        x0, y0 = w2p(-0.3, gv)
        x1, y1 = w2p( 0.3, gv)
        draw.line([(x0, y0), (x1, y1)], fill=C_GRID, width=1)

    # Axis lines
    sx, sy = w2p(-0.26, 0); ex, ey = w2p(0.26, 0)
    draw.line([(sx, sy), (ex, ey)], fill=C_AXIS, width=1)
    sx, sy = w2p(0, -0.26); ex, ey = w2p(0, 0.26)
    draw.line([(sx, sy), (ex, ey)], fill=C_AXIS, width=1)

    # ── extract geometry ────────────────────────────────────────────────────
    # Official Reacher-v5 obs layout (10 values, from gymnasium docs):
    #   obs[0] = cos(θ₁)  obs[1] = cos(θ₂)   — joint 1 (shoulder), joint 2 (elbow)
    #   obs[2] = sin(θ₁)  obs[3] = sin(θ₂)
    #   obs[4] = target_x   obs[5] = target_y   — target in world coords
    #   obs[6] = ω₁          obs[7] = ω₂         — angular velocities
    #   obs[8] = fingertip_x − target_x          — relative vector (NOT absolute!)
    #   obs[9] = fingertip_y − target_y
    cos1, cos2 = float(obs[0]), float(obs[1])
    sin1, sin2 = float(obs[2]), float(obs[3])
    target_x, target_y  = float(obs[4]), float(obs[5])
    # obs[8], obs[9] are relative (fingertip - target), not absolute positions.
    # Recover absolute fingertip via forward kinematics: two links of length L.
    rel_x, rel_y = float(obs[8]), float(obs[9])

    theta1 = math.atan2(sin1, cos1)
    theta2 = math.atan2(sin2, cos2)
    L = 0.1

    shoulder  = (0.0, 0.0)
    elbow     = (L * math.cos(theta1),
                 L * math.sin(theta1))
    # Forward kinematics: fingertip = shoulder + link1 + link2
    finger_x  = L * math.cos(theta1) + L * math.cos(theta1 + theta2)
    finger_y  = L * math.sin(theta1) + L * math.sin(theta1 + theta2)
    # Scalar distance is simply the norm of the relative vector from obs
    dist      = math.hypot(rel_x, rel_y)

    ps  = w2p(*shoulder)
    pe  = w2p(*elbow)
    pf  = w2p(finger_x, finger_y)
    pt  = w2p(target_x, target_y)

    # ── reach radius hint ───────────────────────────────────────────────────
    r_reach = int(2 * L * SCALE)
    draw.ellipse(
        [OX - r_reach, OY - r_reach, OX + r_reach, OY + r_reach],
        outline=(35, 40, 60), width=1
    )

    # ── target circle ───────────────────────────────────────────────────────
    TR = 10
    draw.ellipse([pt[0]-TR, pt[1]-TR, pt[0]+TR, pt[1]+TR],
                 fill=(80, 25, 25), outline=C_TARGET, width=2)
    draw.text((pt[0] + TR + 3, pt[1] - 7), "target", fill=C_TARGET)

    # ── distance line ───────────────────────────────────────────────────────
    # dist already computed above as hypot(obs[8], obs[9]) — the relative vector
    draw.line([pf, pt], fill=(120, 60, 60), width=1)
    mx, my = (pf[0] + pt[0]) // 2, (pf[1] + pt[1]) // 2
    draw.text((mx + 3, my - 9), f"d={dist:.3f}", fill=(160, 90, 90))

    # ── arm links ───────────────────────────────────────────────────────────
    draw.line([ps, pe], fill=C_LINK1, width=6)   # upper arm
    draw.line([pe, pf], fill=C_LINK2, width=5)   # forearm

    # ── joints ──────────────────────────────────────────────────────────────
    # Shoulder (fixed base)
    draw.ellipse([ps[0]-7, ps[1]-7, ps[0]+7, ps[1]+7],
                 fill=(40, 45, 65), outline=C_JOINT, width=2)
    draw.text((ps[0]-3, ps[1]-4), "S", fill=C_JOINT)
    # Elbow
    draw.ellipse([pe[0]-5, pe[1]-5, pe[0]+5, pe[1]+5],
                 fill=(40, 45, 65), outline=C_LINK1, width=2)
    # Fingertip
    draw.ellipse([pf[0]-5, pf[1]-5, pf[0]+5, pf[1]+5],
                 fill=C_FINGER, outline=(180, 140, 30), width=2)

    # ── torque arc indicators ────────────────────────────────────────────────
    def _torque_arc(centre, radius, torque, colour):
        """Draw a small arc indicating torque magnitude and direction."""
        if abs(torque) < 0.01:
            return
        cx, cy = centre
        span = int(abs(torque) * 120)   # max ~120° at torque=1
        span = max(span, 8)
        start = -90
        end   = start + span if torque > 0 else start - span
        draw.arc([cx-radius, cy-radius, cx+radius, cy+radius],
                 start=min(start, end), end=max(start, end),
                 fill=colour, width=3)

    _torque_arc(ps, 18, float(action[0]), C_TORQ1)
    _torque_arc(pe, 14, float(action[1]), C_TORQ2)

    # ── divider ─────────────────────────────────────────────────────────────
    draw.line([(VW, 0), (VW, H)], fill=(40, 45, 60), width=1)

    # ── data panel (right 240 px) ────────────────────────────────────────────
    PX = VW + 16   # panel left edge

    draw.text((PX, 18), "Reacher-v5", fill=C_TITLE)
    draw.text((PX, 36), f"Step {step:,}", fill=C_DIM)

    # Angles
    draw.text((PX, 66), "Joint angles", fill=C_TEXT)
    draw.text((PX, 84), f"  \u03b81 = {math.degrees(theta1):+.1f}\u00b0", fill=C_LINK1)
    draw.text((PX, 102), f"  \u03b82 = {math.degrees(theta2):+.1f}\u00b0", fill=C_LINK2)

    # Velocities
    draw.text((PX, 130), "Velocities", fill=C_TEXT)
    draw.text((PX, 148), f"  \u03c9\u2081 = {float(obs[6]):+.3f}", fill=C_DIM)
    draw.text((PX, 166), f"  \u03c9\u2082 = {float(obs[7]):+.3f}", fill=C_DIM)

    # Target
    draw.text((PX, 194), "Target", fill=C_TEXT)
    draw.text((PX, 212), f"  x = {target_x:+.3f}", fill=C_TARGET)
    draw.text((PX, 230), f"  y = {target_y:+.3f}", fill=C_TARGET)

    # Fingertip (forward-kinematics position)
    draw.text((PX, 258), "Fingertip (FK)", fill=C_TEXT)
    draw.text((PX, 276), f"  x = {finger_x:+.3f}", fill=C_FINGER)
    draw.text((PX, 294), f"  y = {finger_y:+.3f}", fill=C_FINGER)

    # Distance
    draw.text((PX, 322), "Distance to target", fill=C_TEXT)
    dist_pct = max(0.0, 1.0 - dist / 0.3)
    dist_colour = (
        int(80 + 150 * dist_pct),
        int(80 + 130 * dist_pct),
        80
    )
    draw.text((PX, 340), f"  {dist:.4f} m", fill=dist_colour)

    # Distance bar
    bar_w = 200
    bar_h = 10
    bx, by = PX, 358
    draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=(35, 40, 55))
    fill_w = int(bar_w * dist_pct)
    if fill_w > 0:
        draw.rectangle([bx, by, bx + fill_w, by + bar_h], fill=dist_colour)

    # Actions
    draw.text((PX, 386), "Applied torques", fill=C_TEXT)
    for i, (val, col) in enumerate(zip(action, [C_TORQ1, C_TORQ2])):
        by2 = 404 + i * 26
        label = f"  \u03c4{i+1} = {float(val):+.3f}"
        draw.text((PX, by2), label, fill=col)
        bx2, bar_w2 = PX + 110, 94
        cx2 = bx2 + bar_w2 // 2
        draw.rectangle([bx2, by2 + 2, bx2 + bar_w2, by2 + 14], fill=(35, 40, 55))
        fill2 = int(abs(float(val)) * (bar_w2 // 2))
        if fill2 > 0 and float(val) >= 0:
            draw.rectangle([cx2, by2 + 2, cx2 + fill2, by2 + 14], fill=col)
        elif fill2 > 0:
            draw.rectangle([cx2 - fill2, by2 + 2, cx2, by2 + 14], fill=col)
        draw.line([(cx2, by2 + 1), (cx2, by2 + 15)], fill=(80, 85, 100), width=1)

    draw.text((PX, 460), "software render (no GPU)", fill=(50, 55, 70))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/simulation/stream")
async def simulation_stream(action: str = Query("0.0,0.0")):
    """SSE: stream Reacher-v5 frames as base64 JPEG at ~20 fps.

    The initial action comes from the query param.  Subsequent action updates
    arrive via POST /api/simulation/action and are applied in-place to the
    module-level _sim_action array — no reconnection needed when the finger moves.

    The entire MuJoCo env lifecycle runs in a single dedicated daemon thread
    to avoid NULL-pointer crashes from sharing MuJoCo state across thread-pool
    workers (MuJoCo is not safe to use concurrently from multiple threads).

    Requirements: 19.4
    """
    global _sim_action
    try:
        parts = [float(x) for x in action.split(",")]
        if len(parts) != 2:
            parts = [0.0, 0.0]
    except (ValueError, AttributeError):
        parts = [0.0, 0.0]
    # Set initial action on the module-level array
    _sim_action[:] = np.clip(np.array(parts, dtype=np.float32), -1.0, 1.0)

    # Sentinel objects for inter-thread signalling
    _STOP  = object()
    _ERROR = object()

    frame_queue: queue.Queue = queue.Queue(maxsize=4)

    def _env_worker() -> None:
        """Runs entirely in one dedicated thread — no shared thread-pool.

        Acquires _sim_lock to prevent concurrent MuJoCo instances.
        Uses module-level _sim_stop (threading.Event) for interruptible sleep —
        the stop signal wakes the worker immediately rather than waiting for the
        next loop iteration or env.step() to finish.
        Waits up to 2 s for a previous worker to release — handles rapid restart.
        """
        acquired = _sim_lock.acquire(blocking=True, timeout=2.0)
        if not acquired:
            try:
                import io as _io  # noqa: PLC0415
                from PIL import Image, ImageDraw as _ID  # noqa: PLC0415
                img = Image.new("RGB", (400, 80), color=(60, 40, 10))
                d = _ID.Draw(img)
                d.text((16, 12), "Simulation busy", fill=(255, 220, 100))
                d.text((16, 32), "Another stream is already running.", fill=(220, 190, 80))
                d.text((16, 52), "Click Stop simulation, then Start again.", fill=(180, 150, 60))
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                frame_queue.put("software|" + base64.b64encode(buf.getvalue()).decode("ascii"))
                import time as _t  # noqa: PLC0415
                _t.sleep(1)
            except Exception:
                pass
            frame_queue.put(_STOP)
            return

        # We hold the lock — clear the stop event so this worker runs cleanly.
        _sim_stop.clear()

        env = None
        try:
            from utils.gym_utils import make_env  # noqa: PLC0415

            env = make_env("Reacher-v5", "rgb_array")

            actual_render_mode = getattr(env.unwrapped, "render_mode", None)
            use_status_frame = actual_render_mode not in ("rgb_array", "human")

            obs, _ = env.reset()

            while not _sim_stop.is_set():
                # Drop oldest frame if consumer is slow — keep latency low
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                obs, _reward, terminated, truncated, _info = env.step(_sim_action.copy())
                if terminated or truncated:
                    obs, _ = env.reset()

                if not use_status_frame:
                    try:
                        rgb_frame = env.render()
                        if rgb_frame is not None:
                            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                            ok, buf = cv2.imencode(
                                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60]
                            )
                            if ok:
                                frame_queue.put(
                                    "gpu|" + base64.b64encode(buf.tobytes()).decode("ascii")
                                )
                                # Interruptible sleep — wakes immediately on stop signal
                                _sim_stop.wait(timeout=0.05)
                                continue
                        use_status_frame = True
                    except Exception as render_exc:
                        print(
                            f"[sim_worker] render() failed: {render_exc} — "
                            "switching to status frames",
                            file=sys.stderr,
                        )
                        use_status_frame = True

                # PIL status-frame path (no GPU)
                step_n = getattr(_env_worker, "_step", 0)
                _env_worker._step = step_n + 1
                frame_queue.put("software|" + _make_status_frame(obs, _sim_action, step_n))
                # Interruptible sleep — wakes immediately on stop signal
                _sim_stop.wait(timeout=0.05)

        except Exception as exc:
            print(f"[sim_worker] fatal: {exc}", file=sys.stderr)
            try:
                import io as _io  # noqa: PLC0415
                from PIL import Image, ImageDraw as _ID  # noqa: PLC0415
                img = Image.new("RGB", (400, 120), color=(80, 20, 20))
                d = _ID.Draw(img)
                d.text((16, 14), "Simulation error", fill=(255, 200, 200))
                d.text((16, 38), str(exc)[:70], fill=(220, 180, 180))
                d.text((16, 60), "Stop and restart simulation to retry.", fill=(180, 140, 140))
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                frame_queue.put("software|" + base64.b64encode(buf.getvalue()).decode("ascii"))
            except Exception:
                frame_queue.put(_ERROR)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            _sim_lock.release()
            frame_queue.put(_STOP)

    # Start the dedicated worker thread
    t = threading.Thread(target=_env_worker, daemon=True)
    t.start()

    async def event_generator():
        # Heartbeat so the browser doesn't time out while MuJoCo initialises
        yield ": heartbeat\n\n"
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame = await loop.run_in_executor(
                    None, lambda: frame_queue.get(timeout=30)
                )
                if frame is _STOP or frame is _ERROR:
                    break
                yield f"data: {frame}\n\n"
        except asyncio.CancelledError:
            # Client disconnected — signal the worker immediately via the event
            _sim_stop.set()
        except queue.Empty:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/simulation/stop")
async def simulation_stop() -> JSONResponse:
    """Force-stop any running simulation worker immediately.

    Sets the module-level _sim_stop event so the worker wakes from its
    interruptible sleep on the next 50 ms tick at most, exits its loop,
    closes the MuJoCo env, and releases _sim_lock.

    Safe to call even when no simulation is running.
    """
    _sim_stop.set()
    return JSONResponse({"stopped": True})


@app.post("/api/simulation/action")
async def simulation_action(body: dict) -> JSONResponse:
    """Update the action applied to the running simulation without reconnecting.

    Accepts JSON body: {"action": [x, y]}  where x,y are floats in [-1, 1].
    Writes directly into the module-level _sim_action array that the worker
    reads on every env.step() — zero reconnection overhead.
    """
    global _sim_action
    try:
        vals = body.get("action", [0.0, 0.0])
        if len(vals) != 2:
            return JSONResponse({"error": "action must be [x, y]"}, status_code=400)
        _sim_action[:] = np.clip(np.array(vals, dtype=np.float32), -1.0, 1.0)
        return JSONResponse({"ok": True, "action": _sim_action.tolist()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


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
    # Cap job registry to prevent unbounded memory growth across a workshop session
    if len(_jobs) >= 20:
        oldest = next(iter(_jobs))
        _jobs.pop(oldest, None)
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
        loop = asyncio.get_running_loop()   # was get_event_loop() — deprecated Python 3.10+
        while True:
            try:
                item = await loop.run_in_executor(
                    None, lambda: job_queue.get(timeout=30)
                )
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("done"):
                    # Surface training error explicitly so the browser can distinguish
                    # "Training complete!" from "Training failed: <reason>"
                    break
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"
        # Remove completed job from registry to prevent unbounded growth
        _jobs.pop(job_id, None)

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
        _describe_fallbacks = ["A scene captured from the camera."]

    if not key:
        return JSONResponse({
            "text": random.choice(_describe_fallbacks),
            "_source": "cache",
            "_reason": "no_api_key",
        })

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        # Decode and optionally downscale the frame first (before creating client)
        jpg_bytes = base64.b64decode(req.frame_b64) if req.frame_b64 else b""

        if jpg_bytes:
            try:
                import io as _io  # noqa: PLC0415
                from PIL import Image as _Img  # noqa: PLC0415
                img = _Img.open(_io.BytesIO(jpg_bytes))
                img.thumbnail((512, 512), _Img.LANCZOS)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                jpg_bytes = buf.getvalue()
            except Exception:
                pass   # use original bytes if PIL fails

            image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")
            contents = ["Describe this scene in one sentence.", image_part]
        else:
            contents = ["Reply with exactly one sentence: 'API key is valid.'"]

        # Check cache BEFORE creating the API client (avoids creating it on cache hits)
        cache_key = _gemini_cache_key("describe", key, jpg_bytes)
        cached = _gemini_cache_get(cache_key)
        if cached is not None:
            return JSONResponse({**cached, "_source": "gemini", "_cached": True})

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
        )
        result = {"text": response.text}
        _gemini_cache_put(cache_key, result)
        return JSONResponse({"text": response.text, "_source": "gemini"})
    except Exception as exc:
        print(f"[Gemini describe error] {exc}", file=sys.stderr)
        return JSONResponse({
            "text": random.choice(_describe_fallbacks),
            "_source": "cache",
            "_reason": str(exc)[:120],
        })


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
        return JSONResponse({
            **random.choice(action_entries),
            "_source": "cache",
            "_reason": "no_api_key",
        })

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        # Decode and optionally downscale BEFORE cache check and client creation
        jpg_bytes = base64.b64decode(req.frame_b64) if req.frame_b64 else b""

        if jpg_bytes:
            try:
                import io as _io  # noqa: PLC0415
                from PIL import Image as _Img  # noqa: PLC0415
                img = _Img.open(_io.BytesIO(jpg_bytes))
                img.thumbnail((512, 512), _Img.LANCZOS)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                jpg_bytes = buf.getvalue()
            except Exception:
                pass

            image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")
            contents = [system_prompt, image_part]
        else:
            contents = [system_prompt + " (No image available \u2014 respond with: WAIT, reason: no image provided)"]

        # Check cache BEFORE creating the API client
        cache_key = _gemini_cache_key("action", key, jpg_bytes)
        cached = _gemini_cache_get(cache_key)
        if cached is not None:
            return JSONResponse({**cached, "_source": "gemini", "_cached": True})

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
        )

        # Strip markdown fences if the model wrapped the JSON.
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lower().startswith("json"):   # handles both ```json and ```JSON
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        if "action" not in result:
            raise ValueError(f"Response missing 'action' key: {result}")

        _gemini_cache_put(cache_key, result)
        return JSONResponse({**result, "_source": "gemini"})
    except Exception as exc:
        print(f"[Gemini action error] {exc}", file=sys.stderr)
        return JSONResponse({
            **random.choice(action_entries),
            "_source": "cache",
            "_reason": str(exc)[:120],
        })


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

    print(f"Physical AI Lab hub running at http://localhost:{_port}")
    uvicorn.run("api.server:app", host="0.0.0.0", port=_port, reload=False)

