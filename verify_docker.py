#!/usr/bin/env python3
"""
verify_docker.py - Docker-specific verification for the Physical AI Lab.

Unlike verify_install.py (native), this script assumes a HEADLESS container:
  - No physical webcam is expected (browser getUserMedia is the workshop path).
  - No desktop display / interactive OpenGL window is expected.
  - MuJoCo is expected to run headless (MUJOCO_GL=osmesa, MPLBACKEND=Agg).

Checks (each prints PASS / FAIL / WARN / INFO):
  1.  Python runtime (requires 3.12.x).
  2.  Essential package imports (numpy, mujoco, gymnasium, SB3, cv2,
      mediapipe, torch, PIL, matplotlib, huggingface_hub, google.genai,
      fastapi, uvicorn).
  3.  PyTorch CPU tensor execution.
  4.  NumPy operation.
  5.  MediaPipe import + HandLandmarker API availability.
  6.  MuJoCo import.
  7.  Gymnasium environment creation (CartPole-v1, render_mode=None).
  8.  At least one MuJoCo physics step (Reacher-v5 via utils.gym_utils).
  9.  Stable-Baselines3 import + PPO construction (no training).
  10. FastAPI application import (api.server:app).
  11. Backend health (http://localhost:<PORT>/api/config, PORT env default 8000).

Camera testing is intentionally NOT done here - the browser camera workflow
must be tested from the host browser (see DOCKER_SETUP.md § Camera).

Exits 0 when no FAIL, 1 otherwise. WARN/INFO do not fail the run.
Environments are closed correctly; temporary files are not left behind.
"""

from __future__ import annotations

import importlib
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

passed = 0
failed = 0
warned = 0


def _ok(label: str, detail: str = "") -> None:
    global passed
    passed += 1
    print(f"[PASS] {label}" + (f" - {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    global failed
    failed += 1
    print(f"[FAIL] {label}" + (f": {detail}" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    global warned
    warned += 1
    print(f"[WARN] {label}" + (f" - {detail}" if detail else ""))


def _info(label: str, detail: str = "") -> None:
    print(f"[INFO] {label}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Python runtime
# ---------------------------------------------------------------------------
if sys.version_info[:2] == (3, 12):
    _ok("python", f"{sys.version.split()[0]} (requires 3.12.x)")
else:
    _fail("python", f"found {sys.version.split()[0]}, requires 3.12.x")

# ---------------------------------------------------------------------------
# 2. Essential package imports
# ---------------------------------------------------------------------------
ESSENTIAL = [
    "numpy",
    "mujoco",
    "gymnasium",
    "stable_baselines3",
    "cv2",
    "mediapipe",
    "torch",
    "PIL",
    "matplotlib",
    "huggingface_hub",
    "google.genai",
    "fastapi",
    "uvicorn",
]

for name in ESSENTIAL:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        _ok(name, f"v{ver}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        _fail(name, f"{type(exc).__name__}: {exc}")

# Headless env expectations - informational only.
_info("headless env", f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(unset)')}, "
      f"MPLBACKEND={os.environ.get('MPLBACKEND', '(unset)')}")

# ---------------------------------------------------------------------------
# 3. PyTorch CPU tensor execution
# ---------------------------------------------------------------------------
try:
    import torch

    x = torch.randn(3, 3)
    y = x @ x.T
    assert y.shape == (3, 3)
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if torch.cuda.is_available():
        _warn("torch cpu", "CUDA available - workshop uses CPU tensors only")
    else:
        _ok("torch cpu", f"matmul ok, device={device}, v{torch.__version__}")
except Exception as exc:  # noqa: BLE001
    _fail("torch cpu", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 4. NumPy operation
# ---------------------------------------------------------------------------
try:
    import numpy as np

    a = np.arange(6, dtype=float).reshape(2, 3)
    assert float(a.sum()) == 15.0
    _ok("numpy", f"sum ok, v{np.__version__}")
except Exception as exc:  # noqa: BLE001
    _fail("numpy", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 5. MediaPipe + HandLandmarker API
# ---------------------------------------------------------------------------
try:
    from mediapipe.tasks.python import vision as mp_vision

    assert hasattr(mp_vision, "HandLandmarker"), "HandLandmarker missing"
    assert hasattr(mp_vision, "HandLandmarkerOptions"), "HandLandmarkerOptions missing"
    _ok("mediapipe tasks", "HandLandmarker API available")
    # Legacy mp.solutions is used by modules/01_perception/02_hand_tracking.py.
    try:
        import mediapipe as mp  # noqa: PLC0415

        _ = mp.solutions.hands.HAND_CONNECTIONS  # type: ignore[attr-defined]
        _ok("mediapipe solutions", "HAND_CONNECTIONS available")
    except Exception:
        _warn("mediapipe solutions",
              "mp.solutions absent (mediapipe>=0.10.22/1.x) - "
              "02_hand_tracking.py needs mediapipe 0.10.21; "
              "browser MediaPipe (Module 1) is unaffected")
except Exception as exc:  # noqa: BLE001
    _fail("mediapipe tasks", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 6-8. Gymnasium + MuJoCo physics step
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(REPO_ROOT))
    from utils.gym_utils import make_env  # noqa: E402

    # 7. CartPole creation without rendering (training path).
    # NOTE: make_env always probes human->rgb_array->None from the top, so for
    # a pure headless training-path check we use gym.make directly here.
    import gymnasium as gym  # noqa: PLC0415

    env = gym.make("CartPole-v1", render_mode=None)
    env.reset()
    env.step(env.action_space.sample())
    env.close()
    _ok("gymnasium CartPole-v1", "create + reset + step (headless)")

    # 8. At least one MuJoCo physics step via the shared fallback chain.
    env2 = make_env("Reacher-v5", render_mode="rgb_array")
    obs, _ = env2.reset()
    assert len(obs) == 10, f"unexpected Reacher obs len {len(obs)}"
    env2.step(env2.action_space.sample())
    env2.close()
    _ok("mujoco physics", "Reacher-v5 reset + 1 step via make_env")
except Exception as exc:  # noqa: BLE001
    msg = str(exc).lower()
    if any(k in msg for k in ("opengl", "glfw", "display", "egl", "gladloadgl")):
        _warn("mujoco physics", f"headless render unavailable: {exc}")
    else:
        _fail("gymnasium/mujoco", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 9. Stable-Baselines3 PPO construction (no training - fast)
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym  # noqa: PLC0415
    from stable_baselines3 import PPO  # noqa: PLC0415

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0)
    env.close()
    _ok("stable-baselines3", f"PPO(MlpPolicy) constructed, v{PPO.__module__}")
except Exception as exc:  # noqa: BLE001
    _fail("stable-baselines3", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 10. FastAPI application import
# ---------------------------------------------------------------------------
try:
    from api.server import app  # noqa: PLC0415, E402

    routes = [getattr(r, "path", "?") for r in app.routes]
    assert "/api/config" in routes, "/api/config route missing"
    assert "/api/status" in routes, "/api/status route missing"
    _ok("fastapi app", f"{len(routes)} routes, /api/config + /api/status present")
except Exception as exc:  # noqa: BLE001
    _fail("fastapi app", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 11. Backend health (only passes when the server is running)
# ---------------------------------------------------------------------------
port = int(os.environ.get("PORT", "8000"))
url = f"http://localhost:{port}/api/config"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status == 200 and "gemini_key_configured" in body:
            _ok("backend health", f"{url} -> 200")
        else:
            _fail("backend health", f"{url} -> {resp.status} unexpected body")
except Exception as exc:  # noqa: BLE001
    _warn("backend health", f"server not reachable at {url} ({type(exc).__name__}) - "
          "start it with docker compose up, then re-run this script")

# ---------------------------------------------------------------------------
# Camera note - always informational, never a failure.
# ---------------------------------------------------------------------------
_info("camera", "not checked in container - test webcam in host browser via Module 1 "
      "(Start -> allow permission) or the demo video if no camera")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
suffix = f" ({warned} warnings)" if warned else ""
print(f"\nDocker verification: {passed} passed, {failed} failed{suffix}")
sys.exit(0 if failed == 0 else 1)
