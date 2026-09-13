#!/usr/bin/env python3
"""
verify_install.py — Physical AI Lab environment verification script.

Performs 14 checks:
  - 12 package import checks
  - 1 webcam check
  - 1 CartPole Gymnasium environment check

Prints [PASS], [FAIL], or [WARN] per check.
Exits 0 if no failures (warnings are acceptable), 1 if any check failed.

Must complete within 30 seconds.
"""

import importlib
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Package table: (import_name, pip_install_target)
# ---------------------------------------------------------------------------
PACKAGES = [
    ("numpy",            "numpy"),
    ("mujoco",           "mujoco"),
    ("gymnasium",        "gymnasium"),
    ("stable_baselines3","stable-baselines3"),
    ("cv2",              "opencv-python"),
    ("mediapipe",        "mediapipe"),
    ("cvzone",           "cvzone"),
    ("google.genai",     "google-genai"),
    ("torch",            "torch"),
    ("PIL",              "Pillow"),
    ("matplotlib",       "matplotlib"),
    ("huggingface_hub",  "huggingface_hub"),
]

passed = 0
failed = 0
warns  = 0

# ---------------------------------------------------------------------------
# Check 1–12: Package imports
# ---------------------------------------------------------------------------
for import_name, pip_name in PACKAGES:
    try:
        importlib.import_module(import_name)
        print(f"[PASS] {import_name}")
        passed += 1
    except ImportError:
        print(f"[FAIL] {import_name}")
        print(f"       -> pip install {pip_name}")
        failed += 1

def _is_render_warn(exc: Exception) -> bool:
    """Mirror the gym_utils._is_render_error logic for verify_install."""
    msg = str(exc).lower()
    type_name = type(exc).__name__.lower()
    return (
        any(p in msg for p in _RENDER_KEYWORDS)
        or "fatalerror" in type_name
        or "gladloadgl" in msg
    )


# ---------------------------------------------------------------------------
# Check 13/14: Webcam
# ---------------------------------------------------------------------------
try:
    import cv2  # already imported via the loop above if it passed

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            # Display a live window for at most 2 seconds (or until 'q')
            deadline = time.time() + 2.0
            while time.time() < deadline:
                ret2, frame = cap.read()
                if not ret2:
                    break
                cv2.imshow("verify — press q to close", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            cv2.destroyWindow("verify — press q to close")
            cap.release()
            print("[PASS] camera")
            passed += 1
        else:
            cap.release()
            print("[WARN] Camera unavailable — fallback video will be used")
            warns += 1
    else:
        cap.release()
        print("[WARN] Camera unavailable — fallback video will be used")
        warns += 1
except Exception:
    # cv2 itself failed to import (already counted as [FAIL] above); skip
    print("[WARN] Camera unavailable — fallback video will be used")
    warns += 1

# ---------------------------------------------------------------------------
# Check 14/14: CartPole Gymnasium environment — uses make_env to exercise
# the real render fallback chain that participants will encounter.
# ---------------------------------------------------------------------------
_RENDER_KEYWORDS = (
    "opengl", "glfw", "display", "egl", "pyglet",
    "gl error", "render", "framebuffer", "glx",
    # Added to match gym_utils._RENDER_EXCEPTION_PATTERNS — covers Windows
    # no-GPU failures (access violation on GLFW init, gladLoadGL on render).
    "access violation", "gladloadgl", "fatalerror", "wgl",
)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.gym_utils import make_env  # noqa: E402

    env = make_env("CartPole-v1", render_mode="rgb_array")
    if isinstance(env, tuple):   # guard: make_env returns env, not a tuple
        env = env[0]
    env.reset()
    env.step(env.action_space.sample())
    env.close()
    print("[PASS] gymnasium CartPole-v1")
    passed += 1
except ImportError:
    # utils/gym_utils not yet built; fall back to direct gym.make
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("[PASS] gymnasium CartPole-v1")
        passed += 1
    except Exception as e:
        if _is_render_warn(e):
            print("[WARN] OpenGL unavailable — headless rendering will be used")
            warns += 1
        else:
            print(f"[FAIL] gymnasium CartPole-v1: {e}")
            failed += 1
except Exception as e:
    if _is_render_warn(e):
        print("[WARN] OpenGL unavailable — headless rendering will be used")
        warns += 1
    else:
        print(f"[FAIL] gymnasium CartPole-v1: {e}")
        failed += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
warn_suffix = f" ({warns} warnings — fallbacks active)" if warns > 0 else ""
print(f"\nSetup complete: {passed}/14 checks passed{warn_suffix}")

sys.exit(0 if failed == 0 else 1)
