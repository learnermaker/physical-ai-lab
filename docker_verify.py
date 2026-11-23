#!/usr/bin/env python3
"""Verify the Physical AI Lab container runtime without requiring host devices.

Usage:
  python docker_verify.py
  python docker_verify.py --camera

The optional camera check is intentionally advisory: containers do not
automatically receive host webcams. The browser-based hub uses host
getUserMedia and should be tested in the host browser instead.
"""
from __future__ import annotations

import argparse
import importlib
import sys

PACKAGES = [
    "numpy",
    "mujoco",
    "gymnasium",
    "stable_baselines3",
    "cv2",
    "mediapipe",
    "google.genai",
    "torch",
    "PIL",
    "matplotlib",
    "fastapi",
    "uvicorn",
]

def check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[PASS] {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:  # diagnostics should continue through all checks
        print(f"[FAIL] {label} — {type(exc).__name__}: {exc}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true",
                        help="Advisory check for a camera device visible inside this container")
    args = parser.parse_args()

    passed = 0
    failed = 0
    for name in PACKAGES:
        ok = check(f"import {name}", lambda n=name: importlib.import_module(n).__name__)
        passed += int(ok)
        failed += int(not ok)

    def cartpole():
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=7)
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        env.close()
        return f"observation shape={obs.shape}, reward={reward}"

    ok = check("Gymnasium CartPole step", cartpole)
    passed += int(ok)
    failed += int(not ok)

    def reacher():
        import gymnasium as gym
        env = gym.make("Reacher-v5")
        try:
            obs, _ = env.reset(seed=7)
            env.step(env.action_space.sample())
            return f"observation shape={obs.shape}"
        finally:
            env.close()

    ok = check("MuJoCo Reacher physics step", reacher)
    passed += int(ok)
    failed += int(not ok)

    def torch_cpu():
        import torch
        value = torch.tensor([1.0, 2.0, 3.0])
        return f"version={torch.__version__}, device={value.device}"

    ok = check("PyTorch CPU tensor", torch_cpu)
    passed += int(ok)
    failed += int(not ok)

    def fastapi_import():
        import api.server
        return "API application imported"

    ok = check("FastAPI application import", fastapi_import)
    passed += int(ok)
    failed += int(not ok)

    if args.camera:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            available = cap.isOpened()
            cap.release()
            if available:
                print("[PASS] container camera device detected")
            else:
                print("[WARN] no camera device visible inside container; expected unless explicitly passed through")
        except Exception as exc:
            print(f"[WARN] container camera test unavailable: {exc}")
    else:
        print("[INFO] Camera is not tested in-container. Test the hub camera from the host browser at http://localhost:8000.")

    print(f"\nDocker verification: {passed} passed, {failed} failed")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
