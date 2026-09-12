#!/usr/bin/env python3
"""
Module 2 Demo: Gymnasium Introduction
Creates a CartPole-v1 environment, prints its observation and action spaces,
then runs 200 random-action steps with rendering.

OpenGL/display fallbacks are handled transparently by make_env():
    human → rgb_array → None
No code changes are needed by the participant when OpenGL is unavailable.

Press Ctrl+C to exit early.
"""

import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/02_simulation/
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env


def main() -> None:
    env = make_env("CartPole-v1")

    print(f"observation_space: {env.observation_space}")
    print(f"action_space:      {env.action_space}")

    obs, info = env.reset()

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # render() is a no-op when render_mode is None (headless fallback).
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
