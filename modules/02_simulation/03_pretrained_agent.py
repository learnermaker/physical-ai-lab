#!/usr/bin/env python3
"""
Module 2 Demo: Pre-trained SAC Agent on HalfCheetah-v5
Loads a pre-trained Soft Actor-Critic model and runs 3 evaluation episodes.

HalfCheetah-v5 is a continuous-action MuJoCo environment — the agent must
apply torques across 6 joints simultaneously, which is why SAC (rather than
PPO) is used: SAC is designed for continuous action spaces.

This script is referenced in PRE_WORKSHOP_SETUP.md as a pre-session preview.

If OpenGL / display rendering is unavailable, the render fallback in
make_env() activates silently (rgb_array → None), so the script works on
headless machines without any code changes.
"""

import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/02_simulation/
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC  # noqa: E402

from utils.gym_utils import make_env  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "sac-HalfCheetah-v5.zip"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found: {MODEL_PATH}\n"
            "Ensure 'models/sac-HalfCheetah-v5.zip' is present in the repository."
        )

    model = SAC.load(str(MODEL_PATH))
    env = make_env("HalfCheetah-v5", render_mode="human")

    for i in range(3):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {i + 1} reward: {episode_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
