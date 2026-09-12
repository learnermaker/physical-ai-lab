#!/usr/bin/env python3
"""
Module 3 — Demo 2: Train PPO on CartPole-v1 with a live reward plot.

Trains a fresh PPO agent for 50,000 timesteps and saves the result to
models/ppo-CartPole-trained.zip.  A LivePlotCallback updates the reward
curve every 1,000 steps; on headless machines it falls back to file output.
"""

import sys
from pathlib import Path

# Resolve repository root (this file lives at modules/03_rl/, two levels down)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from utils.gym_utils import make_env, LivePlotCallback

# Adapted from SB3 Callbacks documentation:
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)

SAVE_PATH = REPO_ROOT / "models" / "ppo-CartPole-trained.zip"

# ── Environment ────────────────────────────────────────────────────────────────
# rgb_array avoids OpenGL/display issues on headless workshop laptops;
# make_env will automatically fall back further if needed.
env = make_env("CartPole-v1", render_mode="rgb_array")

# ── Model ──────────────────────────────────────────────────────────────────────
model = PPO("MlpPolicy", env, verbose=1)

# ── Callback ───────────────────────────────────────────────────────────────────
callback = LivePlotCallback(check_freq=1000)

# ── Training ───────────────────────────────────────────────────────────────────
model.learn(total_timesteps=50_000, callback=callback)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(str(SAVE_PATH))

# ── Report final mean reward ───────────────────────────────────────────────────
buf = model.ep_info_buffer
if buf:
    import numpy as np
    mean_reward = float(np.mean([ep["r"] for ep in buf]))
    print(f"Final mean episode reward (last {len(buf)} episodes): {mean_reward:.2f}")
else:
    print("Final mean episode reward: no episodes recorded in buffer")

# ── Cleanup ────────────────────────────────────────────────────────────────────
env.close()
print(f"Model saved to {SAVE_PATH}")
