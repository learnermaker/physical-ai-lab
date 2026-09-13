#!/usr/bin/env python3
"""
Module 3 Exercise: Experiment with Training Duration
======================================================
Your task: call model.learn() with different values of total_timesteps
and observe how training duration affects the reward curve.

Try: 10_000, 50_000, 100_000

Questions to answer:
  - At what timestep does the reward start rising sharply?
  - Does training for 10× longer give you 10× better performance?
  - Where does the curve flatten — the point of diminishing returns?

This maps directly to a real Physical AI deployment decision: how long
do you run simulation training before you trust the policy enough to
deploy on physical hardware?

Start here:  Find the TODO block below (~line 32) and replace the
             NotImplementedError with a model.learn() call.
             The solution is commented out at the bottom of this file.

Run from the repo root:
  python modules/03_rl/exercise.py

Adapted from SB3 Callbacks docs:
  https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from stable_baselines3 import PPO
from utils.gym_utils import make_env, LivePlotCallback


def main():
    # render_mode=None: no visual window during training (faster)
    env = make_env("CartPole-v1", render_mode=None)

    # PPO with a small MLP policy — same setup as 02_train_cartpole.py
    model = PPO("MlpPolicy", env, verbose=0)

    # LivePlotCallback draws a reward chart that updates every 1,000 steps.
    # On headless machines (no monitor) it saves to models/training_curve.png.
    callback = LivePlotCallback(check_freq=1000)

    try:
        # ── TODO: replace the NotImplementedError with a model.learn() call ──
        # Syntax:  model.learn(total_timesteps=YOUR_VALUE, callback=callback)
        # Try different values: 10_000, 50_000, 100_000
        # Watch how the reward curve shape changes with each value.
        raise NotImplementedError(
            "Implement main() at line 30. "
            "See # EXPECTED OUTPUT comment below."
        )
        # ── TODO END ─────────────────────────────────────────────────────────
        # SOLUTION HINT: Replace the NotImplementedError with:
        #   model.learn(total_timesteps=20_000, callback=callback)
        # Then change the number and re-run to compare curves.
        #
        # EXPECTED OUTPUT:
        # Final mean episode reward (last N episodes): <float>
        # (Higher = better.  CartPole maximum = 500.)
    finally:
        # env.close() must run even if an exception is raised
        env.close()


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main():
#     env = make_env("CartPole-v1", render_mode="rgb_array")
#     model = PPO("MlpPolicy", env, verbose=0)
#     callback = LivePlotCallback(check_freq=1000)
#
#     model.learn(total_timesteps=20_000, callback=callback)
#
#     save_path = REPO_ROOT / "models" / "ppo-CartPole-exercise.zip"
#     model.save(str(save_path))
#
#     buf = model.ep_info_buffer
#     if buf:
#         mean_reward = float(np.mean([ep["r"] for ep in buf]))
#         print(f"Final mean episode reward (last {len(buf)} episodes): {mean_reward:.2f}")
#     else:
#         print("Final mean episode reward: no episodes recorded in buffer")
#
#     env.close()
#     print(f"Model saved to {save_path}")
