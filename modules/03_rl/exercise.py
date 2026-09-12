#!/usr/bin/env python3
"""
Module 3 Exercise: Experiment with Training Duration
Expected output: Training curve saved to models/training_curve.png (or shown
                 interactively); final mean episode reward printed to console.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from stable_baselines3 import PPO
from utils.gym_utils import make_env, LivePlotCallback

# Adapted from SB3 Callbacks documentation:
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)


def main():
    env = make_env("CartPole-v1", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, verbose=0)
    callback = LivePlotCallback(check_freq=1000)

    # TODO START — Set total_timesteps to observe how training duration affects
    # the reward curve.  Try values like 10_000, 50_000, or 100_000 and compare
    # the results.
    raise NotImplementedError(
        "Implement main() at line 30. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END

    # SOLUTION HINT: Replace the NotImplementedError with a call to
    # model.learn(total_timesteps=YOUR_VALUE, callback=callback).
    # Choose a value different from the default 50_000 — for example 20_000
    # or 100_000 — to see how training duration affects the final reward curve.
    # Then save the model and print the final mean reward.

    # EXPECTED OUTPUT: training_result
    # Final mean episode reward: <float>

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
