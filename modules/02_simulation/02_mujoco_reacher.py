#!/usr/bin/env python3
"""
Module 2 Demo: MuJoCo Reacher
Introduces the Reacher-v5 environment and its state vector.

Reacher-v5 state vector — shape (10,):
  obs[0] = cos(θ₁)   obs[1] = cos(θ₂)   — joint angles, cos-encoded
  obs[2] = sin(θ₁)   obs[3] = sin(θ₂)   — joint angles, sin-encoded
  obs[4] = target_x  obs[5] = target_y   — target world position
  obs[6] = ω₁        obs[7] = ω₂         — joint angular velocities
  obs[8] = fingertip_x − target_x        — relative vector to target
  obs[9] = fingertip_y − target_y        —   (distance = hypot(obs[8], obs[9]))

NOTE: obs[8]/obs[9] are NOT absolute fingertip coordinates — they are the
vector FROM target TO fingertip. This changed in v5 (z was removed).

Action vector — shape (2,):
  [torque_joint0, torque_joint1]   — continuous torques in [-1, 1]

If OpenGL / display rendering is unavailable the render fallback in
make_env() activates silently (rgb_array → None), so the script works on
headless machines without any code changes.
"""

import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/02_simulation/
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env  # noqa: E402


def main() -> None:
    env = make_env("Reacher-v5", render_mode="human")

    # --- Print dimensionality ---
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Observation space: {env.observation_space.shape}  (obs_dim={obs_dim})")
    print(f"Action space:      {env.action_space.shape}       (action_dim={action_dim})")

    # --- Run 100 random-action steps ---
    obs, info = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("100 steps completed.")


if __name__ == "__main__":
    main()
