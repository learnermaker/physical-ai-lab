#!/usr/bin/env python3
"""
Module 2 Demo: MuJoCo Reacher
Introduces the Reacher-v5 environment and its state vector.

Reacher state vector — shape (11,):
  [cos θ0, sin θ0, cos θ1, sin θ1,   # joint angles (cos/sin encoding)
   target_x, target_y,                 # target position in the plane
   angular_vel0, angular_vel1,         # joint angular velocities
   fingertip_x, fingertip_y, dist]     # fingertip position + distance to target

Action vector — shape (2,):
  [torque_joint0, torque_joint1]       # continuous torques in [-1, 1]

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
