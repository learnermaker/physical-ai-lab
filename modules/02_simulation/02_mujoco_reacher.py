#!/usr/bin/env python3
"""
Module 2 Demo 2: MuJoCo Reacher-v5
=====================================
Introduces the environment used throughout Modules 2–5: a two-link robot arm
(shoulder + elbow) that must reach a target point.

Reacher-v5 observation vector — shape (10,):
  obs[0] = cos(θ₁)             — cosine of shoulder joint angle
  obs[1] = cos(θ₂)             — cosine of elbow joint angle
  obs[2] = sin(θ₁)             — sine of shoulder joint angle
  obs[3] = sin(θ₂)             — sine of elbow joint angle
  obs[4] = target_x            — target x position in world space (metres)
  obs[5] = target_y            — target y position in world space (metres)
  obs[6] = ω₁                  — shoulder angular velocity (rad/s)
  obs[7] = ω₂                  — elbow angular velocity (rad/s)
  obs[8] = fingertip_x - target_x  ← RELATIVE vector (NOT absolute position!)
  obs[9] = fingertip_y - target_y  ← distance = sqrt(obs[8]² + obs[9]²)

NOTE: obs[8] and obs[9] are the vector FROM the target TO the fingertip,
not absolute fingertip coordinates.  This changed in v5.

Action vector — shape (2,):
  action[0] = torque on joint 1 (shoulder)  — range [-1, 1]
  action[1] = torque on joint 2 (elbow)     — range [-1, 1]

Why cos/sin instead of the angle directly?
  Angles wrap around: 359° and 1° are physically adjacent but numerically
  far apart.  cos/sin representation is smooth and continuous everywhere,
  which makes learning much easier and prevents large policy jumps at wrap-around.

How it connects to the pipeline:
  This demo shows the simulation layer.  In Module 4, your hand movements
  will generate the action vector that drives this same environment.

Run from the repo root:
  python modules/02_simulation/02_mujoco_reacher.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env  # handles OpenGL / headless fallback


def main() -> None:
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # Reacher-v5 is the environment used in Modules 2–4. Understanding its
    # observation space (what the agent knows) and action space (what it can do)
    # is essential before you try to control it. In Module 4, YOUR HAND will
    # generate the action vector [torque1, torque2] that steps this environment.
    # The obs[8,9] relative vector is how the arm "knows" how far it is from
    # the target — the same error signal a real robot arm PID controller uses.
    # ─────────────────────────────────────────────────────────────────────────
    env = make_env("Reacher-v5", render_mode="human")

    # ── Print dimensionality ─────────────────────────────────────────────────
    obs_dim    = env.observation_space.shape[0]   # 10
    action_dim = env.action_space.shape[0]         # 2

    print(f"Observation space: {env.observation_space.shape}  (obs_dim={obs_dim})")
    print(f"  obs[0..1] = cos(θ₁), cos(θ₂)   — joint angles (cos)")
    print(f"  obs[2..3] = sin(θ₁), sin(θ₂)   — joint angles (sin)")
    print(f"  obs[4..5] = target_x, target_y  — world position of target")
    print(f"  obs[6..7] = ω₁, ω₂             — joint angular velocities")
    print(f"  obs[8..9] = fingertip-target vector  — norm = distance to target")
    print()
    print(f"Action space:      {env.action_space.shape}       (action_dim={action_dim})")
    print(f"  action[0] = torque on shoulder joint  (range [-1, 1])")
    print(f"  action[1] = torque on elbow joint     (range [-1, 1])")
    print()

    # ── Run 100 random-action steps ──────────────────────────────────────────
    obs, info = env.reset()

    print("Running 100 random-action steps...")
    print(f"Initial fingertip-to-target distance: {(obs[8]**2 + obs[9]**2)**0.5:.4f} m")

    for step in range(100):
        # Sample a random torque for each joint from [-1, 1]
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        # reward = -distance_to_target - 0.1*|action|² (penalises both distance and large torques)

        # Episode ends after 50 steps (Reacher-v5 default) — truncated=True
        if terminated or truncated:
            obs, info = env.reset()

    print(f"After 100 steps (random policy):")
    print(f"  Final fingertip-to-target distance: {(obs[8]**2 + obs[9]**2)**0.5:.4f} m")
    print(f"  (Random policy doesn't learn — distance stays roughly constant)")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
