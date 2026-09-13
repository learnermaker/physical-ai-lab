#!/usr/bin/env python3
"""
Module 2 Demo 3: Pre-trained SAC Agent on HalfCheetah-v5
==========================================================
Loads a pre-trained robot controller and watches it run.

HalfCheetah is a 6-joint simulated cheetah body that learns to run forward.
Unlike CartPole (discrete: left/right), HalfCheetah has a *continuous* action
space — the agent must output 6 simultaneous torques as float values.

This is why we use SAC (Soft Actor-Critic) instead of PPO:
  - PPO works well for discrete and simple continuous actions
  - SAC is specifically designed for high-dimensional continuous action spaces
  - SAC maximises reward PLUS an entropy term, which encourages exploration
    and leads to smoother, more robust policies on real robots

What you will see:
  - 3 episodes of the pre-trained cheetah running
  - Episode reward printed after each episode
    (higher reward = running faster with less joint strain)

NOTE: This requires models/sac-HalfCheetah-v5.zip to exist.
      If missing, run: python download_model.py

How it connects to the pipeline:
  This shows what a *trained* agent looks like compared to the random
  policy from 01_gym_intro.py.  In Module 3 you will train your own agent.

Run from the repo root:
  python modules/02_simulation/03_pretrained_agent.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC   # Soft Actor-Critic algorithm
from utils.gym_utils import make_env

MODEL_PATH = REPO_ROOT / "models" / "sac-HalfCheetah-v5.zip"


def main() -> None:
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # This shows the end goal of RL: a trained policy running on real physics.
    # Compare the reward here to a random policy — the gap IS the value of
    # training. HalfCheetah has 6 continuous joints; coordinating them all
    # without learning is practically impossible by hand. This is why RL exists.
    # The same SAC algorithm is used in real robot locomotion research.
    # ─────────────────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found: {MODEL_PATH}\n"
            "Ensure 'models/sac-HalfCheetah-v5.zip' is present in the repository."
        )

    # SAC.load() reads the trained neural network weights from disk.
    # The model stores the policy (what action to take given a state)
    # and the value function (how good a given state is).
    model = SAC.load(str(MODEL_PATH))

    # make_env() falls back gracefully if OpenGL is unavailable
    env = make_env("HalfCheetah-v5", render_mode="human")

    print("Running 3 evaluation episodes with the pre-trained SAC agent...")
    print()

    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # model.predict() runs the policy: given observation → action
            # deterministic=True means always pick the best action (no exploration noise)
            # During training, noise is added to encourage trying new actions.
            # During evaluation (here), we remove noise for maximum performance.
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            # reward ≈ forward_velocity - 0.1 * |joint_torques|²
            # Higher reward = running faster with less energy waste
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

        print(f"Episode {episode + 1}: reward = {episode_reward:.1f}  ({step_count} steps)")

    env.close()
    print()
    print("Compare this reward to the random policy from 01_gym_intro.py —")
    print("the trained agent scores much higher because it learned to coordinate")
    print("all 6 joints simultaneously through thousands of simulated trials.")


if __name__ == "__main__":
    main()
