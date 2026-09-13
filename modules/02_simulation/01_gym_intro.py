#!/usr/bin/env python3
"""
Module 2 Demo 1: Gymnasium Introduction
=========================================
Gymnasium is a Python library that wraps physics simulations behind a
standard interface.  Every environment — from CartPole to a robotic arm —
exposes the same three methods:

    env.reset()          → get the first observation of an episode
    env.step(action)     → apply an action, get back (obs, reward, done, ...)
    env.close()          → release resources when finished

This script creates CartPole-v1, prints its observation and action spaces,
then runs 200 random-action steps.

What you will see:
  - The observation space and action space printed to the terminal
  - CartPole balancing (or failing!) with random actions
  - Automatic resets when the pole falls

How it connects to the pipeline:
  This is the simulation layer — the "World/Sim" box in the pipeline.
  Next: 02_mujoco_reacher.py shows a robot arm simulation.

Run from the repo root:
  python modules/02_simulation/01_gym_intro.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env  # handles OpenGL fallback automatically


def main() -> None:
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # You can't train or test a robot policy on real hardware first —
    # it's too slow, too expensive, and too dangerous. Gymnasium wraps
    # physics simulations behind a standard interface so every algorithm
    # works identically whether you're running CartPole or a full humanoid.
    # The env.step(action) → (obs, reward, done) pattern you learn here is
    # used in every RL framework on every robot in the world.
    # ─────────────────────────────────────────────────────────────────────────
    env = make_env("CartPole-v1")

    # ── Inspect the spaces ───────────────────────────────────────────────────
    # The observation space describes the format of what the agent *sees*.
    # For CartPole it is a Box (continuous range) of 4 float values:
    #   obs[0] = cart position   (metres)
    #   obs[1] = cart velocity   (metres/second)
    #   obs[2] = pole angle      (radians from vertical)
    #   obs[3] = pole ang. vel.  (radians/second)
    print(f"observation_space: {env.observation_space}")
    # → Box([-4.8 -inf -0.418 -inf], [4.8 inf 0.418 inf], (4,), float32)

    # The action space describes what the agent *can do*.
    # For CartPole it is Discrete(2): action 0 = push left, action 1 = push right.
    print(f"action_space:      {env.action_space}")
    # → Discrete(2)

    # ── Run 200 random-action steps ──────────────────────────────────────────
    obs, info = env.reset()
    # obs is now the initial 4-element state vector, e.g. [-0.02, 0.01, 0.03, -0.01]
    # info is a dict of extra diagnostic info (usually empty for CartPole)

    for step in range(200):
        # env.action_space.sample() picks a random valid action.
        # For CartPole this is either 0 (left) or 1 (right) with equal probability.
        action = env.action_space.sample()

        # env.step() applies the action to the simulation for one timestep.
        # Returns a 5-tuple:
        #   obs        — new observation after the action
        #   reward     — +1.0 if the pole is still upright, 0 otherwise
        #   terminated — True if episode ended naturally (pole fell)
        #   truncated  — True if episode hit the step limit (500 for CartPole)
        #   info       — extra diagnostic dict (ignore for now)
        obs, reward, terminated, truncated, info = env.step(action)

        # env.render() draws the current frame.
        # This is a no-op when render_mode is None (headless fallback).
        env.render()

        # An episode ends when terminated OR truncated.
        # After that we must reset before calling step() again.
        if terminated or truncated:
            obs, info = env.reset()
            # obs is now the start of a fresh episode

    # Always close the environment when finished to free memory and GPU resources.
    env.close()


if __name__ == "__main__":
    main()
