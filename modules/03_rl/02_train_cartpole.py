#!/usr/bin/env python3
"""
Module 3 Demo: Train PPO on CartPole-v1
=========================================
Trains a reinforcement learning agent from scratch using PPO
(Proximal Policy Optimisation) and plots the reward curve live.

PPO is a *policy gradient* algorithm — it directly learns a neural network
that maps observations to actions.  It updates the network in small, safe
steps to avoid catastrophically bad policy changes.

After training, the model is saved to models/ppo-CartPole-trained.zip
so you can reload it later without retraining.

What you will see:
  - Training progress logged to the terminal (mean reward per timestep)
  - A live reward chart updating every 1,000 steps
    (on headless machines the chart is saved to models/training_curve.png)
  - Final mean episode reward printed when training ends

How it connects to the pipeline:
  Training produces a *policy* — the agent's brain.  In Module 4 the policy
  is replaced by your hand: you ARE the controller.

Run from the repo root:
  python modules/03_rl/02_train_cartpole.py

Adapted from SB3 Callbacks docs:
  https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from stable_baselines3 import PPO           # the RL algorithm
from utils.gym_utils import make_env, LivePlotCallback

# Where to save the trained model — created if it doesn't exist
SAVE_PATH = REPO_ROOT / "models" / "ppo-CartPole-trained.zip"

# ── Create the training environment ──────────────────────────────────────────
# render_mode="rgb_array" avoids OpenGL/display issues on headless machines.
# We don't need to see the simulation during training — only the reward curve matters.
env = make_env("CartPole-v1", render_mode="rgb_array")

# ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────────
# This is the training loop that produces a deployable policy.
# The reward curve you're watching is the most important diagnostic in
# robot RL — it tells you whether training is working, when it plateaued,
# and whether you need more data. On a real robot training run, this curve
# represents real time, real energy, and real hardware wear. That's why
# sample efficiency matters: every flat region on the left is wasted cost.
# ─────────────────────────────────────────────────────────────────────────────

# ── Build the PPO model ───────────────────────────────────────────────────────
# "MlpPolicy" means the policy is a Multi-Layer Perceptron (a small neural network).
# For CartPole's 4-element observation this is a 2-hidden-layer network (64 units each).
# verbose=1 prints training progress to the terminal every 2,048 steps.
model = PPO("MlpPolicy", env, verbose=1)

# ── Set up the live reward chart ──────────────────────────────────────────────
# LivePlotCallback updates a matplotlib chart every check_freq timesteps.
# On machines without a display it saves to models/training_curve.png instead.
callback = LivePlotCallback(check_freq=1_000)

# ── Train for 50,000 timesteps ────────────────────────────────────────────────
# 50,000 steps ≈ a few minutes on a modern CPU.
# The reward starts low (~20-50) as the agent explores randomly, then rises
# sharply (~200-500) once it discovers the balancing strategy.
# That inflection point is the "aha moment" — watch for it in the chart.
model.learn(total_timesteps=50_000, callback=callback)

# ── Save the trained model ────────────────────────────────────────────────────
# PPO stores the policy network weights, optimiser state, and hyperparameters.
# SAC.load() / PPO.load() can reload this file later.
SAVE_PATH.parent.mkdir(exist_ok=True)   # create models/ if it doesn't exist
model.save(str(SAVE_PATH))

# ── Report final mean reward ──────────────────────────────────────────────────
# ep_info_buffer stores the last 100 completed episodes' rewards.
# Mean reward = how many steps the agent keeps the pole balanced on average.
# CartPole max = 500 steps — a well-trained agent should approach or hit this.
buf = model.ep_info_buffer
if buf:
    mean_reward = float(np.mean([ep["r"] for ep in buf]))
    print(f"\nFinal mean episode reward (last {len(buf)} episodes): {mean_reward:.2f}")
    print(f"CartPole maximum possible reward: 500")
    if mean_reward > 400:
        print("Excellent! The agent has learned to balance reliably.")
    elif mean_reward > 200:
        print("Good progress — try training longer (total_timesteps=100_000).")
    else:
        print("Still learning — try total_timesteps=100_000 to see improvement.")
else:
    print("Final mean episode reward: no episodes recorded in buffer")

env.close()
print(f"\nModel saved to {SAVE_PATH}")
print("Open modules/03_rl/exercise.py to experiment with training duration.")
