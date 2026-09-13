#!/usr/bin/env python3
"""
Module 2 Exercise: Explore the Reacher-v5 State Space
=======================================================
Your task: print the lower and upper bounds of the observation space
and explain what they tell you about how the environment works.

Start here:  Find the TODO block below (~line 27) and add two print statements.
             The solution is commented out at the bottom of this file.

Expected output (approximate):
  Observation space: Box(-inf, inf, (10,), float64)
  One sample observation: [-0.012  0.043 ...]
  Lower bounds: [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
  Upper bounds: [ inf  inf  inf  inf  inf  inf  inf  inf  inf  inf]

Why does this matter?
  All bounds are -inf to +inf, meaning the environment imposes NO clipping
  on its state values.  The learning algorithm must discover the real
  operating range from experience — this makes the policy more robust
  across different robot hardware where limits differ.

Run from the repo root:
  python modules/02_simulation/exercise.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env


def main() -> None:
    env = make_env("Reacher-v5", render_mode="rgb_array")
    obs, info = env.reset()

    # env.observation_space describes the *format* of observations.
    # .shape tells you how many values the state vector has.
    # .low and .high tell you the minimum and maximum possible values.
    print("Observation space:", env.observation_space)
    print("One sample observation:", obs)
    print()

    try:
        # ── TODO: print the bounds ───────────────────────────────────────────
        # env.observation_space.low  is a numpy array of lower bounds
        # env.observation_space.high is a numpy array of upper bounds
        # Both have the same shape as the observation: (10,)
        raise NotImplementedError(
            "Implement main() at line 25. "
            "See # EXPECTED OUTPUT comment below."
        )
        # ── TODO END ─────────────────────────────────────────────────────────
        # SOLUTION HINT: Access the low and high attributes on env.observation_space
        # and print each one with a label like "Lower bounds:" and "Upper bounds:".
        #
        # EXPECTED OUTPUT:
        # Lower bounds: [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
        # Upper bounds: [ inf  inf  inf  inf  inf  inf  inf  inf  inf  inf]
    finally:
        env.close()


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main() -> None:
#     env = make_env("Reacher-v5", render_mode="rgb_array")
#     obs, info = env.reset()
#
#     print("Observation space:", env.observation_space)
#     print("One sample observation:", obs)
#     print()
#
#     print("Lower bounds:", env.observation_space.low)
#     print("Upper bounds:", env.observation_space.high)
#
#     env.close()
