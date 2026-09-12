#!/usr/bin/env python3
"""
Module 2 Exercise: Explore the Reacher-v5 State Space
Expected output: Two numpy arrays showing the lower and upper bounds of the observation space.
"""

import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/02_simulation/
sys.path.insert(0, str(REPO_ROOT))

from utils.gym_utils import make_env  # noqa: E402


def main() -> None:
    env = make_env("Reacher-v5", render_mode="rgb_array")
    obs, info = env.reset()

    print("Observation space:", env.observation_space)
    print("One sample observation:", obs)
    print()

    # TODO START — Print the lower and upper bounds of the observation space
    raise NotImplementedError(
        "Implement main() at line 25. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END
    # SOLUTION HINT: Access the low and high attributes on env.observation_space and print
    # each one with a descriptive label such as "Lower bounds:" and "Upper bounds:".

    # EXPECTED OUTPUT: obs_bounds
    # Lower bounds: [array of 11 floats]
    # Upper bounds: [array of 11 floats]

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
