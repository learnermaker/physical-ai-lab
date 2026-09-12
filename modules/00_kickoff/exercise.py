#!/usr/bin/env python3
"""
Module 0 Exercise: Build Your First State Vector
Expected output: A printed numpy array of 4 float32 values representing
                 the physical state of a cart-pole system.
"""

# --- Standard imports (all provided, no TODOs here) ---
import numpy as np

# --- Background: what is a state vector? ---
#
# A state vector is a numeric snapshot of a physical system at one instant.
# For a cart-pole (a pole balanced on a moving cart) the four quantities we
# track are:
#
#   cart_position   — how far the cart is from the centre (metres)
#   cart_velocity   — how fast the cart is moving (m/s)
#   pole_angle      — how far the pole has tilted from vertical (radians)
#   pole_velocity   — how fast the pole tip is rotating (rad/s)
#
# Everything the agent "sees" at each timestep is exactly this array.
# Your job: construct that array.


def main():
    # Some realistic initial values to use as a starting point.
    # Feel free to change them once you have the exercise working.
    cart_position = 0.0   # metres — cart starts at centre
    cart_velocity = 0.0   # m/s    — cart is stationary
    pole_angle    = 0.05  # rad    — pole is very slightly tilted
    pole_velocity = 0.0   # rad/s  — pole is not rotating yet

    # TODO START — Construct a 4-element float32 numpy array called
    # `state_vector` using the four variables above, then print it.
    raise NotImplementedError(
        "Implement main() at line 37. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END
    # SOLUTION HINT: Use np.array([...], dtype=np.float32) with the four
    # variables in the order [cart_position, cart_velocity, pole_angle,
    # pole_velocity], then pass the result to print().

    # EXPECTED OUTPUT: state_vector
    # [0.   0.   0.05 0.  ]


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main():
#     cart_position = 0.0
#     cart_velocity = 0.0
#     pole_angle    = 0.05
#     pole_velocity = 0.0
#
#     state_vector = np.array(
#         [cart_position, cart_velocity, pole_angle, pole_velocity],
#         dtype=np.float32,
#     )
#     print(state_vector)
#
#
# if __name__ == "__main__":
#     main()
