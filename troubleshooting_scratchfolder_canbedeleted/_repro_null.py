"""Reproduce the NULL pointer access error by stepping the env repeatedly."""
import gymnasium as gym
import numpy as np
import traceback

print("Creating Reacher-v5...")
env = gym.make("Reacher-v5", render_mode=None)
obs, _ = env.reset()
print(f"reset OK, obs.shape={obs.shape}")

# Try stepping with various action values to find what triggers NULL pointer
for action_vals in [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (2.0, 2.0), (5.0, 5.0)]:
    action = np.array(action_vals, dtype=np.float32)
    print(f"Testing action={action_vals}...")
    try:
        for step in range(20):
            obs, r, done, trunc, info = env.step(action)
            if done or trunc:
                obs, _ = env.reset()
                print(f"  Episode ended at step {step}, reset OK")
        print(f"  20 steps OK for action={action_vals}")
    except Exception as e:
        print(f"  FAIL at step {step}: {type(e).__name__}: {e}")
        traceback.print_exc()
        obs, _ = env.reset()

print("Checking action space bounds:")
print("  low:", env.action_space.low)
print("  high:", env.action_space.high)
env.close()
print("DONE")
