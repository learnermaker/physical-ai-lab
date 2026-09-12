"""Check Reacher-v5 obs layout and link geometry for PIL renderer."""
import gymnasium as gym
import numpy as np
import math

env = gym.make('Reacher-v5', render_mode=None)
obs, _ = env.reset()
print('obs shape:', obs.shape)
print('obs values:', obs)
print('obs space low:', env.observation_space.low)
print('obs space high:', env.observation_space.high)

# Take a few steps and print obs
for i in range(3):
    action = np.array([0.5, -0.3])
    obs, r, d, t, info = env.step(action)
    cos1, sin1, cos2, sin2 = obs[0], obs[2], obs[1], obs[3]
    theta1 = math.atan2(sin1, cos1)
    theta2 = math.atan2(sin2, cos2)
    target_x, target_y = obs[4], obs[5]
    finger_x, finger_y = obs[8], obs[9]
    
    # Forward kinematics (link length 0.1 each)
    L = 0.1
    elbow_x = L * math.cos(theta1)
    elbow_y = L * math.sin(theta1)
    fk_x = elbow_x + L * math.cos(theta1 + theta2)
    fk_y = elbow_y + L * math.sin(theta1 + theta2)
    print(f'step {i}: theta1={math.degrees(theta1):.1f}° theta2={math.degrees(theta2):.1f}°')
    print(f'  FK finger=({fk_x:.3f},{fk_y:.3f})  obs finger=({finger_x:.3f},{finger_y:.3f})')
    print(f'  target=({target_x:.3f},{target_y:.3f})')

env.close()
print('DONE')
