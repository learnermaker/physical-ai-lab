"""Test the new _make_status_frame renderer and save sample frames."""
import sys, base64, numpy as np
sys.path.insert(0, r'c:\Users\JIMSEELAN\Desktop\Projects\physical-ai\physical-ai-workshop')

from api.server import _make_status_frame

out_dir = r'c:\Users\JIMSEELAN\Desktop\Projects\physical-ai\physical-ai-workshop\troubleshooting_scratchfolder_canbedeleted'

for scenario, obs_vals, action_vals in [
    ("idle",    [1.0, 0.5, 0.0, 0.87, -0.15, 0.1, 0.0, 0.0, 0.09, 0.09],  [0.0, 0.0]),
    ("action",  [0.7, -0.3, 0.71, 0.95, 0.12, -0.08, 1.2, -0.8, 0.15, -0.05], [0.8, -0.6]),
]:
    obs = np.array(obs_vals, dtype=np.float32)
    act = np.array(action_vals, dtype=np.float32)
    b64 = _make_status_frame(obs, act, 42)
    jpg = base64.b64decode(b64)
    path = f'{out_dir}\\frame_{scenario}.jpg'
    with open(path, 'wb') as f:
        f.write(jpg)
    print(f'Saved {path}  ({len(jpg):,} bytes)')

print('DONE')
