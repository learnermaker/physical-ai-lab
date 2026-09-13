# Common Issues — Physical AI Workshop

Quick-reference troubleshooting for facilitators. Each section gives the symptom, root cause, and a copy-pasteable fix.

---

## 1. ARM64 Windows — MuJoCo Wheel Missing

**Symptom:** `pip install mujoco` fails with `No matching distribution found for mujoco` on an ARM64 Windows machine (e.g. Surface Pro X, Snapdragon-based laptops).

**Cause:** MuJoCo does not publish a pre-built wheel for Windows ARM64. pip cannot fall back to a source build without a working MSVC ARM64 toolchain.

**Fix:** Direct the participant to use WSL2 (Ubuntu) instead of native Windows.

```bash
# 1. Enable WSL2 (run in PowerShell as Administrator, then reboot)
wsl --install

# 2. After reboot, open the Ubuntu terminal and clone the repo
git clone https://github.com/learnermaker/physical-ai-lab.git
cd physical-ai-lab

# 3. Run the Linux setup script
bash setup.sh
```

---

## 2. Windows 11 Camera Privacy Prompt

**Symptom:** The browser webcam feed (Module 1 / Module 4) stays black, or `getUserMedia` throws `NotAllowedError`, even though the participant clicks "Allow" in the browser prompt.

**Cause:** Windows 11 has a system-level camera privacy toggle that blocks all apps, overriding the browser's own permission.

**Fix:** Enable camera access in Windows Settings.

```
Settings → Privacy & security → Camera → toggle "Camera access" ON
```

```bash
# Alternatively, open the Settings page directly from a Run dialog (Win+R):
ms-settings:privacy-webcam
```

---

## 3. Jupyter Wrong Kernel Selected

**Symptom:** VS Code shows `Python 3.x.y` (system Python) in the kernel picker instead of `Physical AI Workshop`. Imports like `import mediapipe` or `import gymnasium` fail with `ModuleNotFoundError`.

**Cause:** VS Code defaulted to a different Python interpreter rather than the workshop `.venv`.

**Fix:** Select the correct kernel manually.

```
1. Open any .ipynb notebook in VS Code.
2. Click the kernel name in the top-right corner of the notebook editor.
3. Select "Python Environments..." → choose "Physical AI Workshop (physical-ai)".
```

If the kernel doesn't appear in the list, re-register it from the activated `.venv`:
```bat
.venv\Scripts\activate
python -m ipykernel install --user --name physical-ai --display-name "Physical AI Workshop"
```

---

## 4. HuggingFace Download Blocked on Corporate Network

**Symptom:** `huggingface_hub.utils._errors.EntryNotFoundError`, a connection timeout, or an HTTP 403 error when a module tries to download a model at runtime.

**Cause:** Corporate firewalls or proxy servers block `huggingface.co`. Some networks also intercept TLS, causing certificate errors.

**Fix:** Pre-download the required files on an unrestricted network and copy them into the repo's `models/` directory.

```bat
.venv\Scripts\activate
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
# Replace repo_id and filename with the actual model needed
hf_hub_download(repo_id='REPO_ID', filename='model.pt', local_dir='models/')
"
```

---

## 5. Windows SmartScreen Blocking `setup.bat`

**Symptom:** A blue dialog appears: "Windows protected your PC — Microsoft Defender SmartScreen prevented an unrecognised app from starting."

**Cause:** `setup.bat` is a newly downloaded file that has not yet accumulated enough reputation with Microsoft's SmartScreen service.

**Fix:** Bypass the SmartScreen warning.

```
1. In the SmartScreen dialog, click "More info".
2. Click "Run anyway".
```

```bash
# Alternative: remove the Mark-of-the-Web (Zone.Identifier) stream from PowerShell:
Unblock-File -Path setup.bat
.\setup.bat
```

---

## 6. Port 8000 Already in Use

**Symptom:** `python api/server.py` exits immediately with `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): only one usage of each socket address` (Windows) or similar.

**Cause:** Another process — a previous server instance, another development server, or an unrelated application — is already listening on port 8000.

**Fix:** Find and terminate the conflicting process.

```bash
# Find the PID of the process using port 8000
netstat -ano | findstr :8000

# Kill it (replace <PID> with the number from the last column above)
taskkill /PID <PID> /F

# Then start the server again
python api/server.py
```

---

## 7. `.env` File Missing or Incorrectly Formatted

**Symptom:** The Experience Hub shows "Gemini API key not configured", or Gemini endpoints return errors like `INVALID_ARGUMENT` or `API key not valid`.

**Cause:** The participant has not created a `.env` file, has named it `.env.txt`, or has formatted the key incorrectly (e.g. added extra spaces or quotes).

**Fix:** Create the `.env` file from the provided example.

```bash
# Copy the example file (Windows cmd)
copy .env.example .env

# Open the file and replace the placeholder with the real key
notepad .env
```

The file should contain exactly one line with no spaces or quotes:

```
GEMINI_API_KEY=AIzaSy...your_actual_key_here
```

---

## 8. Visual C++ Redistributables Missing

**Symptom:** `import mujoco` raises `OSError: [WinError 126] The specified module could not be found` or a similar DLL load error immediately after installation.

**Cause:** MuJoCo's Windows binaries link against the Microsoft Visual C++ 2015–2022 Redistributable runtime DLLs (`vcruntime140.dll`, etc.), which are not pre-installed on all Windows configurations.

**Fix:** Install the Visual C++ Redistributables.

```bash
# Download and run the installer directly from Microsoft (x64):
winget install Microsoft.VCRedist.2015+.x64

# Or open in browser if winget is unavailable:
start https://aka.ms/vs/17/release/vc_redist.x64.exe
```

---

## 9. pip Install Taking 10+ Minutes

**Symptom:** `pip install -r requirements.txt` has been running for more than 10 minutes with no apparent progress.

**Cause:** pip's resolver explores version combinations exhaustively for large dependency graphs. MediaPipe pins many transitive dependencies which slows this down.

**Fix:** Upgrade pip first, then retry.

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Gemini Free-Tier Rate Limit Hit During Demo

**Symptom:** The Experience Hub returns an error banner or the browser console shows `429 Too Many Requests` when calling the Gemini endpoints during the facilitator's live demo.

**Cause:** The Gemini free tier enforces a request-per-minute quota. Rapid repeated calls during a demo can exhaust it.

**Fix:** No action needed — the server automatically falls back to cached responses when a 429 is received. Continue the demo as normal; participants will see valid (pre-cached) responses.

```bash
# To verify the fallback is working, check the server terminal for this log line:
# "Gemini API error — returning cached response"

# If you want to clear the rate limit sooner, wait 60 seconds, then retry.
# Alternatively, use a second API key in .env and restart the server:
# GEMINI_API_KEY=AIzaSy...second_key
python api/server.py
```

---

## 11. mediapipe 1.0.0 `mp.solutions.drawing_utils` Not Available

**Symptom:** `AttributeError: module 'mediapipe' has no attribute 'solutions'` when running any module that imports `mp.solutions.drawing_utils` or `mp.solutions.hands`.

**Cause:** The `mediapipe.solutions` namespace was part of the legacy API retired by Google in March 2023. It is not present in `mediapipe==1.0.0`. The workshop uses the current Tasks API (`mp.tasks.vision.HandLandmarker`) and direct OpenCV drawing — this error only occurs if a participant has copy-pasted older tutorial code.

**Fix:** The workshop's own `exercise.py` files already use the current API and direct OpenCV drawing. Point the participant to the workshop code instead of the older snippet.

```bat
.venv\Scripts\activate
python -c "import mediapipe; print(mediapipe.__version__)"
```
# Expected output: 1.0.0

# The workshop drawing pattern to use instead of mp.solutions.drawing_utils:
python -c "
import cv2
import numpy as np
# Direct OpenCV drawing — no mp.solutions needed
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(frame, (320, 240), 5, (0, 255, 0), -1)
print('OpenCV drawing works correctly')
"
```

---

## 12. Simulation Stream Immediately Closes (OPEN → ERROR Loop)

**Symptom:** In the hub, Module 2 "Start Simulation" button causes the `EventSource` to cycle `OPEN → ERROR:0` repeatedly. No frames are ever displayed. The server log shows one of these sequences:

```
[make_env] render_mode='human' failed: exception: access violation reading 0x...
[make_env] Created 'Reacher-v5' with render_mode='rgb_array'
[simulation_stream] render() failed: gladLoadGL error — switching to status frame mode
```

**Cause:** On Windows machines without a functioning OpenGL driver (no GPU or virtualised display), `gym.make('Reacher-v5', render_mode='human')` succeeds but then crashes with an `OSError: access violation` at the first `env.reset()` call when GLFW tries to open a window. The older `_is_render_error` predicate did not recognise `"access violation"` or `mujoco.FatalError: gladLoadGL error` as render-related exceptions, so the fallback chain was never triggered and the async generator died silently — causing the SSE connection to close immediately.

**Fix (already applied in `utils/gym_utils.py` and `api/server.py`):**

1. `_RENDER_EXCEPTION_PATTERNS` in `gym_utils.py` now includes `"access violation"`, `"gladloadgl"`, `"fatalerror"`, and `"wgl"`.
2. `_is_render_error` now also catches `mujoco.FatalError` by checking the exception type name.
3. The `env.reset()` probe was moved inside the `try/except` in `make_env` so reset-time crashes are handled by the fallback chain.
4. `simulation_stream` wraps `env.render()` in its own `try/except`; on failure it permanently switches to `_make_status_frame` (PIL-generated JPEG with obs/action values as text).
5. The outer `event_generator` is wrapped in a broad `except Exception` that yields an error frame instead of silently closing the SSE stream.

**What participants see after the fix:** The simulation stream shows a dark frame with observation values and torque bars rendered as text — labelled "no-GPU mode". All simulation logic still runs; only the visual rendering differs.

```bash
# Verify the stream is working from the command line:
curl -N "http://localhost:8000/api/simulation/stream?action=0,0" | head -c 200
# Should start with: data: /9j/4AAQSkZ...  (base64 JPEG)
```

---

## 13. RL Training Chart Stays at "Starting training…" Forever

**Symptom:** Clicking "Start Training" in Module 3 posts to `/api/training/start` successfully (returns a `job_id`), but the status text never advances beyond "Starting training…" and the Chart.js canvas receives no data points.

**Cause:** The background training thread called `make_env("CartPole-v1", render_mode=None)`. On machines without OpenGL, `make_env` triggers its fallback probe — which calls `env.reset()` on the `"human"` mode environment. That reset call opens a GLFW window, which crashes with an access violation inside the daemon thread. The thread dies silently; the queue it was supposed to write to never receives any events; the SSE progress stream blocks on `queue.get(timeout=30)` indefinitely.

**Fix (already applied in `api/server.py`):** The `_run_training` function now calls `gym.make("CartPole-v1", render_mode=None)` directly (bypassing `make_env` and its GLFW probe entirely) since RL training never needs rendered frames. A `try/except` around `model.learn()` now also signals `{"done": True, "error": ...}` to the queue if training fails, preventing the SSE stream from blocking forever.

```bash
# Verify training starts and produces data (look for timestep events):
curl -s -X POST http://localhost:8000/api/training/start | python -c "import sys,json; d=json.load(sys.stdin); print('job_id:', d['job_id'])"
```

---

## 14. Gemini Model Deprecated (404 NOT_FOUND on API Calls)

**Symptom:** The server log shows repeated errors like:

```
[Gemini action error] 404 NOT_FOUND. {'error': {'code': 404, 'message':
  'This model models/gemini-2.5-flash is no longer available to new users...'}}
```

The hub still works because endpoints fall back to `cached_responses.json` automatically, but live Gemini API calls fail.

**Cause:** The model name `gemini-2.5-flash` (and subsequently `gemini-2.0-flash`) was deprecated and removed from the API. The error message from Google's API specifies the replacement.

**Fix (already applied in `api/server.py`):** Both Gemini endpoints (`/api/gemini/describe` and `/api/gemini/action`) now use `gemini-2.0-flash-lite`. If this model is also retired in future, update the two `model=` arguments in `api/server.py` to the model name specified in the 404 error message.

```bash
# Check which models are currently available with a valid API key:
python -c "
from google import genai
client = genai.Client(api_key='YOUR_KEY')
for m in client.models.list(): print(m.name)
"
```

---

*Physical AI Workshop — Common Issues — Jim Seelan*
