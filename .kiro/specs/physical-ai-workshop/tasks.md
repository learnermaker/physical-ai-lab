# Implementation Plan: Physical AI Workshop Repository

## Overview

Build a self-contained GitHub repository for a 3-hour Physical AI workshop supporting ~30 participants on Windows 11 x86-64 laptops. The implementation proceeds strictly by dependency order: repo scaffold → dependency files → setup/verify scripts → shared utilities → pre-bundled assets → modules (0–6) → tests → instructor materials → documentation.

All code is Python 3.12. Property-based tests use `hypothesis`; unit tests use `pytest`. No GPU or webcam is required — every hardware-dependent path has a transparent software fallback.

---

## Tasks

- [x] 1. Repository scaffold and root configuration files
  - [x] 1.1 Create root directory structure and static config files
    - Create the five required top-level directories: `modules/`, `utils/`, `models/`, `assets/`, `instructor/`
    - Add a `.gitkeep` to any directory that would otherwise be empty after cloning
    - Create `.gitignore` excluding `.env`, `__pycache__/`, `*.pyc`, `conda-meta/`, `*.egg-info/`, `.DS_Store`, `.vscode/`, `*.ipynb_checkpoints`
    - Create `.env.example` with `GEMINI_API_KEY=your_key_here` and a comment pointing to Google AI Studio
    - Create `LICENSE` (MIT) with author name and current year
    - _Requirements: 1.3, 1.5, 1.6, 1.7, 18.1_
    - _Design: Repository Root Layout_
    - **Acceptance:** `git status` shows all five directories tracked; `.gitignore` causes `.env` to appear as untracked; `LICENSE` contains "MIT"

  - [x] 1.2 Create `environment.yml` for conda environment specification
    - Define conda environment named `physical-ai` with `python=3.12` and no additional conda packages (pip-only extras)
    - Include a `pip:` section that mirrors `requirements.txt` so the file is usable with `conda env create`
    - _Requirements: 2.1_
    - _Design: § 2.6 requirements.txt_
    - **Acceptance:** `conda env create -f environment.yml --dry-run` completes without error

- [x] 2. Dependency manifest
  - [x] 2.1 Create `requirements.txt` with all pinned and unpinned dependencies
    - Pin: `numpy==2.2.4`, `mujoco>=3.1,<4.0` (latest compatible 3.x), `gymnasium[classic_control,mujoco]==1.3.0`, `stable-baselines3==2.9.0`, `opencv-python==4.13.0.92`, `mediapipe==1.0.0`
    - Unpin: `cvzone` (drawing utilities only), `google-genai`, `huggingface_hub`, `huggingface_sb3`, `Pillow`, `matplotlib`, `ipykernel`, `imageio[ffmpeg]`, `python-dotenv`
    - Add header comment explaining PyTorch must be installed separately before this file
    - _Requirements: 2.4_
    - _Design: § 2.6 requirements.txt_
    - **Acceptance:** File parses with `pip install --dry-run -r requirements.txt` (after torch pre-install) without resolver conflicts

- [x] 3. Setup scripts
  - [x] 3.1 Create `setup.bat` (Windows primary path)
    - Step 1: Check `conda` is on PATH; if absent print install instructions and `exit /b 1`
    - Step 2: `conda create -n physical-ai python=3.12 -y --no-default-packages` with 120-second polling timeout using `start /b` + loop + `taskkill`; on timeout print libmamba hint and `exit /b 1`
    - Step 3: `call conda activate physical-ai`
    - Step 4: `pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu`; on failure print internet message and `exit /b 1`
    - Step 5: `pip install -r requirements.txt`; on failure print internet message and `exit /b 1`
    - Step 6: Register ipykernel as `"Physical AI Workshop"`
    - Step 7: Run `python verify_install.py`; if non-zero exit print warning (not fatal)
    - Step 8: Print completion message
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8_
    - _Design: § 2.4 setup.bat_
    - **Acceptance:** Script is valid batch syntax (`cmd /c "setup.bat" /?` parses without error); each error path contains the exact message strings from requirements

  - [x] 3.2 Create `setup.sh` (Mac/Linux fallback)
    - Mirror `setup.bat` logic using POSIX shell: source conda init, `timeout 120 conda create`, same install sequence, kernel registration, verify invocation
    - Use `|| { echo "..."; exit 1; }` error handling throughout
    - _Requirements: 3.1_
    - _Design: § 2.5 setup.sh_
    - **Acceptance:** `bash -n setup.sh` reports no syntax errors; script is executable (`chmod +x`)

- [x] 4. Installation verification script
  - [x] 4.1 Create `verify_install.py`
    - Import-check loop over all 12 packages (numpy, mujoco, gymnasium, stable_baselines3, cv2, mediapipe, cvzone, google.genai, torch, PIL, matplotlib, huggingface_hub) — print `[PASS]` or `[FAIL]` + pip command per package
    - Webcam check (check 13/14): open index 0, show window ≤2 seconds or until `q`, print `[PASS] camera` or `[WARN] Camera unavailable — fallback video will be used`
    - CartPole check (check 14/14): `gym.make("CartPole-v1", render_mode="rgb_array")`, `step()`, `close()`; catch OpenGL exceptions as `[WARN]`, others as `[FAIL]`
    - Final summary: `Setup complete: N/14 checks passed` where N counts neither-fail-nor-warn results
    - `sys.exit(0)` if failed==0, else `sys.exit(1)`
    - Must complete within 30 seconds
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
    - _Design: § 2.3 verify_install.py_
    - **Acceptance:** Running with all packages installed exits 0 and prints exactly `Setup complete: 14/14 checks passed`; running with a missing package exits 1 and prints the pip install command for that package

- [x] 5. Shared utility — camera module
  - [x] 5.1 Create `utils/__init__.py` (empty) and `utils/camera.py` — `LoopingVideoCapture` class
    - Implement `LoopingVideoCapture` with `__init__(path: str)`, `read() → tuple[bool, np.ndarray | None]`, `isOpened() → bool`, `release() → None`
    - `read()` loop logic: call `_cap.read()`; on EOF seek to frame 0 and retry once; if retry fails return `(False, None)`
    - Expose no other `cv2.VideoCapture` methods
    - _Requirements: 7.3_
    - _Design: § 2.1 LoopingVideoCapture_
    - **Acceptance:** Instantiating with a valid 3-frame video and reading 10 times returns `(True, frame)` all 10 times; reading a non-existent file returns `(False, None)` on first `read()`

  - [x] 5.2 Add `open_camera` function to `utils/camera.py`
    - Implement decision tree: validate explicit `fallback_path` first (raise `ValueError` if not openable); try `cv2.VideoCapture(index)`; if not opened warn to stderr and try fallback; raise `RuntimeError` if fallback also fails; otherwise return `LoopingVideoCapture(fallback_path)`
    - Default `fallback_path` value: `None` — when `None`, the function resolves the path as `Path(__file__).resolve().parent.parent / 'assets' / 'fallback_hand_demo.mp4'` (repo-root-relative, always correct regardless of working directory)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
    - _Design: § 2.1 open_camera decision tree_
    - **Acceptance:** With webcam unavailable (mocked), returns a `LoopingVideoCapture`; with invalid explicit fallback_path raises `ValueError`; with missing default fallback raises `RuntimeError`

- [x] 6. Shared utility — gym_utils module
  - [x] 6.1 Create `utils/gym_utils.py` — `_is_render_error` helper and `make_env` function
    - Define `_RENDER_EXCEPTION_PATTERNS` tuple and `_is_render_error(exc)` predicate
    - Implement `make_env(env_id, render_mode="human")` with the three-mode fallback chain `["human", "rgb_array", None]`
    - Print progress/fallback messages to `stdout`; propagate non-rendering exceptions immediately after exactly one `gym.make` call
    - Raise `RuntimeError` if all three modes fail with rendering exceptions
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
    - _Design: § 2.2 make_env, fallback chain_
    - **Acceptance:** Mocking `gym.make` to raise `RuntimeError("opengl error")` on first two calls and succeed on third returns a valid env; mocking `gym.make` to raise `ValueError("unknown env")` results in that `ValueError` propagating after exactly one call

  - [x] 6.2 Add `LivePlotCallback` class to `utils/gym_utils.py`
    - Extend `stable_baselines3.common.callbacks.BaseCallback`
    - `__init__`: attempt `plt.ion()`; on failure switch to Agg backend and set `_file_mode = True`
    - `_on_step`: every `check_freq` steps update episode reward plot from `self.model.ep_info_buffer`; use `plt.pause(0.001)` for non-blocking update; in file mode save to `save_path`
    - `_on_training_end`: final save/show
    - Attribution comment: `# Adapted from SB3 Callbacks documentation: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)`
    - _Requirements: 12.3_
    - _Design: § 2.2 LivePlotCallback_
    - **Acceptance:** Instantiating and calling `_on_step` 1000 times without a display (Agg mode) does not raise; `save_path` file is created in file mode

- [x] 7. Pre-bundled assets
  - [x] 7.1 Download and commit pre-trained model weights
    - Use `huggingface_sb3.load_from_hub("sb3/sac-HalfCheetah-v5", ...)` to download `models/sac-HalfCheetah-v5.zip`
    - Use `huggingface_sb3.load_from_hub("sb3/ppo-CartPole-v1", ...)` to download `models/ppo-CartPole-v1.zip`
    - Write a one-off local helper script `download_models.py` (not committed to the repository � run it once locally on the repo maintainer's machine) to perform the download; commit only the resulting `.zip` files
    - After downloading, load and re-save each model in the target Python 3.12 + numpy 2.2.4 + SB3 2.9.0 environment to re-pickle with current numpy: `model = SAC.load(path); model.save(path)` (for HalfCheetah). `model = PPO.load(path); model.save(path)` (for CartPole). This prevents numpy 1.x → 2.x pickle incompatibility at participant load time.
    - Verify each file loads correctly and runs without warnings
    - _Requirements: 6.1, 6.2_
    - _Design: § 4.4 Pre-trained Model Files_
    - **Acceptance:** `SAC.load("models/sac-HalfCheetah-v5.zip")` succeeds without downloading anything; AND `SAC.load('models/sac-HalfCheetah-v5.zip')` and `PPO.load('models/ppo-CartPole-v1.zip')` both succeed without warnings in a Python 3.12 + numpy 2.2.4 environment

  - [x] 7.2 Create `assets/fallback_hand_demo.mp4`
    - Record or source a 30-second video of a hand performing open/close and pointing gestures suitable for Modules 1 and 4 landmark detection
    - Commit the file directly (well under 100 MB GitHub limit)
    - _Requirements: 6.3_
    - _Design: Fallback Strategy Summary_
    - **Acceptance:** `cv2.VideoCapture("assets/fallback_hand_demo.mp4").isOpened()` returns True; frame count is ≥ 750 frames (30 s at 25 fps)

  - [x] 7.3 Create `modules/05_foundation_models/cached_responses.json`
    - Build keyed JSON with ≥10 entries covering each distinct prompt type used in Module 5 (scene description, action suggestion, object description, gesture identification, joint angles, robot action from image)
    - Keys are normalised (lowercase, stripped, collapsed whitespace) prompt strings; values are response text or dicts with `"action"` and `"reason"` keys
    - Include the `load_cache(path)` helper in `modules/05_foundation_models/__init__.py` or as a standalone module-level function importable by both Module 5 scripts
    - _Requirements: 6.4, 6.5, 14.8_
    - _Design: § 2.8 cached_responses.json, load_cache helper_
    - **Acceptance:** `load_cache()` returns a dict with ≥10 keys; all action-type values contain the key `"action"`; `load_cache("nonexistent.json")` calls `sys.exit(1)`

  - [x] 7.4 Download and commit `assets/hand_landmarker.task`
    - Download the MediaPipe HandLandmarker model bundle: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
    - File is approximately 8 MB; commit directly to the repository (well within GitHub 100 MB file limit)
    - Add the download step to `setup.bat` and `setup.sh` as a fallback if the file is missing: `python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'assets/hand_landmarker.task') if not Path('assets/hand_landmarker.task').exists() else None"`
    - _Requirements: 10.1, 10.2, 13.1_
    - _Design: § 4.2 Landmark Data, Fix D6_
    - **Acceptance:** `Path("assets/hand_landmarker.task").exists()` is True; file is loadable as a MediaPipe HandLandmarker model without error

- [x] 8. Module 0 — Workshop kickoff notebook
  - [x] 8.1 Create `modules/00_kickoff/overview.ipynb`
    - Text cell: define the Physical AI pipeline `[Webcam] → [Perception Layer] → [State Vector] → [Sim Agent] → [Action]` with a one-sentence description of each stage
    - Code cell: `import numpy as np` and print a sample 4-element float32 state vector
    - All cells run to completion without errors under `physical-ai` kernel
    - _Requirements: 9.1, 9.2, 9.3_
    - _Design: § 2.7 Module Interfaces_
    - **Acceptance:** `jupyter nbconvert --to notebook --execute modules/00_kickoff/overview.ipynb` exits 0

  - [x] 8.2 Create `modules/00_kickoff/exercise.py`
    - Scaffold: all imports provided, one `# TODO START` / `# TODO END` block guiding participant to construct and print their own 4-element state vector
    - Include `# SOLUTION HINT:`, `# EXPECTED OUTPUT:`, `NotImplementedError` in unmodified form, and commented-out `# SOLUTION:` block at bottom
    - _Requirements: 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** `python modules/00_kickoff/exercise.py` raises `NotImplementedError` with function name and line number; uncommenting `# SOLUTION` block produces expected output

- [x] 9. Module 1 — Perception
  - [x] 9.1 Create `modules/01_perception/01_webcam_basics.py`
    - Add `sys.path` insert for repo root; import `open_camera` from `utils.camera`
    - Open camera with `open_camera()`; display frame in window; print frame shape to stdout on first frame only; exit on `q`
    - Attribution comment if applicable
    - _Requirements: 10.1, 10.5_
    - _Design: § 2.7, Python Path Convention_
    - **Acceptance:** Script runs without import error; with webcam mocked unavailable, fallback video opens silently

  - [x] 9.2 Create `modules/01_perception/02_hand_tracking.py`
    - Use `mediapipe.tasks.vision.HandLandmarker` (Tasks API, LIVE_STREAM mode) with `num_hands=2` to detect hands. Load model from `assets/hand_landmarker.task`. Draw landmarks using `mp.solutions.drawing_utils` or OpenCV.
    - Display in window; exit on `q`
    - Attribution comment: `# Uses MediaPipe Tasks API: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)`
    - _Requirements: 10.2_
    - _Design: § 4.2 Landmark Data_
    - **Acceptance:** Script detects hands using Tasks API; landmark coordinates are in [0,1] normalised range; drawing overlay visible on frame

  - [x] 9.3 Create `modules/01_perception/03_joint_angles.py`
    - Compute angles for thumb, index, and middle finger MCP joints using landmark positions
    - Print formatted state vector `[θ1, θ2, θ3]` to stdout once per second (time-gated with `time.time()`)
    - Attribution comment: `# Uses MediaPipe Tasks API: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)`
    - _Requirements: 10.3_
    - _Design: § 4.1 State Vector (Module 1)_
    - **Acceptance:** Script prints a line matching `[` followed by three floats `]` when hand is present in fallback video

  - [x] 9.4 Create `modules/01_perception/exercise.py`
    - Scaffold: all of `03_joint_angles.py` logic provided; single `# TODO START` / `# TODO END` block for adding ring finger angle to produce a 4-element state vector
    - Include `NotImplementedError`, `# SOLUTION HINT:`, `# EXPECTED OUTPUT:` (4-element vector), and commented `# SOLUTION:` block
    - _Requirements: 10.4, 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** Unmodified script raises `NotImplementedError`; solution block prints 4-element vector

- [x] 10. Module 2 — Simulation
  - [x] 10.1 Create `modules/02_simulation/01_gym_intro.py`
    - Import `make_env` from `utils.gym_utils`; create `CartPole-v1`; print `observation_space` and `action_space`; run 200 random-action steps with render; close environment
    - _Requirements: 11.1, 11.5_
    - _Design: § 2.2 make_env_
    - **Acceptance:** Script completes 200 steps without raising; render fallback activates silently if OpenGL unavailable

  - [x] 10.2 Create `modules/02_simulation/02_mujoco_reacher.py`
    - Import `make_env`; create `Reacher-v5`; print observation and action vector dimensionality; run 100 random-action steps
    - _Requirements: 11.2, 11.5_
    - _Design: § 4.1 State Vector (Reacher)_
    - **Acceptance:** Script prints two integers (obs dim, action dim) and completes 100 steps without error

  - [x] 10.3 Create `modules/02_simulation/03_pretrained_agent.py`
    - Load `models/sac-HalfCheetah-v5.zip` with `SAC.load`; create `HalfCheetah-v5`; run 3 complete episodes; print episode reward for each
    - This script is referenced in `PRE_WORKSHOP_SETUP.md` as a pre-session preview; the facilitator references results during the live session rather than running it live
    - _Requirements: 11.3_
    - _Design: § 4.4 Pre-trained Model Files_
    - **Acceptance:** Script prints exactly 3 lines each containing a float reward value; HalfCheetah agent visibly runs in simulation

  - [x] 10.4 Create `modules/02_simulation/exercise.py`
    - Scaffold: `Reacher-v5` environment created; single `# TODO START` / `# TODO END` block for printing `observation_space.low` and `observation_space.high` bounds
    - Include `NotImplementedError`, `# SOLUTION HINT:`, `# EXPECTED OUTPUT:`, and commented `# SOLUTION:` block
    - _Requirements: 11.4, 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** Unmodified raises `NotImplementedError`; solution prints two numpy arrays

- [x] 11. Module 3 — Reinforcement Learning
  - [x] 11.1 Create `modules/03_rl/01_mdp_concepts.ipynb`
    - Define MDP concepts (state, action, reward, transition) in text cells; code cells demonstrate each with `CartPole-v1`; notebook runs to completion without error
    - Attribution comment in code cells: `# Concepts based on: https://huggingface.co/learn/deep-rl-course (MIT, low-maintenance reference)`
    - _Requirements: 12.1_
    - _Design: § 2.7 Module Interfaces_
    - **Acceptance:** `jupyter nbconvert --to notebook --execute modules/03_rl/01_mdp_concepts.ipynb` exits 0

  - [x] 11.2 Create `modules/03_rl/02_train_cartpole.py`
    - Import `make_env` and `LivePlotCallback` from `utils.gym_utils`; train PPO on `CartPole-v1` for 50,000 timesteps; pass `LivePlotCallback(check_freq=1000)` to `learn()`
    - After training, save model to `models/ppo-CartPole-trained.zip` and print final mean episode reward
    - Attribution comment: `# Adapted from SB3 Callbacks documentation: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)`
    - _Requirements: 12.2, 12.3, 12.4_
    - _Design: § 2.2 LivePlotCallback_
    - **Acceptance:** Script completes and creates `models/ppo-CartPole-trained.zip`; final mean reward is printed

  - [x] 11.3 Create `modules/03_rl/exercise.py`
    - Scaffold: full training setup provided; single `# TODO START` / `# TODO END` block to change `total_timesteps` and observe reward curve effect
    - Include `NotImplementedError`, `# SOLUTION HINT:`, `# EXPECTED OUTPUT:`, and commented `# SOLUTION:` block
    - _Requirements: 12.5, 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** Unmodified raises `NotImplementedError`; solution trains for the modified timestep count

- [x] 12. Module 4 — Perception to Action
  - [x] 12.1 Create `modules/04_perception_to_action/01_hand_to_reacher.py`
    - Import `open_camera` from `utils.camera` and `make_env` from `utils.gym_utils`
    - Open camera (fall back to `assets/fallback_hand_demo.mp4`); if neither source opens, print error and `sys.exit(1)`
    - Detect hand with `HandLandmarker` (Tasks API). Map index finger tip (landmark 8) normalised position — coordinates are already in [0,1] range from the Tasks API, no pixel division needed — to action via `scale_landmark_to_action`.
    - When no hand detected for >1 s, send zero-vector action and display `"No hand detected — holding position"`
    - When render available, show camera frame with overlaid landmarks plus sim render; when render unavailable, show camera frame plus numeric joint torque commands
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
    - _Design: § 4.2 Landmark normalisation, § 4.1 State Vector (Module 4)_
    - **Acceptance:** `scale_landmark_to_action(0.0)` returns `-1.0`; `scale_landmark_to_action(1.0)` returns `1.0`; `scale_landmark_to_action(0.5)` returns `0.0`; AND `scale_landmark_to_action` is defined at module scope (not inside `main()`) so that `from modules.04_perception_to_action import scale_landmark_to_action` or direct import via importlib works for property testing

  - [x] 12.2 Create `modules/04_perception_to_action/exercise.py`
    - Scaffold: full Module 4 pipeline provided; single `# TODO START` / `# TODO END` block to map a second landmark's normalised position to a third Reacher joint torque command
    - Include `NotImplementedError`, `# SOLUTION HINT:`, `# EXPECTED OUTPUT:`, and commented `# SOLUTION:` block
    - _Requirements: 13.6, 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** Unmodified raises `NotImplementedError`; solution sends a 3-element action to the environment

- [x] 13. Module 5 — Foundation Models
  - [x] 13.1 Create `modules/05_foundation_models/01_gemini_vision.py`
    - Load `GEMINI_API_KEY` from `.env` via `python-dotenv`; initialise `google.genai` client
    - If key absent or empty: print warning to stdout, select random entry from `cached_responses.json` without calling API
    - Otherwise: open camera, capture frame, send to `gemini-2.5-flash` with scene description prompt, print response
    - Attribution comment: `# Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)`
    - _Requirements: 14.1, 14.2_
    - _Design: § 4.3 Gemini API Request/Response, § 2.8 load_cache_
    - **Acceptance:** With no `.env` file, script prints a warning and a cached response without raising

  - [x] 13.2 Create `modules/05_foundation_models/02_gemini_robot_brain.py`
    - Capture frame; encode as base64 JPEG; print `"Calling Gemini..."` before request; send to `gemini-2.5-flash` with structured action prompt
    - Strip markdown fences from response before `json.loads`; print top-level keys and values
    - On any exception or JSON parse error: log to stderr, select random entry from cache
    - If cache missing or empty: log to stderr and `sys.exit(1)`
    - Attribution comment: `# Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)`
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8_
    - _Design: § 4.3 JSON fence stripping, § 2.8 load_cache, Error Handling — Gemini_
    - **Acceptance:** With API raising `ConnectionError`, script logs to stderr and prints a cached action dict; output includes `'Calling Gemini...'` before the API call and either `'Done.'` (on success) or `'[Fallback] Using cached response.'` (on failure) afterward; AND output always contains `'action'` key in the final result

  - [x] 13.3 Create `modules/05_foundation_models/exercise.py`
    - Scaffold: full `02_gemini_robot_brain.py` logic provided; two `# TODO START` / `# TODO END` blocks — one to modify the prompt string, one to observe/print the model output
    - Include `NotImplementedError`, `# SOLUTION HINT:`, `# EXPECTED OUTPUT:`, and commented `# SOLUTION:` block
    - _Requirements: 14.7, 16.1, 16.2, 16.3, 16.4_
    - _Design: § 4.5 Exercise Scaffolding Schema_
    - **Acceptance:** Unmodified raises `NotImplementedError`; solution prints action dict from either live API or cache

- [x] 14. Module 6 — Wrap-up concepts map
  - [x] 14.1 Create `modules/06_wrapup/concepts_map.ipynb`
    - Text cell mapping each module (1–5) to its Physical AI pipeline concept
    - Text cell covering sim-to-real transfer, digital twins, and ≥2 suggested next steps (e.g., Isaac Lab, LeRobot)
    - All cells run to completion without errors under `physical-ai` kernel
    - _Requirements: 15.1, 15.2, 15.3_
    - _Design: § 2.7 Module Interfaces_
    - **Acceptance:** `jupyter nbconvert --to notebook --execute modules/06_wrapup/concepts_map.ipynb` exits 0

- [x] 15. Test suite — unit tests
  - [x] 15.1 Create `tests/__init__.py` and `tests/test_camera.py`
    - `test_open_camera_returns_valid_cap` — mock `cv2.VideoCapture` `isOpened()=True`, verify return type is `cv2.VideoCapture`
    - `test_open_camera_fallback_on_invalid_index` — mock `isOpened()=False` for webcam, verify `LoopingVideoCapture` returned
    - `test_open_camera_raises_runtime_if_fallback_missing` — mock both as failing, verify `RuntimeError`
    - `test_open_camera_raises_value_error_for_explicit_bad_path` — pass non-existent path as `fallback_path`, verify `ValueError`
    - `test_looping_video_capture_resets_on_eof` — use a 3-frame synthetic video fixture, read 6 times, verify all `(True, frame)`
    - _Requirements: 7.1–7.5_
    - _Design: Testing Strategy — test_camera.py_
    - **Acceptance:** `pytest tests/test_camera.py -v` passes all 5 tests

  - [x]* 15.2 Write unit tests for `utils/gym_utils.py` in `tests/test_gym_utils.py`
    - `test_make_env_human_mode_succeeds`
    - `test_make_env_falls_back_to_rgb_array`
    - `test_make_env_falls_back_to_none`
    - `test_make_env_raises_runtime_when_all_fail`
    - `test_make_env_propagates_non_render_exception` — verify `gym.make` called exactly once
    - _Requirements: 8.1–8.7_
    - _Design: Testing Strategy — test_gym_utils.py_
    - **Acceptance:** `pytest tests/test_gym_utils.py -v` passes all 5 tests

  - [x]* 15.3 Write unit tests for `verify_install.py` in `tests/test_verify_install.py`
    - `test_summary_line_format` — run verify script in subprocess with all imports mocked, verify output matches regex `Setup complete: \d+/14 checks passed`
    - _Requirements: 5.5_
    - _Design: Testing Strategy — test_verify_install.py_
    - **Acceptance:** `pytest tests/test_verify_install.py -v` passes

- [x] 16. Test suite — property-based tests
  - [x] 16.1 Create `tests/test_properties.py` with shared fixtures
    - Create `tmp_short_video` pytest fixture: generate a 5-frame synthetic BGR video using `cv2.VideoWriter` into a tmp path; yield the path; clean up after test
    - Create `tmp_cache_file` pytest fixture: write a minimal `cached_responses.json` with 3 entries including one with `"action"` key; yield the path
    - Add `count_passes(passes, fails)` helper that simulates the verify_install counting logic in isolation
    - _Requirements: 5.5, 7.1–7.5, 8.2–8.7, 13.2, 14.5_
    - _Design: Testing Strategy — Property-Based Tests_
    - **Acceptance:** `pytest tests/test_properties.py --collect-only` shows all 8 property test functions

  - [x]* 16.2 Write Property 1 & 2 test — `test_looping_capture_never_exhausts`
    - `@given(n_reads=st.integers(min_value=1, max_value=500))` `@settings(max_examples=100)`
    - For any n_reads, `LoopingVideoCapture(tmp_short_video).read()` returns `(True, frame)` all n_reads times
    - **Property 1 & 2: fallback camera continuity**
    - **Validates: Requirements 7.1, 7.2, 7.3**
    - _Design: § Correctness Properties 1 & 2_
    - **Acceptance:** `pytest tests/test_properties.py::test_looping_capture_never_exhausts -v` passes with ≥100 examples

  - [x]* 16.3 Write Property 3 test — `test_open_camera_value_error_for_bad_path`
    - `@given(bad_path=st.text(min_size=1).filter(lambda s: not Path(s).exists()))` `@settings(max_examples=100)`
    - `open_camera(index=99, fallback_path=bad_path)` raises `ValueError` containing the path
    - **Property 3: ValueError for invalid explicit fallback_path**
    - **Validates: Requirements 7.4**
    - _Design: § Correctness Properties 3_
    - **Acceptance:** `pytest tests/test_properties.py::test_open_camera_value_error_for_bad_path -v` passes

  - [x]* 16.4 Write Property 4 test — `test_make_env_fallback_chain_succeeds`
    - `@given(fail_count=st.integers(min_value=1, max_value=2))` `@settings(max_examples=100)`
    - Mock `gym.make` to raise `RuntimeError("opengl error: display not found")` for first `fail_count` calls, then succeed; verify env returned and `gym.make` called `fail_count + 1` times
    - **Property 4: make_env exhausts modes before raising**
    - **Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.6**
    - _Design: § Correctness Properties 4_
    - **Acceptance:** `pytest tests/test_properties.py::test_make_env_fallback_chain_succeeds -v` passes

  - [x]* 16.5 Write Property 5 test — `test_non_render_exception_propagates_once`
    - `@given(msg=st.text(min_size=1).filter(lambda s: not any(p in s.lower() for p in _RENDER_EXCEPTION_PATTERNS)))` `@settings(max_examples=100)`
    - Mock `gym.make` to raise `ValueError(msg)`; verify `ValueError` propagates and `gym.make` called exactly once
    - **Property 5: non-rendering exceptions propagate without retry**
    - **Validates: Requirements 8.7**
    - _Design: § Correctness Properties 5_
    - **Acceptance:** `pytest tests/test_properties.py::test_non_render_exception_propagates_once -v` passes

  - [x]* 16.6 Write Property 6 test — `test_scale_landmark_linear_map`
    - `@given(x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))` `@settings(max_examples=100)`
    - `scale_landmark_to_action(x)` equals `2.0 * x - 1.0` within 1e-9
    - **Property 6: scale_landmark_to_action is a correct linear map**
    - **Validates: Requirements 13.2**
    - _Design: § Correctness Properties 6_
    - **Acceptance:** `pytest tests/test_properties.py::test_scale_landmark_linear_map -v` passes

  - [x]* 16.7 Write Property 7 test — `test_verify_summary_count_matches`
    - `@given(passes=st.integers(0, 14), fails=st.integers(0, 14))` with `assume(passes + fails == 14)` `@settings(max_examples=100)`
    - `count_passes(passes, fails)` (isolated helper) equals `passes`
    - **Property 7: verify_install summary count matches actual pass count**
    - **Validates: Requirements 5.5**
    - _Design: § Correctness Properties 7_
    - **Acceptance:** `pytest tests/test_properties.py::test_verify_summary_count_matches -v` passes

  - [x]* 16.8 Write Property 8 test — `test_gemini_fallback_always_parseable`
    - `@given(exc=st.sampled_from([Exception("network timeout"), ConnectionError("connection refused"), json.JSONDecodeError("Expecting value", "", 0), RuntimeError("quota exceeded")]))` `@settings(max_examples=100)`
    - Mock `google.genai` call to raise `exc`; call `get_robot_action(cache_path=tmp_cache_file)`; verify result is dict with `"action"` key
    - **Property 8: Gemini fallback always returns parseable action dict**
    - **Validates: Requirements 14.5**
    - _Design: § Correctness Properties 8_
    - **Acceptance:** `pytest tests/test_properties.py::test_gemini_fallback_always_parseable -v` passes

- [x] 17. Checkpoint — all tests pass
  - Ensure all non-optional tests pass: `pytest tests/ -v --ignore=tests/test_properties.py` (property tests require hypothesis dev install)
  - Verify `python verify_install.py` exits 0 in the `physical-ai` conda environment
  - Ask the user if any questions arise before proceeding.

- [x] 18. Instructor support materials
  - [x] 18.1 Create `instructor/FACILITATOR_GUIDE.md`
    - Time-boxed agenda: Pre-session (30 min), Module 0 (15 min), Module 1 (30 min), Module 2 (20 min — concepts + random env steps only; pretrained demo is pre-session), Module 3 (30 min), Module 4 (40 min), Module 5 (25 min), Wrap-up (15 min), Contingency buffer (15 min)
    - Facilitation notes for each segment including key talking points and transition cues
    - _Requirements: 17.1_
    - _Design: instructor/ directory_
    - **Acceptance:** File exists and contains a section for each of the 9 agenda segments (including contingency buffer) with estimated durations

  - [x] 18.2 Create `instructor/COMMON_ISSUES.md`
    - Document and provide copy-pasteable terminal commands for all failure modes listed in Requirement 17.2: ARM64 Windows MuJoCo wheel missing, Windows 11 camera privacy prompt, conda solver hang, PyTorch CUDA build accidentally installed, Jupyter wrong kernel selected, HuggingFace download blocked on corporate network, Windows SmartScreen blocking `setup.bat`, port 8000 already in use, `.env` file missing or incorrectly formatted, Visual C++ Redistributables missing, pip install taking 10+ minutes, Gemini free-tier rate limit hit during demo, mediapipe 1.0.0 `mp.solutions.drawing_utils` not available.
    - Each issue includes: symptom, cause, fix command(s)
    - _Requirements: 17.2, 17.3_
    - _Design: instructor/ directory_
    - **Acceptance:** File contains exactly 13 documented failure modes each with a symptom description, root cause, and at least one fenced code block containing a copy-pasteable fix command

- [x] 19. Root documentation
  - [x] 19.1 Create `README.md`
    - Workshop narrative introduction
    - Prerequisites section (conda, VS Code, pre-session setup link)
    - Quick-start command sequences:
      - Pre-workshop setup: clone → `setup.bat` → `verify_install.py`
      - Workshop primary workflow: `python api/server.py` → open `http://localhost:8000` in Chrome/Edge
      - Exercise editing: open `exercise.py` files in VS Code alongside the browser
    - Module listing in order with one-line description each, linking to their directory
    - **Acknowledgements** section listing all 4 sources: CVZone drawing utilities (MIT), SB3 Callbacks documentation pattern (MIT), HF deep-rl-class conceptual reference (MIT, low-maintenance), Google Gemini cookbook image pattern (Apache-2.0)
    - Note that `setup.sh` is a best-effort fallback validated on Windows 11 only
    - _Requirements: 1.1, 3.2, 18.2_
    - _Design: Repository Root Layout_
    - **Acceptance:** File contains both `api/server.py` start command and `http://localhost:8000` URL in the quick-start section; contains `## Acknowledgements` section with all 5 sources; links to all 6 module directories are present

  - [x] 19.2 Create `PRE_WORKSHOP_SETUP.md`
    - Step-by-step pre-session instructions: install conda, clone repo, run `setup.bat`, run `verify_install.py`, obtain Gemini API key from Google AI Studio, create `.env` file, and start the hub server with `python api/server.py` and verify it opens at `http://localhost:8000` in Chrome/Edge
    - Include a step instructing participants to run `python modules/02_simulation/03_pretrained_agent.py` to watch the HalfCheetah agent before the session — this is the live preview that motivates the RL module
    - Clearly labelled ARM64 Windows warning section with WSL2 setup instructions
    - _Requirements: 1.2, 4.1, 4.2_
    - _Design: PRE_WORKSHOP_SETUP.md_
    - **Acceptance:** File contains a section with "ARM64" in the heading; contains a link to Google AI Studio; contains WSL2 setup steps; contains the `03_pretrained_agent.py` run instruction

- [x] 20. Final checkpoint — end-to-end verification
  - Run `python verify_install.py` in the `physical-ai` env: must exit 0
  - Run `jupyter nbconvert --to notebook --execute` on all three notebooks (00_kickoff, 03_rl/01_mdp_concepts, 06_wrapup/concepts_map): all must exit 0
  - Run each module demonstration script with camera/rendering mocked to confirm no unhandled exceptions
  - Confirm `models/ppo-CartPole-v1.zip` and `models/sac-HalfCheetah-v5.zip` load with `PPO.load` and `SAC.load` respectively
  - Ask the user if any questions arise before concluding.

- [x] 21. Backend API server (`api/server.py`)
  - [x] 21.1 Create `api/__init__.py` (empty) and `api/server.py` — FastAPI app skeleton
    - Import FastAPI, uvicorn, StaticFiles; mount `frontend/` at `/`; add startup message
    - Add `GET /api/config` endpoint: load `.env`, return `{"gemini_key_configured": bool}`
    - Add `if __name__ == "__main__": uvicorn.run(...)` entry point with error handling for port binding failure
    - Insert repo root into `sys.path` at startup: `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` — this enables `from utils.gym_utils import make_env` and `from modules.foundation_models import load_cache` to resolve correctly regardless of working directory
    - Update `requirements.txt` to add `fastapi` and `uvicorn[standard]`
    - Add `SecurityHeadersMiddleware` to set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers on all responses — required for MediaPipe WASM `SharedArrayBuffer` support in Chrome/Edge
    - _Requirements: 19.1, 19.2, 19.3, 19.10_
    - _Design: Experience Hub — api/server.py, Startup sequence_
    - **Acceptance:** `python api/server.py` starts without error; `curl http://localhost:8000/api/config` returns valid JSON; AND `curl -I http://localhost:8000` response headers include `Cross-Origin-Opener-Policy: same-origin`

  - [x] 21.2 Add simulation stream endpoint to `api/server.py`
    - Implement `GET /api/simulation/stream` SSE endpoint
    - Parse `action` query param as comma-separated floats (default `0.0,0.0`)
    - Create `Reacher-v5` env via `make_env`; step with parsed action; render as rgb_array; JPEG encode (quality 60); base64 encode; yield as SSE data event
    - Handle `asyncio.CancelledError` for clean disconnect; close env on disconnect
    - _Requirements: 19.4_
    - _Design: Simulation stream design_
    - **Acceptance:** `curl -N "http://localhost:8000/api/simulation/stream?action=0,0"` streams base64 data events; connection closes cleanly on ctrl-C

  - [x] 21.3 Add training start and progress endpoints to `api/server.py`
    - Implement `POST /api/training/start`: spawn background thread running PPO CartPole 50k steps with `StreamingCallback` posting to a `queue.Queue`; store queue in `_jobs` dict keyed by UUID; return `{"job_id": uuid}`
    - Implement `GET /api/training/progress/{job_id}` SSE: read from job queue, stream `{"timestep": int, "mean_reward": float}` events; stream `{"done": true, "model_path": "..."}` on completion
    - Define `StreamingCallback(BaseCallback)` in `api/server.py` or `utils/gym_utils.py`: accepts a `job_queue: queue.Queue` in `__init__`, puts `{'timestep': int, 'mean_reward': float}` events in `_on_step` every `check_freq=1000` steps, puts `{'done': True, 'model_path': str}` in `_on_training_end` and saves the model. The SSE endpoint reads with `job_queue.get(timeout=30)` and yields heartbeat events on timeout to keep the connection alive.
    - _Requirements: 19.5, 19.6_
    - _Design: Training stream design, StreamingCallback_
    - **Acceptance:** POST to `/api/training/start` returns a `job_id`; SSE stream on `/api/training/progress/{id}` yields at least one reward event before completing; AND `StreamingCallback` is importable and callable with a `queue.Queue` argument without error

  - [x] 21.4 Add Gemini endpoints to `api/server.py`
    - Implement `POST /api/gemini/describe`: accept `{"frame_b64": str, "api_key": str | None}`; key priority: body → `.env` → None; call Gemini or return cached response; return `{"text": str}`
    - Implement `POST /api/gemini/action`: accept `{"frame_b64": str, "api_key": str | None, "system_prompt": str | None}`; same key priority; strip markdown fences; parse JSON; return action dict or cached response
    - Both endpoints use `load_cache()` from Module 5 for fallback
    - _Requirements: 19.7, 19.8_
    - _Design: Gemini endpoints design_
    - **Acceptance:** With no API key configured, both endpoints return a valid response dict containing `"action"` key (or `"text"` for describe); with API key raising `ConnectionError`, fallback activates

- [x] 22. Frontend experience hub (`frontend/index.html`)
  - [x] 22.1 Create `frontend/index.html` — page shell and navigation
    - Build sidebar nav with links to all 6 module sections + Wrap-up
    - Build setup status section: fetch `/api/config`, show key status, render API key input field stored to `sessionStorage`
    - Build schedule section: time-boxed agenda table matching FACILITATOR_GUIDE.md
    - Show `"Server not running — start with: python api/server.py"` banner when `/api/config` unreachable; disable server-dependent controls
    - All CSS inline in `<style>`, no external stylesheets except CDN libraries
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.12, 20.13, 20.14_
    - _Design: frontend/index.html, Page structure_
    - **Acceptance:** Opening `frontend/index.html` directly in Chrome shows the sidebar and schedule without a server; with server running, config check updates the key status panel

  - [x] 22.2 Implement Module 1 — Perception section with in-browser MediaPipe
    - Add `<section id="module-1">` with concept explanation panel and exercise prompt
    - Implement `getUserMedia` webcam capture feeding a hidden `<video>` element
    - Integrate `@mediapipe/tasks-vision` CDN: instantiate `HandLandmarker` in `LIVE_STREAM` mode with a result callback; draw landmarks using `DrawingUtils` from the same package on an overlay `<canvas>`
    - Compute joint angles in JS using `Math.atan2` dot-product formula; display as `[θ1, θ2, θ3]` state vector readout updated each frame
    - _Requirements: 20.5, 20.10_
    - _Design: Module 1 in-browser implementation_
    - **Acceptance:** Module 1 section shows live hand skeleton overlay and updating joint angles without the server running

  - [x] 22.3 Implement Module 2 — Simulation section with SSE stream
    - Add `<section id="module-2">` with concept explanation and "Start Simulation" button
    - On button click: open `EventSource("/api/simulation/stream")`; decode base64 frames; display in `<img>` element at ~20 fps
    - Stop button closes the SSE connection
    - _Requirements: 20.6, 20.10_
    - _Design: Simulation stream design_
    - **Acceptance:** Clicking "Start Simulation" shows animated Reacher frames; clicking "Stop" terminates the stream

  - [x] 22.4 Implement Module 3 — RL section with live training chart
    - Add `<section id="module-3">` with concept explanation and "Start Training" button
    - On button click: POST to `/api/training/start`; open SSE on `/api/training/progress/{job_id}`; update Chart.js line chart dataset on each reward event; show "Training complete" on done event
    - Chart.js loaded from CDN; chart renders in a `<canvas>` element
    - _Requirements: 20.7, 20.10_
    - _Design: Training stream design_
    - **Acceptance:** Clicking "Start Training" shows a live reward curve; chart clearly shows reward increasing over 50k timesteps

  - [x] 22.5 Implement Module 4 — Perception to Action section
    - Add `<section id="module-4">` with concept explanation
    - Combine webcam `@mediapipe/tasks-vision` HandLandmarker (from Module 1 webcamManager) with sim stream: extract landmark 8 position, compute action coords, append as query param to SSE URL
    - Display webcam canvas (with landmarks) and sim frame side by side in a two-column layout
    - _Requirements: 20.8, 20.10_
    - _Design: Module 4 in-browser implementation_
    - **Acceptance:** Hand movement visibly drives Reacher joint positions in the sim panel displayed next to the webcam

  - [x] 22.6 Implement Module 5 — Foundation Models section
    - Add `<section id="module-5">` with concept explanation
    - "Capture and Analyse" button: snapshot from webcam canvas → base64 JPEG → POST to `/api/gemini/describe` with `api_key` from `sessionStorage` → display returned text
    - "Get Robot Action" button: same snapshot → POST to `/api/gemini/action` → display returned action dict as formatted JSON
    - Show spinner while API call in flight; show cached-response notice if fallback was used
    - _Requirements: 20.9, 20.10_
    - _Design: Gemini endpoints design_
    - **Acceptance:** With no API key, both buttons show a cached response; with valid key, live Gemini response is displayed

  - [x] 22.7 Implement Wrap-up section and concept map
    - Add `<section id="wrapup">` with interactive concept map table: columns Module / Concept / Real-world example
    - Add next steps panel with links to LeRobot, Isaac Lab, ROS2 with one-line descriptions each
    - Add sim-to-real and digital twins explanatory text
    - _Requirements: 20.11, 20.14_
    - _Design: Page structure — wrapup section_
    - **Acceptance:** Wrap-up section visible in browser with a populated table mapping all 6 modules to concepts; three next-step links functional

- [x] 23. Final end-to-end hub verification
  - Start `python api/server.py` and open `http://localhost:8000` in Chrome
  - Verify sidebar navigation, schedule, and setup status all render correctly
  - Verify Module 1 webcam + MediaPipe landmarks work in-browser without server
  - Verify Module 2 simulation stream shows Reacher animation
  - Verify Module 3 training starts and live chart updates
  - Verify Module 4 hand movement drives sim in browser
  - Verify Module 5 Gemini integration returns response (or cached fallback)
  - Verify "Server not running" banner appears when server is stopped
  - Ask the user if any questions arise before concluding.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP — the core workshop functions without them
- Property tests (16.2–16.8) require `hypothesis` as a dev dependency (not in `requirements.txt`); install with `pip install hypothesis pytest` in the dev environment
- The `scale_landmark_to_action` function (Property 6) lives inside `modules/04_perception_to_action/01_hand_to_reacher.py`; expose it as a module-level function so it can be imported by the property test
- Pre-trained model files (task 7.1) require a one-time internet download by the repo maintainer; they are committed directly and never re-downloaded at workshop time
- The `fallback_hand_demo.mp4` (task 7.2) must be recorded or sourced by the repo maintainer before Module 1/4 scripts can be tested end-to-end
- Each task references specific requirements for traceability
- Checkpoints (tasks 17, 20) ensure incremental validation before the final documentation pass
- Each participant runs their own `api/server.py` locally — concurrent multi-user stream concerns do not apply
- The Gemini model string is `gemini-2.5-flash` (GA stable alias as of September 2026) — no expiry concern
- `StreamingCallback` is defined in Task 21.3; it must be implemented before Task 22.4 (frontend training chart) can be fully tested
- The MediaPipe Tasks API (`mp.tasks.vision.HandLandmarker`) via `mediapipe==1.0.0` (first stable major release, April 2026) is used for all hand detection. The `.task` model file must be present at `assets/hand_landmarker.task` before running Modules 1 or 4.
- SB3 is pinned to `2.9.0` (latest stable as of June 2026). Requires `numpy==2.2.4` and `torch==2.14.0+cpu` — all compatible with Python 3.12.
- `cvzone` is kept **for drawing utilities only** (overlay rendering, bounding boxes). Hand landmark detection uses the MediaPipe Tasks API directly. In the frontend, `@mediapipe/tasks-vision` (not `@mediapipe/hands`) is used for all JS hand detection.
- HalfCheetah-v5 uses `SAC` (Soft Actor-Critic), not PPO — import `from stable_baselines3 import SAC` in Module 2 scripts and task 7.1.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["3.1", "3.2"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["5.1", "7.3"] },
    { "id": 5, "tasks": ["5.2", "6.1", "7.1", "7.2", "7.4", "8.1", "8.2"] },
    { "id": 6, "tasks": ["6.2", "9.1", "9.2", "9.3", "10.1", "10.2", "10.3", "11.1"] },
    { "id": 7, "tasks": ["9.4", "10.4", "11.2", "12.1", "13.1", "13.2"] },
    { "id": 8, "tasks": ["11.3", "12.2", "13.3", "14.1", "15.1"] },
    { "id": 9, "tasks": ["15.2", "15.3", "16.1"] },
    { "id": 10, "tasks": ["16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8"] },
    { "id": 11, "tasks": ["17"] },
    { "id": 12, "tasks": ["18.1", "18.2"] },
    { "id": 13, "tasks": ["19.1", "19.2"] },
    { "id": 14, "tasks": ["20"] },
    { "id": 15, "tasks": ["21.1", "22.1", "22.2"] },
    { "id": 16, "tasks": ["21.2", "21.3", "21.4"] },
    { "id": 17, "tasks": ["22.3", "22.4", "22.5", "22.6", "22.7"] },
    { "id": 18, "tasks": ["23"] }
  ]
}
```
