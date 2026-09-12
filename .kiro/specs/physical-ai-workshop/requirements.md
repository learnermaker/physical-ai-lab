# Requirements Document

## Introduction

This document defines the requirements for a **Physical AI Workshop** GitHub repository — a self-contained, 3-hour, beginner-friendly workshop for approximately 30 participants on Windows laptops. Participants clone the repository, run a one-command setup, and work through six guided hands-on modules that trace the complete Physical AI pipeline: `[Webcam] → [Perception] → [State Vector] → [Sim Agent] → [Action]`, augmented by a Foundation Model reasoning layer.

The repository must be fully functional offline (after initial setup) with robust fallbacks for every external dependency, so no participant is blocked by hardware or network issues during the session.

---

## Glossary

- **Workshop_Repo**: The GitHub repository `physical-ai-workshop` and its complete file tree.
- **Setup_Script**: The `setup.bat` (Windows) or `setup.sh` (Mac/Linux) script that creates the environment and installs all dependencies.
- **Verify_Script**: The `verify_install.py` script that confirms the environment is functional before the workshop begins.
- **Participant**: An attendee running the workshop on their own Windows laptop.
- **Facilitator**: The instructor delivering the workshop.
- **Module**: One of the six numbered workshop sections (00 through 06), each in its own subdirectory under `modules/`.
- **Exercise**: The `exercise.py` file in each module containing scaffolded starter code with `# TODO` markers.
- **Pipeline**: The Physical AI pipeline — webcam input → perception layer → state vector → sim agent → action output, with optional Foundation Model reasoning.
- **Fallback**: A pre-bundled or offline alternative that activates when an external resource is unavailable.
- **Gemini_Client**: The `google-genai` REST-based SDK used to call the Gemini API.
- **SB3**: Stable-Baselines3, the reinforcement learning library.
- **MediaPipe_Hands**: Hand landmark detection using the MediaPipe Tasks API (`mp.tasks.vision.HandLandmarker`). This is the current, actively maintained MediaPipe API (v1.0.0, April 2026). The legacy `mediapipe.solutions.hands` API was officially retired by Google in March 2023 and is no longer supported. `mediapipe==1.0.0` is pinned for stability.
- **Reacher_Env**: The MuJoCo `Reacher-v5` Gymnasium environment used in Module 4.
- **State_Vector**: A numeric array encoding the observable state of the simulation at one timestep.
- **Experience_Hub**: The browser-based workshop interface served at `http://localhost:8000`, consisting of `api/server.py` (FastAPI backend) and `frontend/index.html` (single-page frontend).
- **Hub_Server**: The FastAPI application in `api/server.py` that serves static files, proxies simulation renders, streams training progress, and handles Gemini API key loading.
- **SSE**: Server-Sent Events — the unidirectional streaming protocol used by the Hub_Server to push simulation frames and training metrics to the browser.

---

## Requirements

### Requirement 1: Repository Structure and Discoverability

**User Story:** As a Participant, I want a clearly organised repository with predictable file locations, so that I can navigate to any module without a facilitator's help.

#### Acceptance Criteria

1. THE Workshop_Repo SHALL contain a top-level `README.md` that describes the workshop narrative, lists prerequisites, provides a quick-start command sequence, and links to each module in the order they are taught; and a primary quick-start that instructs participants to run `python api/server.py` and open `http://localhost:8000` in their browser after completing setup.
2. THE Workshop_Repo SHALL contain a `PRE_WORKSHOP_SETUP.md` that provides step-by-step instructions for completing environment setup before the session, including a link to Google AI Studio for obtaining a Gemini API key.
3. THE Workshop_Repo SHALL contain exactly the following directories at the repository root: `modules/`, `utils/`, `models/`, `assets/`, and `instructor/`, and no other top-level directories beyond these five and any hidden, tooling, or server directories (e.g. `.git/`, `.github/`, `api/`, `frontend/`).
4. WHEN a Participant opens any module directory, THE Workshop_Repo SHALL contain at minimum one numbered demonstration file (`.py` or `.ipynb`) and exactly one `exercise.py` file, where numbering starts at `01_` and increments by one for each subsequent demonstration file.
5. THE Workshop_Repo SHALL contain a `.env.example` file with a placeholder `GEMINI_API_KEY=your_key_here` entry and a comment explaining where to obtain a key.
6. THE Workshop_Repo SHALL include a `.gitignore` that excludes `.env`, `__pycache__/`, `*.pyc`, `conda-meta/`, and common editor artefacts.
7. IF a Participant navigates to any directory listed in criterion 3 and that directory contains no files, THEN THE Workshop_Repo SHALL contain a `README.md` or `.gitkeep` file within that directory so that the directory is present after cloning.

---

### Requirement 2: Environment Setup — Windows (Primary Path)

**User Story:** As a Participant, I want to run one script that installs everything correctly on Windows, so that I arrive at the workshop with a working environment.

#### Acceptance Criteria

1. THE Setup_Script (`setup.bat`) SHALL create a conda environment named `physical-ai` using Python 3.12 only, with no additional conda-managed packages beyond Python itself.
2. WHEN the conda environment is created, THE Setup_Script SHALL activate the environment and install PyTorch CPU-only first by invoking `pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu` before any other pip install step.
3. AFTER PyTorch is installed, THE Setup_Script SHALL invoke `pip install -r requirements.txt` to install all remaining packages.
4. THE `requirements.txt` SHALL pin the following versions exactly: `numpy==2.2.4`, `mujoco>=3.1,<4.0` (latest compatible 3.x), `gymnasium[classic_control,mujoco]==1.3.0`, `stable-baselines3==2.9.0`, `opencv-python==4.13.0.92`, `mediapipe==1.0.0`, and `google-genai` unpinned (version-sensitive conflicts make pinning counterproductive for this dependency — see design.md §2.6 for rationale).
5. WHEN all packages are installed successfully, THE Setup_Script SHALL register the `physical-ai` conda environment as a Jupyter kernel by running `python -m ipykernel install --user --name physical-ai --display-name "Physical AI Workshop"`.
6. WHEN kernel registration completes, THE Setup_Script SHALL invoke `python verify_install.py` and print its output to the terminal; IF `verify_install.py` exits with a non-zero code, THE Setup_Script SHALL print `Setup completed with warnings — see output above`.
7. IF the conda environment creation step does not complete within 120 seconds, THE Setup_Script SHALL abort and print `Conda solver is slow. Re-run with: conda install -n base conda-libmamba-solver && conda config --set solver libmamba`.
8. IF any `pip install` step exits with a non-zero code, THE Setup_Script SHALL print `Pip install failed. Check your internet connection and re-run setup.bat`.

---

### Requirement 3: Environment Setup — Mac/Linux (Fallback Path)

**User Story:** As a Facilitator, I want Mac or Linux participants to follow an equivalent setup path, so that non-Windows attendees are not excluded.

#### Acceptance Criteria

1. THE Setup_Script (`setup.sh`) SHALL perform the same steps as `setup.bat` — conda Python-only environment creation, pip install from `requirements.txt`, kernel registration, and verify script invocation — using POSIX shell syntax.
2. THE `README.md` SHALL note that `setup.sh` is a best-effort fallback and that the workshop is validated on Windows 11 only.

---

### Requirement 4: ARM64 Windows Compatibility Warning

**User Story:** As a Facilitator, I want ARM64 Windows participants (e.g., Surface Pro X) to be identified and redirected before the session, so that no participant is unexpectedly blocked by a missing MuJoCo wheel.

#### Acceptance Criteria

1. THE `PRE_WORKSHOP_SETUP.md` SHALL contain a clearly labelled warning section stating that MuJoCo has no ARM64 Windows wheel and that affected participants must use WSL2.
2. THE `PRE_WORKSHOP_SETUP.md` SHALL provide step-by-step WSL2 setup instructions sufficient to complete the standard `setup.sh` path inside WSL2.

---

### Requirement 5: Installation Verification

**User Story:** As a Participant, I want to run a verification script before the workshop, so that I know exactly which components are working and which are not.

#### Acceptance Criteria

1. THE Verify_Script SHALL attempt to import each of the following 12 packages individually — `numpy`, `mujoco`, `gymnasium`, `stable_baselines3`, `cv2`, `mediapipe`, `cvzone`, `google.genai`, `torch`, `PIL`, `matplotlib`, `huggingface_hub` — and print exactly one result line per package in the format `[PASS] <package>` or `[FAIL] <package>` within 30 seconds of script start.
2. IF any package import fails, THE Verify_Script SHALL print, on the line immediately following the `[FAIL]` line for that package, the corresponding `pip install` command required to install it, using the install target names: `numpy`, `mujoco`, `gymnasium`, `stable-baselines3`, `opencv-python`, `mediapipe`, `cvzone`, `google-genai`, `torch`, `Pillow`, `matplotlib`, `huggingface_hub`.
3. WHEN all package import checks have completed, THE Verify_Script SHALL attempt to open the default webcam (device index 0), display a live frame in a window titled `verify — press q to close` for no longer than 2 seconds or until the user presses `q`, then close the window and release the device; IF the webcam cannot be opened or the frame cannot be read, THE Verify_Script SHALL print `[WARN] Camera unavailable — fallback video will be used` and count the webcam check as 1 failed check in the summary.
4. WHEN the webcam check completes, THE Verify_Script SHALL create a `CartPole-v1` gymnasium environment, call `step()` once with a valid action, then call `close()`; IF an `OpenGL` or rendering-related exception is raised during environment creation or stepping, THE Verify_Script SHALL print `[WARN] OpenGL unavailable — headless rendering will be used` and count the environment check as 1 failed check in the summary.
5. WHEN all checks — 12 package imports, 1 webcam check, and 1 environment check — have been attempted, THE Verify_Script SHALL print a single summary line in the format `Setup complete: N/14 checks passed (W warnings — fallbacks active)` where N is the count of checks that produced neither a `[FAIL]` nor a `[WARN]` result, and W is the count of checks that produced a `[WARN]` result. IF W is 0, the `(W warnings — fallbacks active)` portion SHALL be omitted.

---

### Requirement 6: Pre-bundled Assets and Model Weights

**User Story:** As a Facilitator, I want all assets and model weights committed to the repository, so that the workshop runs fully offline once the repo is cloned.

#### Acceptance Criteria

1. THE Workshop_Repo SHALL contain `models/ppo-CartPole-v1.zip` (a pre-trained SB3 PPO checkpoint for CartPole-v1, approximately 100 KB).
2. THE Workshop_Repo SHALL contain `models/sac-HalfCheetah-v5.zip` (a pre-trained SB3 SAC checkpoint for HalfCheetah-v5, approximately 200 KB).
3. THE Workshop_Repo SHALL contain `assets/fallback_hand_demo.mp4`, a 30-second video recording of a hand performing the gestures used in Modules 1 and 4.
4. THE Workshop_Repo SHALL contain `modules/05_foundation_models/cached_responses.json` with at least 10 pre-recorded Gemini API responses covering each distinct prompt type used in Module 5.
5. THE `cached_responses.json` file SHALL use a keyed structure where the key is a normalised prompt string and the value is the response text, enabling deterministic lookup.

---

### Requirement 7: Shared Utility — Robust Camera Open (`utils/camera.py`)

**User Story:** As a Participant, I want camera-dependent scripts to start correctly even when my webcam is unavailable or blocked, so that I can continue the workshop without assistance.

#### Acceptance Criteria

1. THE `utils/camera.py` module SHALL export a function `open_camera(index=0)` that attempts to open the specified webcam index using `cv2.VideoCapture` and returns an object whose `read()` method yields valid frames.
2. IF `cv2.VideoCapture.isOpened()` returns `False` for the given index, THEN THE `open_camera` function SHALL log a warning message to `stderr` and return a `cv2.VideoCapture` object opened from the fallback video path instead.
3. WHEN the fallback video reaches its last frame, THE `open_camera` fallback SHALL loop back to the first frame so that downstream code receives a continuous stream.
4. IF the caller supplies a `fallback_path` argument and that path cannot be opened as a readable video file, THEN THE `open_camera` function SHALL raise a `ValueError` with a message indicating the path is not a valid video source.
5. IF the fallback video file cannot be opened, THEN THE `open_camera` function SHALL raise a `RuntimeError` with a message indicating that neither the webcam nor the fallback video source is available.

---

### Requirement 8: Shared Utility — Gymnasium Environment Factory (`utils/gym_utils.py`)

**User Story:** As a Participant, I want simulation environments to start regardless of whether OpenGL rendering is available on my machine, so that I am not blocked by display driver issues.

#### Acceptance Criteria

1. THE `utils/gym_utils.py` module SHALL export a function `make_env(env_id, render_mode="human")` that accepts a Gymnasium environment ID string of 1 to 200 characters and a `render_mode` parameter with a default value of `"human"`.
2. WHEN `make_env` is called with a valid `env_id` and `render_mode="human"`, THE `make_env` function SHALL attempt to create and return a Gymnasium environment using `render_mode="human"`.
3. IF creating the environment with `render_mode="human"` raises a rendering-related exception, THEN THE `make_env` function SHALL print a message to `stdout` indicating that `render_mode="human"` failed and that `render_mode="rgb_array"` will be attempted, then retry environment creation with `render_mode="rgb_array"`.
4. IF creating the environment with `render_mode="rgb_array"` raises a rendering-related exception, THEN THE `make_env` function SHALL print a message to `stdout` indicating that `render_mode="rgb_array"` failed and that `render_mode=None` will be attempted, then retry environment creation with `render_mode=None`.
5. WHEN environment creation succeeds on any fallback attempt, THE `make_env` function SHALL print a message to `stdout` indicating the `render_mode` that succeeded.
6. IF all three render modes (`"human"`, `"rgb_array"`, and `None`) raise rendering-related exceptions, THEN THE `make_env` function SHALL raise a `RuntimeError` with a message indicating that environment creation failed for all render modes.
7. IF `make_env` is called with an `env_id` that is not recognised by the Gymnasium registry, THEN THE `make_env` function SHALL propagate the exception from Gymnasium without retrying render modes.

---

### Requirement 9: Module 0 — Workshop Kickoff

**User Story:** As a Participant, I want an interactive notebook that introduces the Physical AI pipeline before I write any code, so that subsequent modules have conceptual grounding.

#### Acceptance Criteria

1. THE `modules/00_kickoff/overview.ipynb` notebook SHALL contain a text cell that defines the Physical AI pipeline as `[Webcam] → [Perception Layer] → [State Vector] → [Sim Agent] → [Action] → [World/Sim] → (back to Webcam)`, noting that the loop is closed: the robot's action changes the environment, which changes the next camera frame.
2. THE `overview.ipynb` notebook SHALL include at least one code cell that imports `numpy` and prints a sample state vector to make the concept concrete.
3. THE `overview.ipynb` notebook SHALL run to completion without errors when the `physical-ai` kernel is selected.

---

### Requirement 10: Module 1 — Perception (Webcam and Hand Tracking)

**User Story:** As a Participant, I want to capture webcam frames, detect hand landmarks, and compute joint angles, so that I understand how raw sensor data becomes a structured state vector.

#### Acceptance Criteria

1. THE `modules/01_perception/01_webcam_basics.py` script SHALL open the camera using `utils.camera.open_camera`, display the live (or fallback) frame in a window, and print the frame shape to `stdout` on the first frame only.
2. THE `modules/01_perception/02_hand_tracking.py` script SHALL use `mediapipe.tasks.vision.HandLandmarker` to detect up to two hands per frame and draw landmarks on the displayed frame using `mediapipe.solutions.drawing_utils`.
3. THE `modules/01_perception/03_joint_angles.py` script SHALL compute the angles for at least the thumb, index, and middle finger MCP joints using landmark positions and print them as a formatted state vector `[θ1, θ2, θ3, ...]` once per second.
4. THE `modules/01_perception/exercise.py` SHALL contain scaffolded code with `# TODO` markers that guide the Participant to add a fourth finger angle to the state vector.
5. IF the camera is unavailable, ALL scripts in `modules/01_perception/` SHALL operate correctly on the fallback video without code changes by the Participant.

---

### Requirement 11: Module 2 — Simulation (Gymnasium and MuJoCo)

**User Story:** As a Participant, I want to step through Gymnasium environments manually and inspect observation and action spaces, so that I understand what a simulation environment exposes to an agent.

#### Acceptance Criteria

1. THE `modules/02_simulation/01_gym_intro.py` script SHALL create a `CartPole-v1` environment using `utils.gym_utils.make_env`, print the `observation_space` and `action_space`, run 200 random-action steps, render each step, and close the environment.
2. THE `modules/02_simulation/02_mujoco_reacher.py` script SHALL create a `Reacher-v5` environment using `utils.gym_utils.make_env`, print the dimensionality of the observation and action vectors, and run 100 random-action steps.
3. THE `modules/02_simulation/03_pretrained_agent.py` script SHALL exist as a standalone script that participants run BEFORE the workshop session (referenced in `PRE_WORKSHOP_SETUP.md`). During the live session, the facilitator references results rather than running the script live.
4. THE `modules/02_simulation/exercise.py` SHALL contain scaffolded code with `# TODO` markers that guide the Participant to print the minimum and maximum bounds of the `Reacher-v5` observation space.
5. IF MuJoCo rendering fails, ALL scripts in `modules/02_simulation/` SHALL fall back gracefully through the `make_env` render fallback chain without terminating.

---

### Requirement 12: Module 3 — Reinforcement Learning (Watch an Agent Learn)

**User Story:** As a Participant, I want to watch a PPO agent learn CartPole in real time with a live reward plot, so that I develop intuition for the RL training loop.

#### Acceptance Criteria

1. THE `modules/03_rl/01_mdp_concepts.ipynb` notebook SHALL define the Markov Decision Process concepts — state, action, reward, transition — with a `CartPole-v1` example in code cells and run to completion without errors.
2. THE `modules/03_rl/02_train_cartpole.py` script SHALL train a `stable_baselines3.PPO` agent on `CartPole-v1` for 50,000 timesteps using a `LivePlotCallback` that updates a matplotlib reward chart every 1,000 timesteps.
3. THE `LivePlotCallback` SHALL be defined in `modules/03_rl/02_train_cartpole.py` or `utils/gym_utils.py` and SHALL update the plot without blocking the training loop (non-interactive matplotlib backend fallback to file output where display is unavailable).
4. AFTER training completes, THE `02_train_cartpole.py` script SHALL save the trained model to `models/ppo-CartPole-trained.zip` and print the final mean episode reward.
5. THE `modules/03_rl/exercise.py` SHALL contain scaffolded code with `# TODO` markers that guide the Participant to change the total training timesteps and observe the effect on the reward curve.

---

### Requirement 13: Module 4 — Perception to Action (Hand Controls Sim Robot)

**User Story:** As a Participant, I want my hand gestures to directly drive joint torques in the MuJoCo Reacher environment, so that I experience the complete sense–plan–act loop and understand how human demonstrations are collected for imitation learning.

#### Acceptance Criteria

1. WHEN the `modules/04_perception_to_action/01_hand_to_reacher.py` script starts, THE script SHALL attempt to open the camera using `utils.camera.open_camera` and `mediapipe.tasks.vision.HandLandmarker`, and, IF no camera device is available, SHALL fall back to the pre-supplied fallback video; IF neither source can be opened, THE script SHALL print an error message and exit with a non-zero exit code.
2. WHEN a hand is detected, THE `01_hand_to_reacher.py` script SHALL map the normalised x/y position of the index finger tip (landmark 8) to the two joint torque commands of the `Reacher-v5` action space, scaling each coordinate linearly from the `[0, 1]` normalised landmark range to the `[-1, 1]` torque command range (positive = torque in one direction, negative = opposite direction).
3. WHEN no hand is detected for more than 1 consecutive second, THE `01_hand_to_reacher.py` script SHALL send a zero-vector action of dimension 2 to the Reacher environment and display an on-screen message `"No hand detected — holding position"`.
4. WHEN the simulation render is available, THE `01_hand_to_reacher.py` script SHALL display the camera frame with hand landmarks overlaid and the simulation render in the same window or in two separate windows updated each frame.
5. IF the simulation render is unavailable, THEN THE `01_hand_to_reacher.py` script SHALL display the camera frame with hand landmarks overlaid alongside a numeric readout showing the current joint torque commands (two floating-point values, one per joint).
6. THE `modules/04_perception_to_action/exercise.py` SHALL contain scaffolded code with `# TODO` markers that guide the Participant to map a second hand landmark's normalised position to a third Reacher joint torque command.

**Implementation note:** Reacher-v5's action space is continuous torque — `[-1, 1]` maps to the magnitude and direction of force applied to each joint. The robot moves in response to applied torque, not to a direct position target.

---

### Requirement 14: Module 5 — Foundation Models (Gemini as Robot Brain)

**User Story:** As a Participant, I want to send camera frames to Gemini and receive structured robot action suggestions, so that I see how a Foundation Model can serve as a reasoning layer in a Physical AI pipeline.

#### Acceptance Criteria

1. THE `modules/05_foundation_models/01_gemini_vision.py` script SHALL load `GEMINI_API_KEY` from a `.env` file using `python-dotenv` and initialise a `google.genai` client.
2. IF the `GEMINI_API_KEY` environment variable is absent or empty after loading the `.env` file, THEN THE `01_gemini_vision.py` script SHALL print a warning message to `stdout` and select a response uniformly at random from the entries in `cached_responses.json` without calling the Gemini API.
3. THE `modules/05_foundation_models/02_gemini_robot_brain.py` script SHALL capture a single frame from the camera (or fallback video), encode it as a base64 JPEG, and send it to `gemini-2.5-flash` with a prompt requesting a structured action suggestion in JSON format.
4. WHEN a valid JSON response is received from the Gemini API, THE `02_gemini_robot_brain.py` script SHALL parse the JSON and print the top-level keys and their values of the suggested action to `stdout`.
5. IF the Gemini API call raises an exception or the response body cannot be parsed as valid JSON, THEN THE `02_gemini_robot_brain.py` script SHALL log an error message indicating the failure reason to `stderr` and select a response uniformly at random from the entries in `cached_responses.json` as the fallback action.
6. WHILE the Gemini API request is in flight, THE `02_gemini_robot_brain.py` script SHALL display the message `"Calling Gemini..."` to `stdout` before the request is issued and follow it with a completion indicator once a response or error is received.
7. THE `modules/05_foundation_models/exercise.py` SHALL contain scaffolded code with at least one `# TODO` marker identifying the prompt string the Participant must modify, and at least one `# TODO` marker identifying where the Participant should observe or print the model's output.
8. IF `cached_responses.json` is absent or contains no valid entries when a fallback is required, THEN THE script requiring the fallback SHALL log an error message to `stderr` and exit with a non-zero exit code.

---

### Requirement 15: Module 6 — Wrap-up Concepts Map

**User Story:** As a Participant, I want a closing notebook that maps every hands-on activity to real-world Physical AI concepts, so that I leave with a mental model I can apply beyond the workshop.

#### Acceptance Criteria

1. THE `modules/06_wrapup/concepts_map.ipynb` notebook SHALL contain a text cell that maps each module (1–5) to its corresponding concept in the Physical AI pipeline.
2. THE `concepts_map.ipynb` notebook SHALL contain a text cell covering sim-to-real transfer, digital twins, and at least two suggested next steps (e.g., Isaac Lab, LeRobot).
3. THE `concepts_map.ipynb` notebook SHALL run to completion without errors when the `physical-ai` kernel is selected.

---

### Requirement 16: Exercise Design and Scaffolding Standards

**User Story:** As a Participant, I want every exercise file to have enough scaffolded code that I only need to fill in small, well-scoped gaps, so that I can complete exercises within the allotted 10–15 minutes.

#### Acceptance Criteria

1. EVERY `exercise.py` file SHALL import all required libraries and define all helper functions; each gap requiring Participant input SHALL be delimited by a `# TODO START` marker on the opening line and a `# TODO END` marker on the closing line, with all Participant-editable code contained exclusively between those markers.
2. EVERY `exercise.py` file SHALL include a `# SOLUTION HINT:` comment immediately after each `# TODO END` marker containing at least one sentence describing the expected approach; the hint SHALL NOT contain executable code or a complete expression that directly solves the task.
3. WHEN a Participant runs an unmodified `exercise.py`, THE exercise SHALL either raise a `NotImplementedError` whose message identifies the function name and line number of the unimplemented `# TODO` block, OR produce output that is missing at least one explicitly labelled expected artifact (such as a printed result, table, or plot title) that is named in a `# EXPECTED OUTPUT:` comment within the same file.
4. EVERY `exercise.py` file SHALL include a `# SOLUTION:` block at the bottom of the file, commented out with `#` on every line, containing a complete solution such that uncommenting the block and running the file produces the same output as the reference solution for that exercise.

---

### Requirement 17: Instructor Support Materials

**User Story:** As a Facilitator, I want comprehensive support materials, so that I can troubleshoot participant issues quickly without disrupting the session.

#### Acceptance Criteria

1. THE `instructor/FACILITATOR_GUIDE.md` file SHALL contain a time-boxed agenda matching the workshop schedule (Pre-session (30 min), Module 0 (15 min), Module 1 (30 min), Module 2 (20 min — simulation concepts only, no pretrained demo), Module 3 (30 min), Module 4 (40 min), Module 5 (25 min), Wrap-up (15 min), Contingency buffer (15 min)) with facilitation notes for each segment.
2. THE `instructor/COMMON_ISSUES.md` file SHALL document at minimum the following failure modes and their mitigations: ARM64 Windows MuJoCo wheel missing, Windows 11 camera privacy prompt, conda solver hang, PyTorch CUDA build accidentally installed, Jupyter wrong kernel selected, HuggingFace download blocked on corporate network, Windows SmartScreen blocking `setup.bat`, port 8000 already in use when starting the hub server, `.env` file missing or incorrectly formatted, Visual C++ Redistributables missing causing MuJoCo import failure, pip install taking longer than 10 minutes, Gemini free-tier rate limit hit during facilitator demonstration, and mediapipe 1.0.0 `mp.solutions.drawing_utils` not available (may be removed in 1.0 rewrite — use `DrawingUtils` from `@mediapipe/tasks-vision` in JS or OpenCV drawing in Python instead).
3. THE `instructor/COMMON_ISSUES.md` file SHALL provide copy-pasteable terminal commands for each mitigation described.

---

### Requirement 18: License and Attribution

**User Story:** As a Facilitator, I want all reused open-source code to be clearly attributed, so that the repository is legally compliant and models good open-source practice for participants.

#### Acceptance Criteria

1. THE Workshop_Repo SHALL contain a `LICENSE` file using the MIT License with the workshop author's name and year.
2. THE `README.md` SHALL contain an **Acknowledgements** section that lists each reused code pattern, its source, and its licence: CVZone drawing utilities (MIT), Raffin/SB3 callbacks documentation pattern (MIT), HF deep-rl-class MDP notebook structure (MIT, low-maintenance reference), and Google Gemini cookbook image pattern (Apache-2.0).
3. EVERY source file that incorporates a reused pattern SHALL include an inline comment of the form `# Adapted from: <source URL> (MIT/Apache-2.0)` at the point of use.

---

### Requirement 19: Experience Hub — Backend Server (`api/server.py`)

**User Story:** As a Participant, I want to start a single command and have all workshop demos available in my browser, so that I don't need to manage terminal windows, VS Code instances, and multiple Python scripts simultaneously.

#### Acceptance Criteria

1. THE Workshop_Repo SHALL contain `api/server.py`, a FastAPI application that starts with `python api/server.py` and serves the Experience_Hub on `http://localhost:8000`.
2. THE `requirements.txt` SHALL include `fastapi` and `uvicorn[standard]` as additional dependencies.
3. WHEN `api/server.py` starts, THE Hub_Server SHALL load `GEMINI_API_KEY` from `.env` using `python-dotenv` and expose it via a `GET /api/config` endpoint that returns `{"gemini_key_configured": true}` if the key is present, or `{"gemini_key_configured": false}` if absent — it SHALL NOT return the key value itself.
4. THE Hub_Server SHALL expose a `GET /api/simulation/stream` Server-Sent Events endpoint that, when called, creates a `Reacher-v5` Gymnasium environment, steps it continuously with actions received via query parameters, and streams each rendered frame as a base64-encoded JPEG event.
5. THE Hub_Server SHALL expose a `POST /api/training/start` endpoint that launches PPO CartPole training (50,000 timesteps) in a background thread and returns a `{"job_id": "<uuid>"}` response immediately.
6. THE Hub_Server SHALL expose a `GET /api/training/progress/{job_id}` SSE endpoint that streams `{"timestep": int, "mean_reward": float}` events as the training job progresses and a final `{"done": true, "model_path": "models/ppo-CartPole-trained.zip"}` event on completion.
7. THE Hub_Server SHALL expose a `POST /api/gemini/describe` endpoint that accepts a base64-encoded JPEG frame and an optional `api_key` field in the request body, uses the provided key (falling back to the server-side `.env` key), calls `gemini-2.5-flash` with a scene description prompt, and returns the response text; IF neither key is available, it SHALL return a random entry from `cached_responses.json`.
8. THE Hub_Server SHALL expose a `POST /api/gemini/action` endpoint that accepts a base64-encoded JPEG frame, an optional `api_key`, and an optional `system_prompt` field, calls `gemini-2.5-flash`, and returns a parsed JSON action dict; on any API error or JSON parse failure it SHALL return a random entry from `cached_responses.json`.
9. THE Hub_Server SHALL serve `frontend/index.html` and any co-located static assets (`frontend/`) at the root path `/`.
10. IF the Hub_Server cannot bind to port 8000, THE `api/server.py` script SHALL print a clear error message and exit with a non-zero code.

---

### Requirement 20: Experience Hub — Frontend Page (`frontend/index.html`)

**User Story:** As a Participant, I want a single browser page that gives me everything I need for the workshop — schedule, instructions, live demos, and exercises — so that I can focus on learning rather than navigating files and terminals.

#### Acceptance Criteria

1. THE `frontend/index.html` page SHALL be a single self-contained HTML file with all CSS and JavaScript inline or loaded from CDN; it SHALL NOT require a build step, bundler, or Node.js.
2. THE page SHALL display a persistent left sidebar navigation listing all six modules (00–05) and a Wrap-up section, with the active module highlighted; clicking a module name SHALL scroll or navigate to that module's section without a page reload.
3. THE page SHALL display a workshop schedule section showing the time allocation for each module (matching the FACILITATOR_GUIDE.md timings) so participants know where they are in the session.
4. THE page SHALL contain a setup status section that calls `GET /api/config` on load and displays whether the Gemini API key is configured, with a text input field for participants to enter their own key if it is not; the entered key SHALL be stored in the browser's `sessionStorage` and used for all subsequent Gemini API calls from that browser tab.
5. WHEN a Participant navigates to Module 1 (Perception), THE page SHALL activate the browser webcam using `getUserMedia`, run MediaPipe Hand Landmarker JavaScript (`@mediapipe/tasks-vision` from CDN — `HandLandmarker` API) on the live frame, overlay 21 hand landmarks on a `<canvas>` element, and display the computed joint angle state vector as a numeric readout updated each frame — all without any Python backend call.
6. WHEN a Participant navigates to Module 2 (Simulation), THE page SHALL display an embedded simulation panel with a "Start Simulation" button; clicking it SHALL open a `GET /api/simulation/stream` SSE connection and render the incoming base64 JPEG frames in an `<img>` or `<canvas>` element at a target of 20 fps.
7. WHEN a Participant navigates to Module 3 (Reinforcement Learning), THE page SHALL display a "Start Training" button; clicking it SHALL call `POST /api/training/start`, then open a `GET /api/training/progress/{job_id}` SSE connection, and render a live reward chart (using Chart.js from CDN) that updates with each incoming event.
8. WHEN a Participant navigates to Module 4 (Perception to Action), THE page SHALL combine the browser webcam with the simulation stream: hand landmark positions SHALL be extracted using MediaPipe JS, converted to Reacher action coordinates, sent as query parameters to `GET /api/simulation/stream`, and the resulting simulation frames SHALL be displayed alongside the webcam feed.
9. WHEN a Participant navigates to Module 5 (Foundation Models), THE page SHALL provide a "Capture and Analyse" button that takes a snapshot from the browser webcam, sends it to `POST /api/gemini/describe`, and displays the returned text; a second "Get Robot Action" button SHALL send the same snapshot to `POST /api/gemini/action` and display the returned action dict.
10. EACH module section on the page SHALL include: a concept explanation panel (3–5 sentences covering the Physical AI concept for that module), a "Run Demo" interactive area (live demo controls as described in criteria 5–9), and an exercise prompt panel describing what the participant should modify in the corresponding `exercise.py` file.
11. THE page SHALL include a Wrap-up section containing an interactive concept map table showing the mapping of each module to its Physical AI pipeline stage, and links to the three recommended next steps (LeRobot, Isaac Lab, ROS2).
12. THE page SHALL be functional on Chrome and Edge (the two most common browsers on Windows) without any browser extension or plugin.
13. IF the Hub_Server is unreachable (e.g. not started), THE page SHALL display a `"Server not running — start with: python api/server.py"` banner and disable all server-dependent controls, while keeping the webcam access permission check (via `getUserMedia`) and the schedule/instructions display fully operational. Note: MediaPipe JS WASM inference requires `SharedArrayBuffer` which is only available when served over HTTPS or from `localhost` — the page is designed to be served by `api/server.py`, not opened as a `file://` URL.
14. THE page SHALL NOT require participants to open any workshop content file (concept explanations, module instructions, schedule, or API key setup) outside of the browser. Exercise code editing (modifying `exercise.py` files) takes place in VS Code or a text editor alongside the browser — the page SHALL display clear exercise prompts so participants know exactly what to edit.
