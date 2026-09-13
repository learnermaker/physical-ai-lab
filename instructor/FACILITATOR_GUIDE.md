# Facilitator Guide — Physical AI Workshop

**Total running time:** ~220 minutes (3 h 40 min including contingency buffer)

---

## Agenda at a Glance

| # | Segment | Duration |
|---|---------|----------|
| 0 | Pre-session | 30 min |
| 1 | Module 0 — Kickoff | 15 min |
| 2 | Module 1 — Perception | 30 min |
| 3 | Module 2 — Simulation | 20 min |
| 4 | Module 3 — Reinforcement Learning | 30 min |
| 5 | Module 4 — Perception to Action | 40 min |
| 6 | Module 5 — Foundation Models | 25 min |
| 7 | Wrap-up | 15 min |
| — | Contingency buffer | 15 min |
| **Total** | | **~220 min** |

---

## Segment 1 — Pre-session (30 min)

**What this time is for:** Participants arrive and complete environment validation. You are troubleshooting, not presenting.

### Facilitator actions

1. Display `PRE_WORKSHOP_SETUP.md` on the projector as a reference for latecomers.
2. Ask participants to open a terminal in the `physical-ai-workshop` directory and run:
   ```
   conda activate physical-ai
   python verify_install.py
   ```
3. Walk the room. For any `[FAIL]` lines, consult `instructor/COMMON_ISSUES.md`.
4. Start the hub server on your own machine:
   ```
   python api/server.py
   ```
   Confirm it opens at `http://localhost:8000` in Chrome or Edge.
5. Remind participants to keep the terminal running throughout the session.

### Key talking points

- "If you see `Setup complete: 14/14 checks passed`, you're good to go."
- "Webcam warnings are fine — the scripts fall back to a demo video automatically."
- ARM64 Windows participants should already be on WSL2 (flagged in pre-workshop comms). If not, see COMMON_ISSUES.md.

### Transition cue

Once most participants show a passing verify output (allow stragglers to catch up during Module 0), move to the projector and open `modules/00_kickoff/overview.ipynb`.

---

## Segment 2 — Module 0: Kickoff (15 min)

**Goal:** Set the conceptual stage. No code written by participants yet.

### Facilitator actions

1. Open `modules/00_kickoff/overview.ipynb` in Jupyter (kernel: **Physical AI Workshop**).
2. Run all cells live on the projector.
3. Draw the pipeline on the whiteboard or reference the diagram in the notebook:
   ```
   [Webcam] → [Perception] → [State Vector] → [Sim Agent] → [Action] → [World/Sim] → (back to Webcam)
   ```

### Key talking points

- "Every module today builds one piece of this pipeline. By Module 4, the full loop is closed."
- "The loop being *closed* is the key idea — the robot's action changes the environment, which changes the next camera frame."
- "We're not training a model from scratch today. We're wiring together pre-trained pieces and writing the glue code."

### Transition cue

"Let's start at the very beginning of the pipeline — the camera." Open a terminal and navigate to `modules/01_perception/`.

---

## Segment 3 — Module 1: Perception (30 min)

**Goal:** Participants run webcam capture, see hand landmarks drawn in real time, and extend the joint-angle state vector.

### Facilitator actions

1. Run `01_webcam_basics.py` on the projector:
   ```
   python modules/01_perception/01_webcam_basics.py
   ```
2. Run `02_hand_tracking.py`:
   ```
   python modules/01_perception/02_hand_tracking.py
   ```
3. Run `03_joint_angles.py` and point out the printed state vector in the terminal.
4. Direct participants to open `exercise.py` in VS Code alongside the browser and complete the `# TODO` block (add a fourth finger angle). Allow ~10 min.
5. Ask 1–2 volunteers to share their terminal output.

### Key talking points

- "The MediaPipe model runs entirely on CPU — no GPU needed. This is the 2026 Tasks API, not the deprecated `solutions` API."
- "A state vector is just a list of numbers describing the world at this instant. We just built one from raw pixels."
- "The fallback video loops automatically if your camera isn't detected — your code path is identical either way."

### Transition cue

"We now have a state vector. Next we need something to *receive* that state — a simulation environment." Switch terminal to `modules/02_simulation/`.

---

## Segment 4 — Module 2: Simulation (20 min)

**Goal:** Participants observe Gymnasium environments, understand observation and action spaces, and run random agents.

> **Note:** The pretrained HalfCheetah demo (`03_pretrained_agent.py`) is run by participants *before* the session as a motivating preview. Do **not** run it live — reference the results instead.

### Facilitator actions

1. Run `01_gym_intro.py` on the projector:
   ```
   python modules/02_simulation/01_gym_intro.py
   ```
   Point out the printed `observation_space` and `action_space` lines.
2. Run `02_mujoco_reacher.py`:
   ```
   python modules/02_simulation/02_mujoco_reacher.py
   ```
   Highlight the dimensionality of the Reacher observation vector.
3. Direct participants to open `exercise.py` and complete the `# TODO` block (print observation space bounds). Allow ~8 min.

### Key talking points

- "CartPole has 4 observations and 2 discrete actions. Reacher has 11 observations and 2 continuous torques. That's the action space Module 4 will control with your hand."
- "A random agent is the baseline: if a trained agent can't beat random, something is wrong."
- "You all ran the HalfCheetah pretrained agent before the session — that's what 1 million training steps buys you."

### Transition cue

"How does a sim agent actually *learn* to produce those actions? That's the RL module." Switch terminal to `modules/03_rl/`.

---

## Segment 5 — Module 3: Reinforcement Learning (30 min)

**Goal:** Participants watch a PPO agent learn CartPole live with a reward plot, then experiment with timestep count.

### Facilitator actions

1. Open `modules/03_rl/01_mdp_concepts.ipynb` briefly and run the first two cells to define state/action/reward.
2. Run `02_train_cartpole.py` on the projector:
   ```
   python modules/03_rl/02_train_cartpole.py
   ```
   Keep the matplotlib window visible so the live reward curve is on screen.
3. While it trains (~2–3 min), walk through the MDP slide / whiteboard diagram.
4. After training completes, confirm `models/ppo-CartPole-trained.zip` was saved.
5. Direct participants to open `exercise.py` and modify the timestep count. Allow ~10 min.

### Key talking points

- "Reward is the only supervision signal. The agent figures out the policy by trial and error."
- "Watch the reward curve — it stays flat early, then climbs sharply. That inflection point is when the agent discovers balance."
- "We're using PPO from Stable-Baselines3. In practice this is the same algorithm that trains many real robot controllers."

### Transition cue

"We have perception. We have a simulation. We've seen a trained agent. Now let's wire them together — your hand drives the robot." Switch terminal to `modules/04_perception_to_action/`.

---

## Segment 6 — Module 4: Perception to Action (40 min)

**Goal:** Participants close the physical AI loop: webcam → hand landmark → torque command → Reacher joint movement.

### Facilitator actions

1. Run `01_hand_to_reacher.py` on the projector with a camera or fallback video:
   ```
   python modules/04_perception_to_action/01_hand_to_reacher.py
   ```
2. Move your index finger left/right and up/down while participants watch the Reacher arm respond.
3. Point out the "No hand detected — holding position" message when the hand leaves the frame.
4. Direct participants to open `exercise.py` and complete the `# TODO` block (map a second landmark to a third torque channel). Allow ~15 min.
5. Ask 1–2 participants to demo their result on the projector.

### Key talking points

- "The mapping is linear: the normalised [0,1] position of landmark 8 scales to a [-1,1] torque command."
- "Torque means *force applied*, not *position target*. The arm overshoots and corrects — that's physics."
- "This is exactly how human demonstration data is collected for imitation learning at scale."
- "The full loop is now closed: camera → state → action → sim → camera."

### Transition cue

"We just used our hand as the 'brain'. What if we replaced it with a language model?" Switch terminal to `modules/05_foundation_models/`.

---

## Segment 7 — Module 5: Foundation Models (25 min)

**Goal:** Participants send a camera frame to Gemini and receive a structured action suggestion.

### Facilitator actions

1. Confirm your `.env` file is present with a valid `GEMINI_API_KEY`. If not, the scripts fall back to `cached_responses.json` automatically.
2. Run `01_gemini_vision.py` on the projector:
   ```
   python modules/05_foundation_models/01_gemini_vision.py
   ```
3. Run `02_gemini_robot_brain.py`:
   ```
   python modules/05_foundation_models/02_gemini_robot_brain.py
   ```
   Point out the "Calling Gemini..." message and the structured JSON action response.
4. Direct participants to open `exercise.py` and modify the prompt string in the `# TODO` block. Allow ~10 min.

### Key talking points

- "Gemini is receiving a base64-encoded JPEG and returning a JSON action dictionary. That's the full Foundation Model pattern for robotics."
- "The fallback cache means the exercise works even without an API key or network access."
- "This pattern — capture, encode, prompt, parse, act — is how VLMs are integrated into real robot pipelines today."

### Transition cue

"Let's bring everything together and talk about where this goes next." Switch to `modules/06_wrapup/`.

---

## Segment 8 — Wrap-up (15 min)

**Goal:** Map every module to the Physical AI pipeline and point participants toward next steps.

### Facilitator actions

1. Open `modules/06_wrapup/concepts_map.ipynb` and run all cells.
2. Summarise the pipeline on the whiteboard, labelling each module.
3. Highlight the two suggested next steps: **Isaac Lab** (sim-to-real with industrial-grade MuJoCo) and **LeRobot** (end-to-end imitation learning from HuggingFace).
4. Invite questions. If questions run long, move overflow into the contingency buffer.
5. Share the repo URL and remind participants that everything runs offline.

### Key talking points

- "Perception → State Vector is Module 1. State Vector → Action is Module 4. Foundation Model reasoning is Module 5. The RL training loop is Module 3."
- "Sim-to-real transfer is the hard part in industry. The simulation we used today is a research-grade MuJoCo environment — the same physics used in real robot labs."
- "The skills you practised — camera I/O, environment wrappers, reward callbacks, REST API calls — are the literal building blocks of modern robotics stacks."

### Transition cue

End the structured session. Remain available for individual questions during the contingency buffer.

---

## Segment 9 — Contingency Buffer (15 min)

**Purpose:** Absorb overruns, support participants who hit issues, and handle Q&A overflow.

### When to use this time

| Situation | Action |
|-----------|--------|
| A module ran long and participants haven't finished their exercise | Let them complete it now. Walk the room. |
| A common setup issue wasn't resolved pre-session | Use `COMMON_ISSUES.md` terminal commands projected on screen. |
| A participant's MuJoCo environment fails to render | The hub simulation stream automatically shows state values as text frames (labelled "no-GPU mode"). No code change needed. |
| Gemini API rate limit hit during Module 5 | Confirm `cached_responses.json` is loaded; all scripts fall back automatically. |
| Extra time remains | Demo the HalfCheetah pretrained agent (`03_pretrained_agent.py`) live as a bonus. |

### Key reminders

- All exercises have a `# SOLUTION:` block commented out at the bottom — participants can uncomment it to see the reference implementation.
- `verify_install.py` can be re-run at any time to diagnose environment issues.
- The repo is self-contained and fully offline after setup — participants can continue at home.

---

## Quick Reference

### Key commands

```bash
# Activate environment
conda activate physical-ai

# Start hub server (recommended — health-checks port before starting)
start_hub.bat

# Start hub server (alternative — direct Python invocation)
python api/server.py

# Verify installation
python verify_install.py

# Module scripts (run from repo root)
python modules/01_perception/01_webcam_basics.py
python modules/01_perception/02_hand_tracking.py
python modules/01_perception/03_joint_angles.py
python modules/02_simulation/01_gym_intro.py
python modules/02_simulation/02_mujoco_reacher.py
python modules/03_rl/02_train_cartpole.py
python modules/04_perception_to_action/01_hand_to_reacher.py
python modules/05_foundation_models/01_gemini_vision.py
python modules/05_foundation_models/02_gemini_robot_brain.py
```

### Fallback reminders

| Issue | Automatic behaviour |
|-------|---------------------|
| No webcam | Scripts use `assets/fallback_hand_demo.mp4` |
| No OpenGL / display | `make_env` falls back through `rgb_array` → `None`; hub simulation stream shows state values as text in a PIL-generated frame (labelled "no-GPU mode") |
| No Gemini API key | Module 5 scripts use `cached_responses.json` |
| No internet during session | All assets are pre-bundled in the repo |

For detailed per-issue commands, see `instructor/COMMON_ISSUES.md`.

