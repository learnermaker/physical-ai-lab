# Facilitator Guide — Physical AI Workshop

Total running time: approximately 3 hours 40 minutes including the contingency buffer.

---

## Agenda at a Glance

| # | Segment | Time |
|---|---------|------|
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

## Pre-session (30 min)

**Goal:** Everyone gets a green verify_install output before you start presenting.

### Your setup (do this first)

Start the hub on your machine:
```bat
start_hub.bat
```
Confirm http://localhost:8000 opens in Chrome or Edge. Keep this running throughout.

### For participants

Put the `README.md` on the projector (or open the `SETUP_GUIDE.html` in a browser tab). Ask everyone to open a terminal in the `physical-ai-lab` folder and run:
```bat
.venv\Scripts\activate
python verify_install.py
```

Walk the room. For any `[FAIL]` lines, the fix is in `instructor/COMMON_ISSUES.md`. Most failures are one command to resolve.

**What's fine:**
- `[WARN] Camera unavailable` — fallback video kicks in automatically
- `[WARN] OpenGL unavailable` — simulation falls back to software render

**Talking points:**
- "If you see 13 or 14 passes, you're good."
- "The fallback video is a looping hand demo — your code runs identically whether there's a real webcam or not."

**Transition:** Once most people show passing output, move to the projector.

---

## Module 0 — Kickoff (15 min)

**Goal:** Set the conceptual frame. No code written by participants yet.

Open the hub at http://localhost:8000 on the projector and scroll to Module 0. Then open the notebook:
```bat
modules\00_kickoff\overview.ipynb
```
Select the **Physical AI Workshop** kernel in VS Code, run all cells.

Draw the loop on the whiteboard or point to the pipeline diagram in the hub:
```
[Webcam] → [Perception] → [State Vector] → [Sim Agent] → [Action] → [World/Sim] → (back to Webcam)
```

**Talking points:**
- "Every module today builds one piece of this loop."
- "By Module 4, your hand is controlling the simulation in real time — the full loop is closed."
- "The loop being *closed* is the key idea. The robot's action changes the environment, which changes the next frame."

**Transition:** "Let's start at the beginning — the camera."

---

## Module 1 — Perception (30 min)

**Goal:** Participants see hand landmarks drawn live and extend the joint-angle state vector.

Run these on the projector in order:
```bat
python modules\01_perception\01_webcam_basics.py
python modules\01_perception\02_hand_tracking.py
python modules\01_perception\03_joint_angles.py
```

Point out the state vector printing in the terminal every second while 03 runs. Then direct participants to open `exercise.py` in VS Code — the task is to add the ring finger angle (one line of code, same pattern as the others). Allow ~10 minutes.

You can also use the Module 1 panel in the hub — Start Camera shows the same landmark overlay in the browser.

**Talking points:**
- "We went from raw pixels to four numbers. That's the Perception stage."
- "MediaPipe runs entirely on CPU — no GPU needed."
- "Those four numbers are what the rest of the pipeline operates on."

**Transition:** "We have a state vector. Now we need somewhere to send it — a simulation."

---

## Module 2 — Simulation (20 min)

**Goal:** Understand observation and action spaces. Watch the arm move.

```bat
python modules\02_simulation\01_gym_intro.py
python modules\02_simulation\02_mujoco_reacher.py
```

Point out the printed observation space (10 values for Reacher-v5) and action space (2 continuous torques). In the hub, open the Module 2 panel — participants can drag the sliders and watch the arm respond immediately.

> **Note on the pre-trained agent:** Participants were asked to run `03_pretrained_agent.py` before the session as a preview. Don't run it live — just reference those results.

Participants open `exercise.py` — print the observation space bounds. Allow ~8 minutes.

**Talking points:**
- "Ten numbers describe the entire arm. Two numbers control it."
- "Random torques is the baseline — any trained or hand-crafted policy needs to do better."
- "The arm overshoots and doesn't snap to position. That's physics — real robot arms have the same problem."

**Transition:** "How does an agent learn to produce *good* torques? That's what RL does."

---

## Module 3 — Reinforcement Learning (30 min)

**Goal:** Watch a PPO agent learn CartPole live. Understand the reward curve.

Open the notebook briefly to define MDP terms:
```bat
modules\03_rl\01_mdp_concepts.ipynb
```

Run the training script on the projector with the hub open (both show the live reward curve):
```bat
python modules\03_rl\02_train_cartpole.py
```

While it trains (2–3 minutes), walk through the MDP diagram. Point out the inflection on the reward curve — that's the moment the agent discovers balance.

After training completes, confirm `models/ppo-CartPole-trained.zip` was saved. Participants open `exercise.py` — try different timestep counts and compare the curves. Allow ~10 minutes.

**Talking points:**
- "The only supervision is the reward signal. No labels, no teacher."
- "The flat part of the curve is exploration — random stumbling. The jump is the breakthrough."
- "PPO is the same algorithm that trains many real robot controllers today."

**Transition:** "We have perception. We have a simulation. We've seen a trained agent. Now let's wire them together."

---

## Module 4 — Perception to Action (40 min)

**Goal:** Close the loop. Hand position drives the Reacher arm in real time.

```bat
python modules\04_perception_to_action\01_hand_to_reacher.py
```

Move your index finger while participants watch the arm respond. Point out the status text and the distance-to-target reading. Show the "No hand detected — holding position" message by briefly moving your hand out of frame.

The Module 4 panel in the hub does the same thing in the browser — both approaches work fine.

Participants open `exercise.py` — add landmark 4 (thumb tip) as a third control channel. Allow ~15 minutes. Ask a couple of people to demo on the projector.

**Talking points:**
- "The mapping is two lines of maths: `action = 2 * x - 1`. That's the entire brain right now."
- "This is teleoperation — the same pattern surgeons use with a da Vinci robot."
- "The full loop is now closed: camera → landmarks → torque → physics → camera."
- "This setup is also exactly how demonstration data is collected for imitation learning."

**Transition:** "We just used our hand as the brain. What if we replaced it with a language model?"

---

## Module 5 — Foundation Models (25 min)

**Goal:** Send a camera frame to Gemini, get a structured robot action back.

Check your `.env` file has a valid `GEMINI_API_KEY`. If not, the scripts fall back to `cached_responses.json` automatically — the exercise still works either way.

```bat
python modules\05_foundation_models\01_gemini_vision.py
python modules\05_foundation_models\02_gemini_robot_brain.py
```

Point out the "Calling Gemini..." message and the JSON action response. Show both scripts in the hub — Module 5 panel has buttons to describe the scene and get a robot action.

Participants open `exercise.py` — write their own system prompt and observe how the model's reasoning changes. Allow ~10 minutes.

**Talking points:**
- "Gemini sees a JPEG and returns JSON. The prompt is the only thing connecting a billion-parameter model to a physical machine."
- "Change the prompt, change what the robot can do — without any retraining."
- "The fallback cache shows the same JSON structure whether Gemini is live or not."

**Transition:** "Let's bring everything together."

---

## Wrap-up (15 min)

**Goal:** Map every module to the pipeline. Point toward next steps.

Open the hub and scroll to the **Wrap-up** section on the projector. Walk through the concept map. Label each module on the whiteboard:

```
Module 1 = Perception → State Vector
Module 2 = World/Sim (the environment)
Module 3 = Training the agent
Module 4 = Closed loop
Module 5 = Foundation model reasoning layer
```

Then scroll to **What to build next** — 9 independent projects ranging from laptop-only beginner projects to Pi + servo hardware builds. Point out the hardware tier column.

Two recommended paths to highlight:
- **Isaac Lab** — for those who want to go deep on sim-to-real transfer
- **LeRobot** — for those who want to take the Module 4 teleoperation data and train a real policy

Invite questions. Overflow goes into the contingency buffer.

**Talking points:**
- "Every project in the capstone section has a closed feedback loop — that's the Physical AI test."
- "The hard part in industry is sim-to-real transfer. Everything you did today was training in simulation — deploying on hardware is the next step."
- "The skills you practised — camera I/O, env wrappers, reward callbacks, REST API calls — are the actual building blocks of modern robotics stacks."

---

## Contingency Buffer (15 min)

| Situation | What to do |
|-----------|------------|
| A module ran long | Let participants finish. Walk the room. |
| A setup issue from pre-session is still unresolved | Use `COMMON_ISSUES.md` — project the fix command. |
| Simulation shows text instead of graphics | That's the software render fallback — all physics still runs. No fix needed. |
| Gemini rate limit hit | Cache fallback is already active. Continue normally. |
| Extra time | Demo the HalfCheetah pretrained agent live as a bonus: `python modules\02_simulation\03_pretrained_agent.py` |

---

## Quick Reference

### Activate the environment
```bat
.venv\Scripts\activate
```

### Start the hub
```bat
start_hub.bat
```
Or directly: `python api/server.py`

### Key scripts
```bat
python verify_install.py

python modules\01_perception\01_webcam_basics.py
python modules\01_perception\02_hand_tracking.py
python modules\01_perception\03_joint_angles.py

python modules\02_simulation\01_gym_intro.py
python modules\02_simulation\02_mujoco_reacher.py

python modules\03_rl\02_train_cartpole.py

python modules\04_perception_to_action\01_hand_to_reacher.py

python modules\05_foundation_models\01_gemini_vision.py
python modules\05_foundation_models\02_gemini_robot_brain.py
```

### Automatic fallbacks

| When this is missing | What happens |
|----------------------|--------------|
| No webcam | Scripts use `assets/fallback_hand_demo.mp4` |
| No OpenGL / GPU | Simulation falls back to software render (text frame, labelled "no-GPU mode") |
| No Gemini API key | Module 5 uses `cached_responses.json` |
| No internet during session | All assets are pre-bundled in the repo |

Full per-issue fixes: `instructor/COMMON_ISSUES.md`

---

*Physical AI Workshop — Facilitator Guide — Jim Seelan*
