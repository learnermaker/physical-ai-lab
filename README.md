# Physical AI Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2011-lightgrey.svg)]()

> A hands-on, 3-hour lab covering the complete Physical AI pipeline — webcam → hand tracking → simulation → reinforcement learning → teleoperation → foundation model control. No GPU required.

```
[Webcam] → [Perception Layer] → [State Vector] → [Sim Agent] → [Action] → [World/Sim]
    ↑                                                                             ↓
    └─────────────────────────────────────────────────────────────────────────────┘
                                          ↑
                             [Foundation Model Reasoning]
```

Every module has a working fallback — no GPU, no dedicated webcam, no Gemini API key required.

---

## Prerequisites

- Windows 11 (Mac/Linux: use `setup.sh` — best-effort, not fully tested)
- Internet connection for the initial ~700 MB package download
- Chrome or Edge browser (not Firefox — MediaPipe requires SharedArrayBuffer)

---

## Setup (once, before the lab)

Double-click **`setup.bat`**. It handles everything:

| Step | What happens |
|---|---|
| 1 | Finds Python 3.12 — downloads the official installer if missing |
| 2 | Creates `.venv` virtual environment in the repo folder |
| 3–5 | Installs PyTorch (CPU-only) and all lab packages |
| 6 | Registers the Jupyter kernel for VS Code |
| 7 | Downloads the MediaPipe hand landmark model (~8 MB) |
| 8 | Runs `verify_install.py` to confirm everything works |
| 9 | Launches the hub and opens http://localhost:8000 |

> **Windows SmartScreen** — click "More info → Run anyway" if a blue warning appears.

For the full step-by-step guide: open **[SETUP_GUIDE.html](SETUP_GUIDE.html)** in any browser.

---

## On the day

Double-click **`start_hub.bat`** — it starts the server, waits until healthy, and opens the browser.

Or from a terminal:
```bat
.venv\Scripts\activate
python api/server.py
```

Open **http://localhost:8000** in Chrome or Edge.

---

## Modules

| Module | Directory | What you do |
|--------|-----------|-------------|
| 0 — Kickoff | [modules/00_kickoff/](modules/00_kickoff/) | Run the notebook. Understand the closed loop before you build any piece of it. |
| 1 — Perception | [modules/01_perception/](modules/01_perception/) | Camera on. MediaPipe finds your hand. Numbers appear that a robot can act on. |
| 2 — Simulation | [modules/02_simulation/](modules/02_simulation/) | Meet the robot arm. Drag the sliders. Try to reach the target. |
| 3 — Reinforcement Learning | [modules/03_rl/](modules/03_rl/) | Watch a PPO agent learn CartPole from zero. Watch the chart. |
| 4 — Perception to Action | [modules/04_perception_to_action/](modules/04_perception_to_action/) | Your hand drives the robot arm. The loop closes. |
| 5 — Foundation Models | [modules/05_foundation_models/](modules/05_foundation_models/) | Send a camera frame to Gemini. Get a structured robot action back. |

---

## Key files

| File | Purpose |
|---|---|
| `setup.bat` | One-click full setup — run once before the lab |
| `start_hub.bat` | Start the hub on the day — health-checks before opening the browser |
| `start_hub.py` | Hub launcher (called by both bat files; also run directly to restart) |
| `verify_install.py` | Checks all dependencies are installed correctly |
| `download_model.py` | Downloads the MediaPipe hand landmark model (called by setup.bat) |
| `SETUP_GUIDE.html` | Visual setup guide — open in any browser |
| `api/server.py` | FastAPI server — simulation stream, Gemini calls, training SSE, file open |

---

## What to build next

Nine independent projects are listed in the **What to build next** section at the bottom of the hub page (http://localhost:8000). They range from a posture coach you can build today to a sim-to-real RL balancer that takes a few weeks and a Raspberry Pi. Every project has a closed feedback loop — that's the Physical AI test.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| setup.bat closes immediately | Right-click → Run as administrator, or open a terminal and run it from there |
| Stops at "Locating Python 3.12" | Install Python 3.12 from python.org with "Add to PATH" ticked |
| `[FAIL] mujoco` on ARM64 Windows | MuJoCo has no ARM64 Windows wheel — use WSL2 (see `instructor/COMMON_ISSUES.md`) |
| Hub page blank | Use Chrome or Edge — not Firefox |
| Port 8000 in use | `netstat -ano \| findstr :8000` → `taskkill /PID <n> /F`, then restart |
| Gemini returns errors | Check `.env` has a valid key; cached fallback activates automatically |
| Simulation shows text instead of graphics | Software render mode — no GPU detected. All physics still runs correctly. |

For facilitators: see `instructor/COMMON_ISSUES.md` for detailed per-issue fixes.

---

## Acknowledgements

- **Stable-Baselines3** — Antonin Raffin et al. (MIT). PPO implementation and callback patterns.
- **MediaPipe Tasks API** — Google. Hand landmark model and inference pipeline.
- **MuJoCo** — Google DeepMind (Apache-2.0). Physics engine for all robot environments.
- **Gymnasium** — Farama Foundation (MIT). Environment interface and CartPole/Reacher environments.
- **Hugging Face deep-rl-class** — (MIT). MDP conceptual structure referenced in Module 3.
- **Google Gemini Cookbook** — (Apache-2.0). `types.Part.from_bytes()` image pattern in Module 5.

---

*Physical AI Lab — Jim Seelan*
