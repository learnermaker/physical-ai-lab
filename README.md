# Physical AI Workshop

**Created by Jim Seelan** — a self-contained, 3-hour, beginner-friendly workshop for ~30 participants on Windows 11 laptops. Work through the complete Physical AI pipeline end-to-end:

```
[Webcam] → [Perception Layer] → [State Vector] → [Sim Agent] → [Action] → [World/Sim]
    ↑                                                                             ↓
    └─────────────────────────────────────────────────────────────────────────────┘
                                          ↑
                             [Foundation Model Reasoning]
```

Every module has a working fallback — no GPU, no dedicated webcam required.

---

## Prerequisites

- Windows 11 (Mac/Linux: use `setup.sh` — best-effort, not fully tested)
- Internet connection for the initial ~700 MB package download
- Chrome or Edge browser (Firefox not supported — MediaPipe requires SharedArrayBuffer)

---

## Setup (once, before the workshop)

Double-click **`setup.bat`**. It handles everything:

| Step | What happens |
|---|---|
| 1 | Finds Python 3.12 — downloads and installs if missing |
| 2 | Creates `.venv` virtual environment |
| 3–5 | Installs PyTorch (CPU), all workshop packages |
| 6 | Registers Jupyter kernel in VS Code |
| 7 | Downloads MediaPipe hand landmark model |
| 8 | Runs `verify_install.py` to confirm everything works |
| 9 | Launches the hub and opens http://localhost:8000 |

> **Windows SmartScreen** — click "More info → Run anyway" if a blue warning appears.

For the full step-by-step guide, see **[SETUP_GUIDE.html](SETUP_GUIDE.html)** (open in any browser).

---

## On the day

Double-click **`start_hub.bat`** — it activates the environment, starts the server, waits until healthy, and opens the browser.

Or manually:
```bash
.venv\Scripts\activate
python api/server.py
```

Open **http://localhost:8000** in Chrome or Edge.

---

## Modules

| Module | Directory | Description |
|--------|-----------|-------------|
| 0 — Kickoff | [modules/00_kickoff/](modules/00_kickoff/) | Physical AI pipeline overview — the closed loop from camera to action |
| 1 — Perception | [modules/01_perception/](modules/01_perception/) | Webcam → MediaPipe hand landmarks → joint angle state vector |
| 2 — Simulation | [modules/02_simulation/](modules/02_simulation/) | Gymnasium environments (CartPole, Reacher-v5), observation and action spaces |
| 3 — Reinforcement Learning | [modules/03_rl/](modules/03_rl/) | Train a PPO agent on CartPole with a live reward chart |
| 4 — Perception to Action | [modules/04_perception_to_action/](modules/04_perception_to_action/) | Hand position drives MuJoCo Reacher joint torques — the full sense–plan–act loop |
| 5 — Foundation Models | [modules/05_foundation_models/](modules/05_foundation_models/) | Camera frames → Gemini → structured robot action suggestions |
| 6 — Wrap-up | [modules/06_wrapup/](modules/06_wrapup/) | Concept map: sim-to-real transfer, digital twins, next steps |

---

## Key files

| File | Purpose |
|---|---|
| `setup.bat` | One-click full setup (run once before the workshop) |
| `start_hub.bat` | Start the hub on workshop day (health-checks before opening browser) |
| `start_hub.py` | Hub launcher called by both `setup.bat` and `start_hub.bat` — also run directly to restart the hub without re-running setup |
| `download_model.py` | Downloads MediaPipe hand landmark model (called by setup.bat) |
| `verify_install.py` | Checks all 14 dependencies are installed correctly |
| `SETUP_GUIDE.html` | Visual setup guide — open in any browser |

---

## Acknowledgements

- **CVZone drawing utilities** — [github.com/cvzone/cvzone](https://github.com/cvzone/cvzone) (MIT). Landmark overlay drawing in Modules 1 and 4.
- **Stable-Baselines3 Callbacks** — [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) by Antonin Raffin et al. (MIT). `LivePlotCallback` and `StreamingCallback` follow the SB3 callback pattern.
- **Hugging Face deep-rl-class** — [github.com/huggingface/deep-rl-class](https://github.com/huggingface/deep-rl-class) (MIT, low-maintenance reference). MDP concepts structure in Module 3.
- **Google Gemini Cookbook** — [github.com/google-gemini/cookbook](https://github.com/google-gemini/cookbook) (Apache-2.0). `types.Part.from_bytes()` image pattern in Module 5.
