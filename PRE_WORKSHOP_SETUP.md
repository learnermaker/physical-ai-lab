# Pre-Workshop Setup — Physical AI Workshop
*Created by Jim Seelan*

> **Quick reference:** For a visual guide with tabs, open **`SETUP_GUIDE.html`** in your browser.

---

## The short version

1. Clone or download the repository
2. Double-click **`setup.bat`**
3. Wait ~15 minutes — it installs everything automatically
4. The browser opens at **http://localhost:8000**

---

## What setup.bat does

| Step | Action |
|---|---|
| [1/9] | Finds Python 3.12 — downloads the official installer if not present |
| [2/9] | Creates `.venv` virtual environment in the repo folder |
| [3/9] | Activates it |
| [4/9] | Installs PyTorch CPU-only (~200 MB) |
| [5/9] | Installs all workshop packages from `requirements.txt` |
| [6/9] | Registers the Jupyter kernel for VS Code |
| [7/9] | Downloads the MediaPipe hand landmark model (~8 MB) via `download_model.py` |
| [8/9] | Runs `verify_install.py` — prints PASS/WARN/FAIL for each package |
| [9/9] | Launches the hub server and opens http://localhost:8000 |

> **Windows SmartScreen:** Click **More info → Run anyway** if a protection dialog appears.

---

## Step 1: Get the repository

```bash
git clone <repo-url>
cd physical-ai-workshop
```

Or download the ZIP from GitHub and extract it.

---

## Step 2: Double-click setup.bat

No admin rights needed. Expected total time: **10–15 minutes** on a decent connection.

```
[1/9] Locating Python 3.12...
[OK] Found: C:\...\Python312\python.exe
[2/9] Setting up virtual environment...
[OK] Virtual environment healthy.
...
Setup complete!
```

If setup fails partway, fix the reported issue and run `setup.bat` again — it skips steps already completed.

---

## Step 3: Check the verification output

```
[PASS] numpy
[PASS] mujoco
...
[WARN] Camera unavailable -- fallback video will be used
Setup complete: 13/14 checks passed
```

- `[PASS]` — all good
- `[WARN]` — fallback active, workshop still works
- `[FAIL]` — needs fixing; the script prints the exact `pip install` command

---

## Step 4: Get a Gemini API key (optional)

Module 5 uses Google Gemini to reason about camera frames. Without a key it uses cached responses — it still works, but live results are more interesting.

1. Go to [https://aistudio.google.com/](https://aistudio.google.com/)
2. Sign in → **Get API key** → copy it
3. Copy `.env.example` to `.env` and set: `GEMINI_API_KEY=your_key`

```bat
copy .env.example .env
rem Open .env in Notepad and paste your key
```

---

## Step 5: Start on workshop day

Double-click **`start_hub.bat`** — it starts the server, waits until healthy, and opens the browser automatically.

Or run manually:
```bat
.venv\Scripts\activate
python api/server.py
```

Open **http://localhost:8000** in **Chrome or Edge** (not Firefox).

---

## Step 6: Preview the HalfCheetah agent (optional)

```bat
.venv\Scripts\activate
python modules/02_simulation/03_pretrained_agent.py
```

---

## ARM64 Windows (Surface Pro X, Snapdragon)

MuJoCo has no ARM64 Windows wheel. Use WSL2:

```powershell
# PowerShell (Administrator)
wsl --install
# Restart, open Ubuntu, then:
bash setup.sh
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| setup.bat closes immediately | Right-click → Open, or run from a terminal |
| Stops at "Locating Python 3.12" | Install Python 3.12 from python.org, tick "Add to PATH" |
| `[FAIL] mujoco` on ARM64 | See ARM64 section above |
| Hub page blank | Use Chrome or Edge — not Firefox |
| Port 8000 in use | `netstat -ano \| findstr :8000` → `taskkill /PID <n> /F` |
| Gemini fails | Check `.env` has a valid key; cached fallback activates automatically |
