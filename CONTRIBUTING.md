# Contributing to Physical AI Workshop

Thanks for your interest in contributing. This is a self-contained workshop repo — contributions that make it clearer, more robust, or more accessible are very welcome.

---

## What kinds of contributions are welcome

- Bug fixes — broken scripts, incorrect outputs, setup failures
- Typo and clarity fixes in documentation or code comments
- Compatibility fixes — package version updates, platform issues
- Fallback improvements — better no-camera, no-GPU, no-API-key behaviour
- Exercise hints or solution improvements

## What is out of scope

- Restructuring the module order or renaming files (participants follow the numbered sequence)
- Replacing core dependencies (MediaPipe, MuJoCo, Stable-Baselines3, Gemini)
- Adding new modules — the 3-hour scope is intentional
- Conda/Poetry/other environment managers — the repo uses `.venv` by design for simplicity on Windows

If you're unsure whether something fits, open an issue first and ask.

---

## How to contribute

1. **Fork** the repository on GitHub.

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/physical-ai-workshop.git
   cd physical-ai-workshop
   ```

3. **Create a branch** with a short descriptive name:
   ```bash
   git checkout -b fix/module3-reward-plot
   ```

4. **Set up the environment** (if you haven't already):
   ```bat
   setup.bat
   ```

5. **Make your changes.** Run the affected script to verify it works:
   ```bash
   python verify_install.py
   python modules/<affected_module>/<script>.py
   ```

6. **Commit** with a clear message:
   ```bash
   git commit -m "fix: reward plot not rendering on headless Windows"
   ```

7. **Push** to your fork:
   ```bash
   git push origin fix/module3-reward-plot
   ```

8. **Open a Pull Request** against the `master` branch of this repo. Fill in:
   - What the problem was
   - What you changed
   - How you tested it (OS, Python version, with/without webcam)

---

## Code style

- Follow the existing style in each file — no linter is enforced, but consistency matters.
- Keep line length reasonable (~100 chars).
- Add a comment if something non-obvious is happening.
- Do not remove or alter the `# TODO` / `# SOLUTION` blocks in exercise files — participants rely on them.

---

## Commit message format

Use a short prefix:

| Prefix | Use for |
|--------|---------|
| `fix:` | Bug fixes |
| `docs:` | Documentation only |
| `chore:` | Setup, dependencies, tooling |
| `feat:` | New content (discuss in an issue first) |

---

## Questions

Open a GitHub Issue and tag it `question`. Response time may vary — this repo is maintained alongside workshop delivery.
