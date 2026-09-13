# Contributing

Thanks for taking the time. Contributions that make this workshop clearer, more reliable, or more accessible are welcome.

---

## What fits

- Bug fixes — broken scripts, wrong outputs, setup failures on different machines
- Typos and clarity improvements in docs or code comments
- Compatibility fixes — package version updates, Windows version differences, platform edge cases
- Fallback improvements — better handling when there's no camera, no GPU, or no API key
- Exercise hint or solution improvements

## What doesn't fit

- Restructuring modules or renaming files — participants follow the numbered sequence and tools reference these paths
- Replacing core dependencies (MediaPipe, MuJoCo, Stable-Baselines3, Gemini) — these are deliberate choices
- Adding new modules — the 3-hour scope is intentional
- Switching to conda, Poetry, or other environment managers — the repo uses `.venv` for simplicity on Windows

If you're not sure whether something fits, open an issue first.

---

## How to contribute

1. **Fork** the repository.

2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/physical-ai-lab.git
   cd physical-ai-lab
   ```

3. **Create a branch:**
   ```bash
   git checkout -b fix/brief-description
   ```

4. **Set up the environment:**
   ```bat
   setup.bat
   ```

5. **Make your change** and verify it works:
   ```bat
   .venv\Scripts\activate
   python verify_install.py
   python modules/<affected_module>/<script>.py
   ```

6. **Commit:**
   ```bash
   git commit -m "fix: brief description of what changed"
   ```

7. **Push and open a Pull Request** against the `master` branch. Include:
   - What the problem was
   - What you changed
   - How you tested it (OS, Python version, with/without webcam)

---

## Code style

- Match the existing style in the file you're editing
- Keep lines to ~100 characters
- Add a comment when something non-obvious is happening
- Don't touch the `# TODO` / `# SOLUTION` blocks in exercise files — participants depend on them

## Commit prefixes

| Prefix | Use for |
|--------|---------|
| `fix:` | Bug fixes |
| `docs:` | Documentation only |
| `chore:` | Setup, dependencies, tooling |
| `feat:` | New content (open an issue first) |

---

## Questions

Open a GitHub Issue tagged `question`. Response time varies — this repo is maintained alongside active workshop delivery.

---

*Physical AI Workshop — Jim Seelan*
