"""
download_model.py — downloads all binary assets that are gitignored.

Called by setup.bat (step 7) and setup.sh (step 7) on a fresh clone.

Assets downloaded:
  1. assets/hand_landmarker.task  (~8 MB)   — MediaPipe hand landmark model
  2. models/sac-HalfCheetah-v5.zip (~3 MB)  — pre-trained SAC policy for Module 2

Both files are safe to re-download (idempotent — skipped if already present).
"""
import urllib.request
import ssl
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. hand_landmarker.task
# ---------------------------------------------------------------------------
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
LANDMARKER_DEST = REPO_ROOT / "assets" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# 2. sac-HalfCheetah-v5.zip  (Module 2 — 03_pretrained_agent.py)
#    Source: farama-minari/HalfCheetah-v5-SAC-expert on HuggingFace
#    License: MIT (Farama Foundation)
# ---------------------------------------------------------------------------
SAC_URL = (
    "https://huggingface.co/farama-minari/HalfCheetah-v5-SAC-expert/"
    "resolve/main/halfcheetah-v5-sac-expert.zip"
)
SAC_DEST = REPO_ROOT / "models" / "sac-HalfCheetah-v5.zip"

# ---------------------------------------------------------------------------
# 3. assets/fallback_hand_demo.mp4
#    Copied from frontend/assets/ which IS tracked in git.
#    This ensures the Python scripts (Modules 1, 4) have the demo video
#    even if the participant has no webcam.
# ---------------------------------------------------------------------------
DEMO_SRC  = REPO_ROOT / "frontend" / "assets" / "fallback_hand_demo.mp4"
DEMO_DEST = REPO_ROOT / "assets" / "fallback_hand_demo.mp4"


def _download(url: str, dest: Path, label: str) -> bool:
    """Download url → dest. Returns True on success, False on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[OK] {label} already present.")
        return True
    print(f"Downloading {label} (~{_size_hint(url)})...")
    try:
        # Disable certificate verification only as last resort
        ctx = ssl.create_default_context()
        try:
            urllib.request.urlretrieve(url, dest,
                reporthook=lambda b, bs, ts: None)
        except ssl.SSLError:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as r, open(dest, "wb") as f:
                f.write(r.read())
        mb = dest.stat().st_size / 1024 / 1024
        print(f"[OK] {label} downloaded ({mb:.1f} MB).")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}", file=sys.stderr)
        return False


def _copy_local(src: Path, dest: Path, label: str) -> bool:
    """Copy a local file. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[OK] {label} already present.")
        return True
    if not src.exists():
        print(f"[FAIL] {label}: source not found at {src}", file=sys.stderr)
        return False
    import shutil
    shutil.copy2(src, dest)
    mb = dest.stat().st_size / 1024 / 1024
    print(f"[OK] {label} copied ({mb:.1f} MB).")
    return True


def _size_hint(url: str) -> str:
    if "hand_landmarker" in url:
        return "~8 MB"
    if "HalfCheetah" in url or "sac" in url.lower():
        return "~3 MB"
    return "unknown size"


def main() -> int:
    ok = True
    ok &= _download(LANDMARKER_URL, LANDMARKER_DEST, "hand_landmarker.task")
    ok &= _download(SAC_URL,        SAC_DEST,        "sac-HalfCheetah-v5.zip")
    ok &= _copy_local(DEMO_SRC,     DEMO_DEST,       "assets/fallback_hand_demo.mp4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
