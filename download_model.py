"""download_model.py — called by setup.bat to download hand_landmarker.task"""
import urllib.request
import sys
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEST = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"

def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print("[OK] hand_landmarker.task already present.")
        return 0
    print(f"Downloading to {DEST} ...")
    try:
        urllib.request.urlretrieve(URL, DEST)
        print("[OK] hand_landmarker.task downloaded.")
        return 0
    except Exception as e:
        print(f"[FAIL] Download failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
