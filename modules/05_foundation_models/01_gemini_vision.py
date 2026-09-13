#!/usr/bin/env python3
"""
Module 5 Demo 1: Gemini Vision
Captures a single frame from the camera and asks Gemini to describe the scene
in one sentence.

If GEMINI_API_KEY is absent or empty, the script prints a warning and selects
a response at random from cached_responses.json — no API call is made.

# Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)
"""

import importlib.util
import os
import random
import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The package directory starts with a digit, so normal `import` is not valid
# Python syntax.  Load __init__.py directly via importlib.
_init_path = Path(__file__).resolve().parent / "__init__.py"
_spec = importlib.util.spec_from_file_location("fm_init", _init_path)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)
load_cache = _fm.load_cache


def main() -> None:
    """Run the Gemini Vision demo."""
    import cv2
    from dotenv import load_dotenv
    from PIL import Image
    from utils.camera import open_camera

    # -----------------------------------------------------------------------
    # 1. Load environment
    # -----------------------------------------------------------------------
    load_dotenv(REPO_ROOT / ".env")
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # -----------------------------------------------------------------------
    # 2. Load cache (exits with code 1 if the file is missing or empty)
    # -----------------------------------------------------------------------
    cache = load_cache()  # defaults to sibling cached_responses.json

    PROMPT = "Describe this scene in one sentence."

    # -----------------------------------------------------------------------
    # 3. No API key — fall back to cache immediately (no camera, no API call)
    # -----------------------------------------------------------------------
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Using cached response (no API call).")
        print(random.choice(list(cache.values())))
        return

    # -----------------------------------------------------------------------
    # 4. API key present — open camera, capture one frame
    # -----------------------------------------------------------------------
    cap = open_camera()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Warning: Could not read a frame from the camera. Using cached response.")
        print(random.choice(list(cache.values())))
        return

    # Convert BGR (OpenCV) → RGB (PIL)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # -----------------------------------------------------------------------
    # 5. Call Gemini — deferred import so the fallback path never requires the
    #    google-genai package to be importable (useful in offline environments)
    # -----------------------------------------------------------------------
    from google import genai  # noqa: E402

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[PROMPT, pil_image],
    )
    print(response.text)


if __name__ == "__main__":
    main()
