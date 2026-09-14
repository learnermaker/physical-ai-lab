#!/usr/bin/env python3
"""
Module 5 Demo 1: Gemini Vision
================================
Captures a single camera frame and asks Gemini to describe what it sees.

This is the simplest possible VLM (Vision-Language Model) interaction:
  1. Capture one JPEG frame from the camera
  2. Send it to Gemini with the text prompt "Describe this scene in one sentence."
  3. Print the text response

No API key?  No problem.  The script falls back to a cached response
automatically — you'll see a static description from the pre-built cache.

What you will see:
  - A one-sentence description of whatever the camera sees (or the cache)

How it connects to the pipeline:
  This replaces the MediaPipe perception layer with a foundation model.
  Instead of computing joint angles, the model reasons about the scene.
  Next: 02_gemini_robot_brain.py asks the model for a structured action.

Run from the repo root:
  python modules/05_foundation_models/01_gemini_vision.py

Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)
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
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # This is the simplest possible VLM interaction: image in → text out.
    # For robots, the insight is: a model trained on billions of internet images
    # already "understands" scenes, objects, and spatial relationships.
    # Instead of training a custom perception model for each new task,
    # you can zero-shot prompt a VLM and get useful structured output.
    # ─────────────────────────────────────────────────────────────────────────
    import cv2
    from dotenv import load_dotenv
    from PIL import Image
    from utils.camera import open_camera

    # ── 1. Load API key from .env file ───────────────────────────────────────
    # load_dotenv() reads GEMINI_API_KEY=... from the .env file in the repo root.
    # If the key is absent or empty, we fall back to the cache (no API call).
    load_dotenv(REPO_ROOT / ".env")
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # ── 2. Load the response cache ───────────────────────────────────────────
    # cache is a dict of {prompt_text: response_text} pairs.
    # If the cache file is missing or empty, load_cache() exits with an error.
    cache = load_cache()   # defaults to sibling cached_responses.json

    PROMPT = "Describe this scene in one sentence."

    # ── 3. No API key — use cache immediately ────────────────────────────────
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Using cached response (no API call).")
        print("Add your key in the Setup section of the hub, or in a .env file.")
        print()
        # Pick a random cached description and print it
        print(random.choice(list(cache.values())))
        return

    # ── 4. Capture one camera frame ──────────────────────────────────────────
    cap = open_camera()
    ret, frame = cap.read()
    cap.release()  # release the camera immediately — we only need one frame

    if not ret:
        print("Warning: Could not read a frame from the camera. Using cached response.")
        print(random.choice(list(cache.values())))
        return

    # Convert BGR (OpenCV's format) → RGB (what PIL and Gemini expect)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ── 5. Call Gemini API ────────────────────────────────────────────────────
    # Deferred import: google.genai is only imported here, not at module load.
    # This means the script can still run (using the cache) even if google-genai
    # is not installed.
    from google import genai

    client = genai.Client(api_key=api_key)

    # generate_content() sends both the text prompt and the image to the model.
    # gemini-3.1-flash-lite is the fastest and cheapest Gemini model (Sep 2026).
    # It still handles vision tasks well for simple scene descriptions.
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[PROMPT, pil_image],
    )
    # response.text is the model's plain-text reply
    print(response.text)


if __name__ == "__main__":
    main()
