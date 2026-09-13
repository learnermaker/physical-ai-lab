#!/usr/bin/env python3
"""
Module 5 Exercise: Design Your Own Robot Prompt
=================================================
In 02_gemini_robot_brain.py the system prompt defines the action vocabulary:
  LEFT / RIGHT / FORWARD / BACK / WAIT

Your task: write a DIFFERENT system prompt and observe how the model's
reasoning changes.  Try things like:
  - A different action set (e.g. PICK_UP / PUT_DOWN / ROTATE)
  - Joint torques instead of directional commands
  - A different output schema (e.g. {"gesture": "...", "confidence": 0.9})

Remember: the prompt IS the interface between the LLM and the robot's hardware.
Changing it changes what the robot can do — without any code changes or retraining.

Start here:  Find the TODO block below (~line 165) and define MY_PROMPT,
             then call ask_gemini() with it.
             The solution is commented out at the bottom of this file.

Expected output (varies with your prompt and camera scene):
  Calling Gemini...
  Done.
  action: INSPECT
  reason: Several objects are visible on the desk

No API key?  The cache fallback runs automatically and a cached response is used.
The cache may not match your custom prompt — that's fine for understanding the structure.

Run from the repo root:
  python modules/05_foundation_models/exercise.py

Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)
"""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Python path convention (Design § 2.7)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Load load_cache from the package __init__ (the directory name starts with a
# digit, making a normal 'import' invalid Python syntax).
_init_path = Path(__file__).resolve().parent / "__init__.py"
_spec = importlib.util.spec_from_file_location("fm_init", _init_path)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)
load_cache = _fm.load_cache

from utils.camera import open_camera  # noqa: E402


# ---------------------------------------------------------------------------
# Default prompt (used only by the fallback path below — replace in the TODO)
# ---------------------------------------------------------------------------
_DEFAULT_PROMPT = (
    'You are a robot controller. Looking at this image, suggest an action. '
    'Respond ONLY in JSON with no markdown: '
    '{"action": "LEFT|RIGHT|FORWARD|BACK|WAIT", "reason": "one sentence"}'
)


# ---------------------------------------------------------------------------
# ask_gemini — fully implemented, no changes needed
# ---------------------------------------------------------------------------
def ask_gemini(
    prompt: str,
    frame,
    api_key: str,
    cache: dict,
) -> dict:
    """Send *frame* to Gemini with *prompt* and return the parsed response dict.

    Falls back to a random cache entry on any API error or JSON parse failure.

    Args:
        prompt:  The system prompt string to send.
        frame:   BGR numpy array (captured from the camera).
        api_key: Gemini API key string.  May be empty — triggers cache fallback.
        cache:   Dict returned by ``load_cache()``.

    Returns:
        dict — the parsed JSON response, or a random cached dict entry.
    """
    # Gather action-type entries for the fallback pool.
    action_entries = [
        v for v in cache.values() if isinstance(v, dict) and "action" in v
    ]
    if not action_entries:
        action_entries = [{"action": "WAIT", "reason": "No action entries in cache."}]

    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Using cached response (no API call).")
        return random.choice(action_entries)

    # Encode the frame as JPEG bytes.
    success, jpg_buf = cv2.imencode(".jpg", frame)
    if not success:
        print("Warning: Could not encode frame as JPEG. Using cached response.")
        return random.choice(action_entries)
    jpg_bytes = jpg_buf.tobytes()

    # Deferred import — keeps the offline fallback path importable without
    # google-genai installed.
    from google import genai  # noqa: E402
    from google.genai import types  # noqa: E402

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")

    print("Calling Gemini...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
        )

        # Strip markdown fences if the model wrapped the JSON (Design § 4.3).
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)

        if "action" not in result:
            raise ValueError(f"Response missing 'action' key: {result}")

        print("Done.")
        return result

    except Exception as exc:
        print(f"Error calling Gemini: {exc}", file=sys.stderr)
        print("[Fallback] Using cached response.")
        return random.choice(action_entries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # load_cache exits(1) if the file is missing or empty — satisfies Req 14.8
    cache = load_cache()

    # Open camera and capture one frame.
    cap = open_camera()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Warning: Could not read a frame. Using cached response.")
        action_entries = [
            v for v in cache.values() if isinstance(v, dict) and "action" in v
        ]
        frame = None  # ask_gemini handles None via the api_key-absent branch

    # TODO START — Write your own prompt and observe how the robot reasons ──
    #
    # Step 1: Define a custom system prompt string.
    #   Example:
    #     MY_PROMPT = (
    #         'Describe the main objects you can see. '
    #         'Respond ONLY in JSON with no markdown: '
    #         '{"action": "INSPECT", "reason": "one sentence describing the objects"}'
    #     )
    #   TIP: Always include "Respond ONLY in JSON with no markdown:" so the
    #        parser can read the response.  The JSON schema you define here
    #        IS your robot's action vocabulary.
    #
    # Step 2: Call ask_gemini() with your prompt:
    #   result = ask_gemini(MY_PROMPT, frame, api_key, cache)
    #
    # Step 3: Print each key-value pair from the result dict:
    #   for key, value in result.items():
    #       print(f"{key}: {value}")
    raise NotImplementedError(
        "Implement main() at the TODO block around line 164. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END ──────────────────────────────────────────────────────────────
    # EXPECTED OUTPUT (varies with your prompt):
    # action: <value from your prompt schema>
    # reason: <one-sentence explanation from the model>


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main() -> None:
#     load_dotenv(REPO_ROOT / ".env")
#     api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
#
#     cache = load_cache()
#
#     cap = open_camera()
#     ret, frame = cap.read()
#     cap.release()
#
#     if not ret or frame is None:
#         frame = None
#
#     MY_PROMPT = (
#         'Describe the main objects you can see in the scene. '
#         'Respond ONLY in JSON with no markdown: '
#         '{"action": "INSPECT", "reason": "one sentence describing the objects"}'
#     )
#
#     result = ask_gemini(MY_PROMPT, frame, api_key, cache)
#
#     for key, value in result.items():
#         print(f"{key}: {value}")
