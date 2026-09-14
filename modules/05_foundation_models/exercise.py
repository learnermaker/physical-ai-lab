#!/usr/bin/env python3
"""
Module 5 Exercise: Design Your Own Robot Prompt
=================================================
ARIA's prompt (in 02_gemini_robot_brain.py) defines a full JSON schema:

    {
      "action":      "APPROACH|WAIT|ALERT|RETREAT",
      "confidence":  0.0,
      "observation": "what was seen",
      "reason":      "why this action",
      "next_step":   "what ARIA does next"
    }

Your task: write a DIFFERENT system prompt for a DIFFERENT robot.
Try things like:
  - A warehouse picking robot: PICK / PLACE / NAVIGATE / WAIT
  - A surgical assistant:      HAND_TOOL / RETRACT / HOLD / WAIT
  - A home companion:          GREET / REMIND / ASSIST / STANDBY
  - Joint torques instead of named actions

Remember: the prompt IS the robot's brain.
Changing it changes what the robot can do — without any code changes or retraining.
This is exactly how production VLA systems like RT-2 and π0 are programmed.

Start here: Find the TODO block below (~line 165) and define MY_PROMPT.

Expected output (varies with your prompt and camera scene):
    Calling Gemini...
    Done.
    action: NAVIGATE
    confidence: 0.84
    observation: A clear path is visible in the workspace
    reason: No items to pick and the aisle is clear — robot should reposition
    next_step: Move to the next waypoint at row B, slot 7

No API key?  The cache fallback runs automatically and a cached response is used.
The cache may not exactly match your custom prompt — that's intentional.
Notice what the model returns when the prompt doesn't match the cache.

Run from the repo root:
    python modules/05_foundation_models/exercise.py

See the full ARIA prompt in:  modules/05_foundation_models/02_gemini_robot_brain.py
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
# Default prompt — ARIA schema (same as 02_gemini_robot_brain.py)
# Replace this entirely in the TODO block below
# ---------------------------------------------------------------------------
_DEFAULT_PROMPT = (
    "You are ARIA, an assistive robot in a workspace. Your job is to observe people "
    "and decide if they need help. Look at the person in this image. "
    "Respond ONLY in JSON with no markdown: "
    '{"action": "APPROACH|WAIT|ALERT|RETREAT", '
    '"confidence": 0.0, '
    '"observation": "one sentence describing what you see", '
    '"reason": "one sentence explaining your action choice", '
    '"next_step": "one sentence describing what ARIA does next"} '
    "— APPROACH if they seem confused; WAIT if calm; "
    "ALERT if unwell; RETREAT if leaving."
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
            model="gemini-3.1-flash-lite",
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

    # TODO START — Write your own prompt for a different robot ─────────────
    #
    # ARIA's schema (in 02_gemini_robot_brain.py) is:
    #   {action, confidence, observation, reason, next_step}
    #
    # Your task: define a prompt for a DIFFERENT kind of robot.
    # Example — warehouse picking robot:
    #
    #   MY_PROMPT = (
    #       'You are a warehouse picking robot. Look at this image of the workspace. '
    #       'Decide what to do next. '
    #       'Respond ONLY in JSON with no markdown: '
    #       '{"action": "PICK|PLACE|NAVIGATE|WAIT", '
    #       '"confidence": 0.0, '
    #       '"observation": "what you see in the image", '
    #       '"reason": "why this action", '
    #       '"next_step": "specific movement or task to execute"}'
    #   )
    #
    # Then call ask_gemini() and print the result:
    #   result = ask_gemini(MY_PROMPT, frame, api_key, cache)
    #   for key, value in result.items():
    #       print(f"{key}: {value}")
    #
    # TIP: Always include "Respond ONLY in JSON with no markdown:" or the
    #      json.loads() parser will fail on markdown-wrapped responses.
    raise NotImplementedError(
        "Define MY_PROMPT in the TODO block and call ask_gemini().\n"
        "See modules/05_foundation_models/02_gemini_robot_brain.py for the ARIA example."
    )
    # TODO END ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main() -> None:
#     load_dotenv(REPO_ROOT / ".env")
#     api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
#     cache = load_cache()
#     cap = open_camera()
#     ret, frame = cap.read()
#     cap.release()
#     if not ret or frame is None:
#         frame = None
#
#     # Warehouse picking robot — different robot, different vocabulary, same pipeline
#     MY_PROMPT = (
#         'You are a warehouse picking robot. Look at this image of the workspace. '
#         'Decide what to do next based on what you see. '
#         'Respond ONLY in JSON with no markdown: '
#         '{"action": "PICK|PLACE|NAVIGATE|WAIT", '
#         '"confidence": 0.0, '
#         '"observation": "one sentence describing what you see in the workspace", '
#         '"reason": "one sentence explaining why you chose this action", '
#         '"next_step": "specific movement or task to execute next"}'
#     )
#
#     result = ask_gemini(MY_PROMPT, frame, api_key, cache)
#     for key, value in result.items():
#         print(f"{key}: {value}")
