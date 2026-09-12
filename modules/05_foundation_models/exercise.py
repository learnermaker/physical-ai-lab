#!/usr/bin/env python3
"""
Module 5 Exercise: Design Your Own Robot Prompt
================================================
In ``02_gemini_robot_brain.py`` the system prompt always asks for one of five
navigation actions (LEFT / RIGHT / FORWARD / BACK / WAIT).

Your task: write a *different* system prompt and observe how the model's
reasoning changes.  For example, you might ask it to:

  - Describe the objects it can see in the scene.
  - Rate the lighting quality of the image.
  - Suggest an emotion that matches the scene.

The ``ask_gemini`` helper is already implemented — you only need to fill in
the TODO block.

Expected output
---------------
Running the completed exercise should print every key-value pair from the
model response, for example::

    Calling Gemini...
    Done.
    action: INSPECT
    reason: Several objects are visible on the desk

(The exact values depend on your prompt and the camera scene.)

If no API key is available the cache fallback is used automatically and a
different cached entry may be printed.

Unmodified, the script raises ``NotImplementedError`` at the TODO block.

# Adapted from: https://github.com/google-gemini/cookbook (Apache-2.0)
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

    # TODO START — Write your own prompt and observe how the robot's reasoning changes
    #
    # Steps:
    #   1. Define a custom system prompt string, e.g.:
    #          MY_PROMPT = (
    #              'Describe the main objects you can see. '
    #              'Respond ONLY in JSON with no markdown: '
    #              '{"action": "INSPECT", "reason": "one sentence describing the objects"}'
    #          )
    #   2. Call ask_gemini with your prompt, the captured frame, the api_key,
    #      and the cache:
    #          result = ask_gemini(MY_PROMPT, frame, api_key, cache)
    #   3. Print each key-value pair from the result:
    #          for key, value in result.items():
    #              print(f"{key}: {value}")
    raise NotImplementedError(
        "Implement main() at the TODO block around line 174. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END
    # SOLUTION HINT: Define a string variable that describes what you want the
    # model to reason about (be specific — the more focused your prompt, the
    # more informative the response).  Pass that string as the first argument
    # to ask_gemini, then iterate over the returned dict and print each entry.
    # Do not pass executable Python expressions inside the prompt string.

    # EXPECTED OUTPUT: result
    # action: <value from your prompt>
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
