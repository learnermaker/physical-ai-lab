#!/usr/bin/env python3
"""
Module 5 Demo 2: Gemini Robot Brain
Captures a single frame from the camera, encodes it as a JPEG, and sends it
to Gemini with a structured prompt asking for a robot action decision.

Expected output:
  Calling Gemini...
  Done.
  action: FORWARD
  reason: Path ahead appears clear

If the API call fails for any reason, a cached response is used instead and
'[Fallback] Using cached response.' is printed in place of 'Done.'.

If cached_responses.json is missing or empty the script exits with code 1.

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
# Structured action prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    'You are a robot controller. Looking at this image, suggest an action. '
    'Respond ONLY in JSON with no markdown: '
    '{"action": "LEFT|RIGHT|FORWARD|BACK|WAIT", "reason": "one sentence"}'
)


# ---------------------------------------------------------------------------
# Core testable function (supports Property 8 in test_properties.py)
# ---------------------------------------------------------------------------
def get_robot_action(
    cache_path: Path | None = None,
    api_key: str | None = None,
    frame=None,
) -> dict:
    """
    Capture a frame (or use the supplied one), send it to Gemini, and return
    an action dict with at least the ``"action"`` key.

    On any exception (API error, JSON parse error, missing key) the function
    falls back to a random entry from the cache that contains ``"action"``.

    Args:
        cache_path: Explicit path to ``cached_responses.json``.  Defaults to
                    the sibling file when *None*.
        api_key:    Gemini API key.  When *None*, reads ``GEMINI_API_KEY``
                    from the environment.
        frame:      Pre-captured BGR frame (numpy array).  When *None* the
                    function opens the camera and captures one frame.

    Returns:
        dict with at least ``"action"`` key.

    Raises:
        SystemExit(1): if the cache is missing or empty (via ``load_cache``).
    """
    cache = load_cache(cache_path)

    # Action-type entries are the dict values that contain the "action" key.
    action_entries = [v for v in cache.values() if isinstance(v, dict) and "action" in v]
    if not action_entries:
        # Shouldn't happen with the shipped cache, but be safe.
        action_entries = [{"action": "WAIT", "reason": "No action entries in cache."}]

    resolved_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()

    if not resolved_key:
        print("Warning: GEMINI_API_KEY not set. Using cached response (no API call).")
        return random.choice(action_entries)

    # Capture a frame if none was supplied.
    if frame is None:
        cap = open_camera()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("Warning: Could not read a frame. Using cached response.")
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

    client = genai.Client(api_key=resolved_key)
    image_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")

    print("Calling Gemini...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[SYSTEM_PROMPT, image_part],
        )

        # Strip markdown fences if the model wrapped the JSON (Design § 4.3).
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)

        # Validate expected shape — must have at least "action".
        if "action" not in result:
            raise ValueError(f"Response missing 'action' key: {result}")

        print("Done.")
        return result

    except Exception as exc:
        print(f"Error calling Gemini: {exc}", file=sys.stderr)
        print("[Fallback] Using cached response.")
        return random.choice(action_entries)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # load_cache exits(1) if the file is missing or empty — satisfies Req 14.8
    cache = load_cache()

    action_entries = [v for v in cache.values() if isinstance(v, dict) and "action" in v]
    if not action_entries:
        action_entries = [{"action": "WAIT", "reason": "No action entries in cache."}]

    # No API key: warn and fall back immediately (no camera, no API call).
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Using cached response (no API call).")
        result = random.choice(action_entries)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    # Open camera and capture one frame.
    cap = open_camera()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Warning: Could not read a frame from the camera. Using cached response.")
        result = random.choice(action_entries)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    result = get_robot_action(api_key=api_key, frame=frame)

    # Print all top-level keys and values.
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
