#!/usr/bin/env python3
"""
Module 5 Demo 2: Gemini Robot Brain
======================================
Builds on 01_gemini_vision.py by asking Gemini for a *structured robot action*
instead of a free-text description.

The key idea: by specifying the exact JSON format in the prompt, we constrain
the model to return something our code can parse and act on.  This structured
output format is called a "schema" — the model must fill it in correctly or
the parser will reject it and fall back to the cache.

The prompt acts as the interface between the LLM and the robot's hardware:

  SYSTEM PROMPT → defines action vocabulary + output format
  IMAGE INPUT   → provides the visual context
  JSON OUTPUT   → {"action": "LEFT|RIGHT|FORWARD|BACK|WAIT", "reason": "..."}
                   ↓
              robot executes the action

What you will see:
  - "Calling Gemini..." while the API call runs
  - "Done." followed by the action and reason from the model
  - Or "[Fallback] Using cached response." if the API call fails

No API key? The cache fallback runs automatically.

How it connects to the pipeline:
  This is the complete VLA (Vision-Language-Action) pattern.
  Next: exercise.py — write your own prompt and action vocabulary.

Run from the repo root:
  python modules/05_foundation_models/02_gemini_robot_brain.py

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
# ARIA — Assistive Robot for Intelligent Awareness
# ---------------------------------------------------------------------------
# ARIA observes people in a workspace and decides what to do.
# This prompt IS the robot's behaviour policy — change it, change the robot.
#
# Key design choices:
#   1. "Respond ONLY in JSON with no markdown" — keeps json.loads() reliable.
#   2. Four actions constrain the space to what a care robot actually does.
#   3. The inline guidance (APPROACH if confused...) ensures Gemini reasons
#      about human states, not just object geometry.
#   4. Real photos of people with no labels — Gemini must read body language.
SYSTEM_PROMPT = (
    "You are ARIA, an assistive robot in a workspace. Your job is to observe people "
    "and decide if they need help. Look at the person in this image. "
    "Respond ONLY in JSON with no markdown: "
    '{"action": "APPROACH|WAIT|ALERT|RETREAT", "reason": "one sentence"} '
    "— APPROACH if they seem confused, stuck, or are inviting interaction; "
    "WAIT if they are working calmly and do not need help; "
    "ALERT if they appear unwell, distressed, or unresponsive; "
    "RETREAT if they are leaving or clearly want space."
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

        # ── Parse the JSON response ──────────────────────────────────────────
        # The model should return something like:
        #   {"action": "FORWARD", "reason": "Clear path ahead"}
        # But it sometimes wraps it in markdown fences (```json ... ```) —
        # we strip those first.
        text = response.text.strip()
        if text.startswith("```"):
            # Strip the opening ``` or ```json
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # json.loads() converts the JSON string into a Python dict
        result = json.loads(text)

        # Validate: the response must contain the "action" key
        # A missing "action" means the model ignored our format — fall back
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
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # This script implements the complete VLA (Vision-Language-Action) pattern:
    # image → prompt → structured JSON → robot action.
    # The SYSTEM_PROMPT below IS the robot's action vocabulary. Change it and
    # you change what the robot can do — no retraining, no code changes.
    # Production systems (RT-2, OpenVLA, π0) use the same pattern at scale.
    # ─────────────────────────────────────────────────────────────────────────
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
