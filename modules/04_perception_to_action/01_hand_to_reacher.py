#!/usr/bin/env python3
"""
Module 4 Demo: Hand-to-Reacher Teleoperation
==============================================
This is the full Physical AI pipeline closed loop:

  Your hand → MediaPipe landmarks → state vector → action → Reacher-v5 → repeat

You ARE the controller.  Move your index finger left/right to apply torque to
joint 1 (shoulder).  Move it up/down to apply torque to joint 2 (elbow).

How the mapping works:
  MediaPipe returns the index fingertip (landmark 8) as normalised coords [0, 1].
  The Reacher action space expects torques in [-1, 1].
  The mapping is a simple linear stretch:
    action = 2 * landmark_x - 1
  So finger at left edge (x=0) → torque=-1, centre (x=0.5) → 0, right (x=1) → +1.

This is called TELEOPERATION — a human directly controlling a robot's actuators
through a sensory interface.  The da Vinci surgical robot and warehouse picking
robots use exactly this pattern.  The same setup is also used to collect
demonstration datasets for training imitation-learning policies (BC, ACT, Diffusion Policy).

What you will see:
  - Camera window with hand landmarks overlaid
  - Reacher simulation driven by your finger in real time
  - Torque values printed in the camera window

Press 'q' to quit.

Run from the repo root:
  python modules/04_perception_to_action/01_hand_to_reacher.py

Uses MediaPipe Tasks API:
  https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)
"""

import queue
import sys
import time
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
# This file lives at modules/04_perception_to_action/01_hand_to_reacher.py
# → parents[0] = modules/04_perception_to_action/
# → parents[1] = modules/
# → parents[2] = <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.camera import open_camera
from utils.gym_utils import make_env

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL_PATH = str(REPO_ROOT / "assets" / "hand_landmarker.task")

# How long to wait without a detected hand before sending zero-vector action
_NO_HAND_TIMEOUT_S = 1.0

# Landmark drawing style (matches Module 1 conventions)
_LANDMARK_RADIUS = 5
_LANDMARK_COLOR = (0, 255, 0)        # green dots
_CONNECTION_COLOR = (255, 255, 255)  # white skeleton lines
_CONNECTION_THICKNESS = 2


# ---------------------------------------------------------------------------
# scale_landmark_to_action — defined at MODULE SCOPE for importability
# ---------------------------------------------------------------------------

def scale_landmark_to_action(x: float) -> float:
    """
    Map a normalised landmark coordinate [0, 1] to Reacher action space [-1, 1].

    Linear map:  f(x) = 2*x - 1

    Why this formula?
      MediaPipe returns positions in [0, 1] (fraction of frame width/height).
      Reacher expects torques in [-1, 1].
      A simple linear stretch maps the two ranges:
        x=0.0 (left edge)   → action = -1.0 (full torque left)
        x=0.5 (centre)      → action =  0.0 (no torque)
        x=1.0 (right edge)  → action = +1.0 (full torque right)

    This two-line "brain" is the entire intelligence of Module 4.
    In Module 5, Gemini replaces this formula.

    Examples
    --------
    >>> scale_landmark_to_action(0.0)
    -1.0
    >>> scale_landmark_to_action(0.5)
    0.0
    >>> scale_landmark_to_action(1.0)
    1.0
    """
    return 2.0 * x - 1.0


# ---------------------------------------------------------------------------
# Async callback and result queue
# ---------------------------------------------------------------------------

# The HandLandmarker callback fires on a MediaPipe-internal thread.
# Results are placed on a queue; the main loop drains it each frame.
_result_queue: queue.Queue = queue.Queue(maxsize=1)


def _on_result(
    result: mp.tasks.vision.HandLandmarkerResult,
    output_image: mp.Image,   # noqa: ARG001 — required by callback signature
    timestamp_ms: int,         # noqa: ARG001
) -> None:
    """Place the latest HandLandmarker result on the queue.

    ``maxsize=1`` ensures the queue never holds stale results: if the main
    loop is slower than detection, the older result is discarded.
    """
    # Discard an unread result rather than accumulating a backlog
    try:
        _result_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        _result_queue.put_nowait(result)
    except queue.Full:
        pass  # Should not occur after the get_nowait above, but safe to ignore


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def _draw_landmarks(
    frame: "cv2.Mat",
    result: mp.tasks.vision.HandLandmarkerResult,
) -> None:
    """Draw hand landmarks and skeleton connections onto *frame* (in-place, BGR)."""
    h, w = frame.shape[:2]
    connections = mp.solutions.hands.HAND_CONNECTIONS

    for hand_landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Connections first (rendered underneath the dots)
        for start_idx, end_idx in connections:
            cv2.line(
                frame,
                pts[start_idx],
                pts[end_idx],
                _CONNECTION_COLOR,
                _CONNECTION_THICKNESS,
            )

        # Landmark dots on top
        for pt in pts:
            cv2.circle(frame, pt, _LANDMARK_RADIUS, _LANDMARK_COLOR, -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Verify model file
    # ------------------------------------------------------------------
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] HandLandmarker model not found: {MODEL_PATH}", file=sys.stderr)
        print("        Run setup.bat (or setup.sh) to download it.", file=sys.stderr)
        sys.exit(1)

    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # This script closes the Physical AI loop: camera → landmark → action → sim.
    # You ARE the controller. The two-line mapping `action = 2*x - 1` is the
    # entire "brain" here. In Module 5, Gemini replaces those two lines.
    # In production (RT-2, π0), a 7B-parameter model replaces them.
    # The loop structure — perceive, decide, act, repeat — is unchanged.
    # ─────────────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Open camera — falls back to assets/fallback_hand_demo.mp4 if no webcam
    # ------------------------------------------------------------------
    try:
        cap = open_camera()
    except RuntimeError as err:
        print(f"[ERROR] Could not open camera or fallback video: {err}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create Reacher-v5
    # make_env() tries render_mode="human" (a window) first.
    # On Azure VDs and headless machines it falls back to "rgb_array" or None.
    # ------------------------------------------------------------------
    env = make_env("Reacher-v5", render_mode="human")
    obs, _ = env.reset()

    # actual_render_mode tells us what we actually got after the fallback chain
    actual_render_mode = env.unwrapped.render_mode

    # ------------------------------------------------------------------
    # Configure HandLandmarker (LIVE_STREAM mode — same as Module 1)
    # ------------------------------------------------------------------
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,       # detect up to 2 hands
        result_callback=_on_result,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    # timestamp_ms increments each frame — MediaPipe requires a monotonically
    # increasing timestamp to order async results correctly.
    timestamp_ms = 0
    last_hand_time = time.monotonic()  # tracks when a hand was last detected
    # Default action: zero torques (arm holds position) until a hand is found
    action = np.zeros(2, dtype=np.float32)

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # ── 1. Read camera frame ─────────────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # ── 2. Submit frame to MediaPipe (async) ─────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 1

            # ── 3. Read latest landmark result (non-blocking) ────────────────
            try:
                result = _result_queue.get_nowait()
            except queue.Empty:
                result = None

            no_hand_text = None

            if result is not None and result.hand_landmarks:
                # Hand detected — read landmark 8 (index fingertip)
                lm8 = result.hand_landmarks[0][8]
                norm_x, norm_y = lm8.x, lm8.y

                # ── THE MAPPING — these two lines are the robot's "brain" ────
                # scale_landmark_to_action maps [0,1] → [-1,1]
                # action[0] = torque for joint 1 (shoulder) — driven by x position
                # action[1] = torque for joint 2 (elbow)   — driven by y position
                action = np.array(
                    [scale_landmark_to_action(norm_x),
                     scale_landmark_to_action(norm_y)],
                    dtype=np.float32,
                )
                last_hand_time = time.monotonic()

                # Draw landmarks on the camera frame (visual feedback)
                _draw_landmarks(frame, result)
            else:
                # No hand detected for more than _NO_HAND_TIMEOUT_S seconds
                # → send zero torques so the arm doesn't drift
                if time.monotonic() - last_hand_time > _NO_HAND_TIMEOUT_S:
                    action = np.zeros(2, dtype=np.float32)
                    no_hand_text = "No hand detected \u2014 holding position"

            # ── 4. Step the simulation ───────────────────────────────────────
            # env.step() applies the torques for one simulation timestep (0.02 s)
            # and returns the new observation, reward, and whether episode ended.
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                # Episode ends after 50 steps — reset to a new random target position
                obs, _ = env.reset()

            # --- Display ---
            # Overlay "no hand" message if applicable
            if no_hand_text is not None:
                cv2.putText(
                    frame,
                    no_hand_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),  # orange
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Hand Tracking — press q to quit", frame)

            # Render the sim and decide secondary display strategy
            if actual_render_mode == "rgb_array":
                # Render returns an RGB numpy array — show it in a second window
                rgb_render = env.render()
                if rgb_render is not None:
                    bgr_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Reacher-v5 Sim", bgr_render)
            elif actual_render_mode is None or actual_render_mode == "None":
                # No visual render available — overlay numeric joint torques on
                # the camera frame instead
                torque_text = f"Torques: [{action[0]:.2f}, {action[1]:.2f}]"
                cv2.putText(
                    frame,
                    torque_text,
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),  # cyan
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Hand Tracking — press q to quit", frame)
            # else: render_mode="human" — sim renders in its own window automatically

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    env.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
