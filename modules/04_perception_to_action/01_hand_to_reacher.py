#!/usr/bin/env python3
"""
Module 4 Demo: Hand-to-Reacher Teleoperation
Maps the index finger tip position (MediaPipe landmark 8) to joint torque
commands for the Gymnasium Reacher-v5 environment in real time.

Landmark coordinates from the Tasks API are already normalised to [0.0, 1.0].
These are linearly mapped to the Reacher action space [-1.0, 1.0] via
``scale_landmark_to_action``.

Press 'q' to quit.

Pedagogical note: this teleoperation pattern is exactly how demonstration
data is collected for imitation learning (Behaviour Cloning, ACT, Diffusion
Policy) — the connection made explicit in Module 6.
"""

# Uses MediaPipe Tasks API:
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

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
    """Map a normalised landmark coordinate [0, 1] to action space [-1, 1].

    Linear map:  f(x) = 2*x - 1

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

    # ------------------------------------------------------------------
    # Open camera — falls back transparently to assets/fallback_hand_demo.mp4
    # RuntimeError means neither webcam nor fallback video could be opened.
    # ------------------------------------------------------------------
    try:
        cap = open_camera()
    except RuntimeError as err:
        print(f"[ERROR] Could not open camera or fallback video: {err}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create Reacher-v5 environment
    # make_env handles the render fallback chain: human → rgb_array → None
    # ------------------------------------------------------------------
    env = make_env("Reacher-v5", render_mode="human")
    obs, _ = env.reset()

    # Detect which render mode we actually got (determines display strategy)
    actual_render_mode = env.unwrapped.render_mode

    # ------------------------------------------------------------------
    # Configure HandLandmarker (Tasks API, LIVE_STREAM mode)
    # ------------------------------------------------------------------
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_on_result,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    timestamp_ms = 0
    last_hand_time = time.monotonic()  # tracks when a hand was last detected
    action = np.zeros(2, dtype=np.float32)

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # --- Read frame ---
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # --- Send frame to HandLandmarker (async) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 1

            # --- Consume latest landmark result ---
            try:
                result = _result_queue.get_nowait()
            except queue.Empty:
                result = None

            no_hand_text = None

            if result is not None and result.hand_landmarks:
                # Hand detected — use index finger tip (landmark 8)
                lm8 = result.hand_landmarks[0][8]
                norm_x, norm_y = lm8.x, lm8.y

                action = np.array(
                    [scale_landmark_to_action(norm_x), scale_landmark_to_action(norm_y)],
                    dtype=np.float32,
                )
                last_hand_time = time.monotonic()

                # Draw landmarks onto the camera frame
                _draw_landmarks(frame, result)
            else:
                # No result yet, or result has no hands
                if time.monotonic() - last_hand_time > _NO_HAND_TIMEOUT_S:
                    action = np.zeros(2, dtype=np.float32)
                    no_hand_text = "No hand detected \u2014 holding position"

            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
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
