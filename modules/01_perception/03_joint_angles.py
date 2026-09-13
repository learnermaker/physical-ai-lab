#!/usr/bin/env python3
"""
Module 1 Demo 3: Joint Angle State Vector
==========================================
Builds on 02_hand_tracking.py by turning raw landmark positions into
meaningful numbers — joint angles in degrees.

Why angles?  A robot arm controller doesn't care about pixel coordinates
on a screen.  It cares about the *opening angle* at each knuckle — the same
information a servo motor encoder would provide on a real robotic hand.

This script computes the MCP (metacarpophalangeal) joint angle for the thumb,
index, and middle fingers and prints them as a 3-element state vector once
per second:  [θ_thumb, θ_index, θ_middle]

What you will see:
  - Live camera feed with landmark overlay
  - State vector printed to the terminal every second
    e.g. [145.2, 163.8, 171.4]

How it connects to the pipeline:
  Webcam → landmarks → [THIS FILE] → state vector → next: exercise.py (add ring finger)

Press 'q' to quit.
"""

# Uses MediaPipe Tasks API:
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

import math
import queue
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.camera import open_camera

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = str(REPO_ROOT / "assets" / "hand_landmarker.task")

# MediaPipe legacy drawing helpers — still the easiest way to draw the
# full 21-point hand skeleton with standard styling.
_mp_drawing = mp.solutions.drawing_utils
_mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def _compute_angle(a: tuple, vertex: tuple, b: tuple) -> float:
    """
    Compute the angle AT the vertex formed by points a, vertex, and b.

    Picture three points:
        a --------- vertex --------- b
    This function returns the angle at vertex (in degrees).

    Formula: θ = acos( (u · v) / (|u| × |v|) )
    where u = a - vertex and v = b - vertex are vectors from vertex to each end.

    The result is clamped to [0°, 180°] to handle floating-point rounding.

    For hand joint angles:
      - a     = wrist (landmark 0) — the fixed anchor
      - vertex = MCP joint (knuckle)
      - b     = PIP joint (next knuckle along the finger)
    So the angle measures how "open" the finger is at that knuckle.
    """
    # Compute vectors from vertex to each of the other two points
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    bx, by = b[0] - vertex[0], b[1] - vertex[1]

    # Dot product: measures how much the two vectors point in the same direction
    dot = ax * bx + ay * by

    # Magnitudes: length of each vector
    mag_a = math.sqrt(ax * ax + ay * ay)
    mag_b = math.sqrt(bx * bx + by * by)

    # Guard against zero-length vectors (happens when two landmarks overlap)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0

    # Clamp to [-1, 1] to prevent math domain errors from floating-point noise
    # (e.g. dot/(mag_a*mag_b) might be 1.0000001 due to rounding)
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))

    return math.degrees(math.acos(cos_theta))


def compute_state_vector(landmarks: list) -> list[float]:
    """
    Compute MCP joint angles for three fingers and return a 3-element state vector.

    Landmark indices used (see the hand diagram in the hub):
      Thumb:   lm0 (wrist) → lm1 (CMC) → lm2 (MCP)
      Index:   lm0 (wrist) → lm5 (MCP) → lm6 (PIP)
      Middle:  lm0 (wrist) → lm9 (MCP) → lm10 (PIP)

    Returns [θ_thumb, θ_index, θ_middle] in degrees.
    0° = fully closed (fist), ~180° = fully open (flat hand).
    """
    def lm(i):
        # Helper: return landmark i as a (x, y) tuple
        # Normalised coordinates in [0, 1] — no need to multiply by frame size
        # for angle computation (ratios cancel out)
        return (landmarks[i].x, landmarks[i].y)

    theta_thumb  = _compute_angle(lm(0), lm(1), lm(2))
    theta_index  = _compute_angle(lm(0), lm(5), lm(6))
    theta_middle = _compute_angle(lm(0), lm(9), lm(10))

    return [theta_thumb, theta_index, theta_middle]


# ---------------------------------------------------------------------------
# MediaPipe result queue
# ---------------------------------------------------------------------------
# We use a Queue instead of a plain variable because MediaPipe calls _on_result
# on its own thread, while the main loop reads results on the main thread.
# Queue.get_nowait() is thread-safe — no lock needed.
_result_queue: queue.Queue = queue.Queue(maxsize=1)
# maxsize=1: if the main loop is slower than detection, old results are discarded.


def _on_result(
    result: mp_vision.HandLandmarkerResult,
    output_image: mp.Image,  # required by callback signature, unused here
    timestamp_ms: int,       # required by callback signature, unused here
) -> None:
    """
    Called by MediaPipe each time a frame is processed.
    Puts the result in the queue, discarding any unread result first.
    """
    try:
        _result_queue.get_nowait()  # discard stale result if main loop is slow
    except queue.Empty:
        pass
    _result_queue.put_nowait(result)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = open_camera()

    # Configure HandLandmarker — same as 02_hand_tracking.py
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    last_result: mp_vision.HandLandmarkerResult | None = None
    last_print_time: float = 0.0   # time.time() of last state vector print
    frame_index: int = 0

    print("Running — press 'q' in the window to quit.")
    print("State vector format: [θ_thumb, θ_index, θ_middle] in degrees")
    print()
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # The 3-element array printed each second IS the state vector — the central
    # data structure of the pipeline. A robot gripper controller reads exactly
    # this kind of array (joint angles from encoders) to decide how to move.
    # In Module 4, these numbers will flow directly into a MuJoCo simulation
    # as joint torque commands. In Module 5, Gemini will reason about a camera
    # frame instead — but the state → action pattern stays the same.
    # ─────────────────────────────────────────────────────────────────────────

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Send frame to MediaPipe for async detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # timestamp_ms uses both wall-clock time and frame count to stay monotonic
        # even if the clock resolution is low
        timestamp_ms = int(time.time() * 1000) + frame_index
        landmarker.detect_async(mp_image, timestamp_ms)
        frame_index += 1

        # Try to get the latest result (non-blocking)
        try:
            last_result = _result_queue.get_nowait()
        except queue.Empty:
            pass  # keep using the previous result until a new one arrives

        # Draw landmarks if we have a detection
        if last_result and last_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in last_result.hand_landmarks:
                # Convert Tasks API NormalizedLandmark objects to the legacy
                # NormalizedLandmarkList format that drawing_utils expects
                landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                landmark_list.landmark.extend([
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z
                    )
                    for lm in hand_landmarks
                ])
                # draw_landmarks renders the full 21-point skeleton
                _mp_drawing.draw_landmarks(
                    frame,
                    landmark_list,
                    _mp_hands_connections,
                )

            # Print state vector once per second (avoids console flooding)
            now = time.time()
            if now - last_print_time >= 1.0:
                angles = compute_state_vector(last_result.hand_landmarks[0])
                # angles[0] = thumb, [1] = index, [2] = middle
                print(f"[{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}]")
                last_print_time = now

        cv2.imshow("Joint Angles — press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
