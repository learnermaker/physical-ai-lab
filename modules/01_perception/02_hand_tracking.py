#!/usr/bin/env python3
"""
Module 1 Demo 2: Hand Tracking
================================
Builds on 01_webcam_basics.py by adding MediaPipe hand detection.

MediaPipe's HandLandmarker finds up to 21 points on each hand — fingertips,
knuckles, and the wrist — and returns their (x, y) positions normalised to
[0.0, 1.0].  This means (0, 0) is the top-left of the frame and (1, 1) is
the bottom-right, regardless of camera resolution.

What you will see:
  - Live camera feed
  - Green dots on each hand landmark
  - White lines connecting landmarks into a skeleton

How it connects to the pipeline:
  Webcam → raw frames → [THIS FILE] → landmark positions → next: 03_joint_angles.py

Press 'q' to quit.
"""

# Uses MediaPipe Tasks API:
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

import sys
import threading
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

# Path to the downloaded hand landmark model file (8 MB).
# Run setup.bat / download_model.py if this file is missing.
MODEL_PATH = str(REPO_ROOT / "assets" / "hand_landmarker.task")

# Visual style for drawing landmarks on the frame
_LANDMARK_RADIUS = 5
_LANDMARK_COLOR = (0, 255, 0)       # green in BGR (Blue=0, Green=255, Red=0)
_CONNECTION_COLOR = (255, 255, 255)  # white
_LANDMARK_THICKNESS = -1            # -1 = filled circle
_CONNECTION_THICKNESS = 2           # line thickness in pixels


# ---------------------------------------------------------------------------
# Thread-safe result storage
# ---------------------------------------------------------------------------
# MediaPipe runs detection on a background thread and calls _on_result()
# when a frame is processed.  We store the latest result in a list so the
# main loop can read it safely using a Lock.
#
# Why a list?  Python's basic types (int, None) aren't directly thread-safe
# to reassign.  A single-element list [result] lets us replace _results_holder[0]
# safely with the Lock protecting the write.
_results_holder: list = [None]
_results_lock = threading.Lock()  # prevents simultaneous read/write


def _on_result(
    result: mp.tasks.vision.HandLandmarkerResult,
    output_image: mp.Image,   # required by MediaPipe callback signature but unused here
    timestamp_ms: int,         # required by MediaPipe callback signature but unused here
) -> None:
    """
    Called by MediaPipe on its internal thread each time a frame is processed.

    We just store the result so the main loop can draw it on the next frame.
    The Lock ensures the main thread doesn't read a half-written result.
    """
    with _results_lock:
        _results_holder[0] = result


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def _draw_landmarks(frame: "cv2.Mat", result: mp.tasks.vision.HandLandmarkerResult) -> None:
    """
    Draw all detected hand landmarks and skeleton connections onto *frame*.

    MediaPipe returns landmark coordinates normalised to [0, 1].
    We convert them to pixel coordinates by multiplying by frame width/height.

    Example: lm.x = 0.5, frame width = 640 → pixel x = 320 (centre of frame)
    """
    h, w = frame.shape[:2]  # height, width in pixels

    # HAND_CONNECTIONS is a list of (start_index, end_index) pairs defining
    # which landmarks to connect with lines (the skeleton structure).
    connections = mp.solutions.hands.HAND_CONNECTIONS

    for hand_landmarks in result.hand_landmarks:
        # Convert all 21 normalised landmarks to pixel coordinates
        pts = [
            (int(lm.x * w), int(lm.y * h))  # (pixel_x, pixel_y)
            for lm in hand_landmarks
        ]

        # Draw skeleton lines first so dots appear on top
        for start_idx, end_idx in connections:
            cv2.line(frame, pts[start_idx], pts[end_idx],
                     _CONNECTION_COLOR, _CONNECTION_THICKNESS)

        # Draw a filled circle at each of the 21 landmark positions
        for pt in pts:
            cv2.circle(frame, pt, _LANDMARK_RADIUS,
                       _LANDMARK_COLOR, _LANDMARK_THICKNESS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # Raw pixels are useless for control — there are too many of them and they
    # change with lighting. MediaPipe reduces each frame to 21 (x,y) points:
    # the skeleton of the hand. This structured representation is what the
    # controller can act on. The same "perception → structured features" step
    # appears in every physical AI system: factory cameras → bounding boxes,
    # depth cameras → point clouds, IMUs → pose estimates.
    # ─────────────────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] HandLandmarker model not found: {MODEL_PATH}", file=sys.stderr)
        print("        Run setup.bat (or setup.sh) to download it.", file=sys.stderr)
        sys.exit(1)

    cap = open_camera()

    # --- Configure the HandLandmarker ---
    # BaseOptions: points MediaPipe to the model file on disk.
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    # HandLandmarkerOptions sets the detection mode and parameters.
    # LIVE_STREAM mode: frames are submitted asynchronously (detect_async).
    #   - MediaPipe processes frames as fast as the model allows.
    #   - Results arrive via the result_callback (on a background thread).
    #   - This is the right mode for real-time camera feeds.
    # num_hands=2: detect up to 2 hands simultaneously.
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_on_result,
    )

    # timestamp_ms must increase monotonically with each frame.
    # MediaPipe uses this to keep results in order and detect dropped frames.
    timestamp_ms = 0

    # Using `with` ensures the landmarker is properly closed when we exit,
    # releasing the model from memory even if an exception occurs.
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB images, but OpenCV gives us BGR.
            # cvtColor converts between the two colour orderings.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # mp.Image wraps the numpy array in MediaPipe's image type.
            # SRGB is the standard colour space for camera images.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # detect_async submits the frame for processing and returns immediately.
            # The result will arrive via _on_result() on a background thread.
            # We increment timestamp_ms by 1 each frame — good enough for ordering.
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 1

            # Read the most recent result (may be None on the first few frames
            # while the model is warming up)
            with _results_lock:
                result = _results_holder[0]

            # Draw landmarks if we have a result and at least one hand was found
            if result is not None and result.hand_landmarks:
                _draw_landmarks(frame, result)

            cv2.imshow("Hand Tracking — press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
