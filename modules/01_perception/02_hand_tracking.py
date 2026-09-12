#!/usr/bin/env python3
"""
Module 1 Demo: Hand Tracking
Detects up to two hands in the webcam feed using the MediaPipe Tasks API
(HandLandmarker in LIVE_STREAM mode).  Landmark positions are already
normalised to [0.0, 1.0] by the Tasks API — no pixel division required.

Landmark overlay is drawn with OpenCV directly, using the 21-point skeleton
defined in mp.solutions.hands.HAND_CONNECTIONS.

Press 'q' to quit.
"""

# Uses MediaPipe Tasks API: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

import sys
import threading
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/01_perception/
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

# Landmark drawing style
_LANDMARK_RADIUS = 5
_LANDMARK_COLOR = (0, 255, 0)       # green dots
_CONNECTION_COLOR = (255, 255, 255)  # white skeleton lines
_LANDMARK_THICKNESS = -1            # filled circle
_CONNECTION_THICKNESS = 2


# ---------------------------------------------------------------------------
# Result callback
# ---------------------------------------------------------------------------

# results_holder[0] is updated from the MediaPipe callback thread;
# the main loop reads it each frame.  A threading.Lock protects the write.
_results_holder: list = [None]
_results_lock = threading.Lock()


def _on_result(
    result: mp.tasks.vision.HandLandmarkerResult,
    output_image: mp.Image,   # noqa: ARG001 — unused but required by the callback signature
    timestamp_ms: int,         # noqa: ARG001
) -> None:
    """Store the latest HandLandmarker result for the main loop to consume."""
    with _results_lock:
        _results_holder[0] = result


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def _draw_landmarks(frame: "cv2.Mat", result: mp.tasks.vision.HandLandmarkerResult) -> None:
    """
    Draw all detected hand landmarks and skeleton connections onto *frame*
    (in-place, BGR).

    Normalised landmark coordinates (.x, .y in [0,1]) are multiplied by the
    frame dimensions to obtain pixel positions.
    """
    h, w = frame.shape[:2]

    # mp.solutions.hands.HAND_CONNECTIONS is a frozenset of (start_idx, end_idx) tuples
    connections = mp.solutions.hands.HAND_CONNECTIONS

    for hand_landmarks in result.hand_landmarks:
        # Convert normalised to pixel coordinates for this hand
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        # Draw skeleton connections first (underneath the dots)
        for start_idx, end_idx in connections:
            cv2.line(frame, pts[start_idx], pts[end_idx], _CONNECTION_COLOR, _CONNECTION_THICKNESS)

        # Draw landmark dots on top
        for pt in pts:
            cv2.circle(frame, pt, _LANDMARK_RADIUS, _LANDMARK_COLOR, _LANDMARK_THICKNESS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Verify the model file exists before handing it to MediaPipe
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] HandLandmarker model not found: {MODEL_PATH}", file=sys.stderr)
        print("        Run setup.bat (or setup.sh) to download it.", file=sys.stderr)
        sys.exit(1)

    # Open webcam, falling back to looping video if unavailable
    cap = open_camera()

    # Configure HandLandmarker — Tasks API, LIVE_STREAM mode
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_on_result,
    )

    timestamp_ms = 0  # monotonically increasing timestamp sent with each frame

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR frame to MediaPipe Image (RGB) and send for async detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 1

            # Read the latest result (may be None on the first few frames)
            with _results_lock:
                result = _results_holder[0]

            if result is not None and result.hand_landmarks:
                _draw_landmarks(frame, result)

            cv2.imshow("Hand Tracking — press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
