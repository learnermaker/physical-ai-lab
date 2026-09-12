#!/usr/bin/env python3
"""
Module 1 Exercise: Joint Angle State Vector (4-element)
Expected output: a 4-element list printed to stdout once per second,
e.g. [145.2, 163.8, 171.4, 168.9]

Your task: extend compute_state_vector() to include the ring finger MCP angle
so the returned state vector has four elements instead of three.
"""

# Uses MediaPipe Tasks API: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

import math
import queue
import sys
import time
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

# MediaPipe drawing helpers (still available in legacy solutions namespace)
_mp_drawing = mp.solutions.drawing_utils
_mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def _compute_angle(a: tuple, vertex: tuple, b: tuple) -> float:
    """
    Compute the angle at *vertex* formed by the vectors vertex→a and vertex→b,
    returning the result in degrees.

    Uses the dot-product / acos formula.  The cosine value is clamped to
    [-1, 1] before acos to prevent a domain error from floating-point
    rounding that can push the value fractionally outside that range.

    Parameters
    ----------
    a, vertex, b : tuple of (x, y)
        Normalised 2-D landmark coordinates (z is ignored for the angle).
    """
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    bx, by = b[0] - vertex[0], b[1] - vertex[1]

    dot = ax * bx + ay * by
    mag_a = math.sqrt(ax * ax + ay * ay)
    mag_b = math.sqrt(bx * bx + by * by)

    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0  # degenerate case: landmark overlap

    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_theta))


def compute_state_vector(landmarks: list) -> list[float]:
    """
    Compute angles for thumb, index, middle, and ring finger MCP joints.

    Angle definitions (all angles measured at the MCP/intermediate joint):
      θ_thumb : angle at lm1  (THUMB_CMC)       between lm0→lm1  and lm1→lm2
      θ_index : angle at lm5  (INDEX_MCP)        between lm0→lm5  and lm5→lm6
      θ_middle: angle at lm9  (MIDDLE_MCP)       between lm0→lm9  and lm9→lm10
      θ_ring  : angle at lm13 (RING_FINGER_MCP)  between lm0→lm13 and lm13→lm14

    Parameters
    ----------
    landmarks : list of NormalizedLandmark (21 elements, index 0..20)

    Returns
    -------
    list of four floats in degrees [θ_thumb, θ_index, θ_middle, θ_ring]
    """
    def lm(i):
        return (landmarks[i].x, landmarks[i].y)

    theta_thumb  = _compute_angle(lm(0), lm(1),  lm(2))
    theta_index  = _compute_angle(lm(0), lm(5),  lm(6))
    theta_middle = _compute_angle(lm(0), lm(9),  lm(10))

    # TODO START — Compute the ring finger MCP angle and append it to the state vector
    raise NotImplementedError(
        "Implement compute_state_vector at line 98. "
        "See # EXPECTED OUTPUT comment below."
    )
    # TODO END
    # SOLUTION HINT: Follow the same pattern used for the middle finger above.
    # The ring finger MCP joint is landmark 13; the landmarks on either side of
    # that joint are landmark 0 (wrist) and landmark 14 (ring finger PIP joint).
    # Call _compute_angle with those three landmark positions and add the result
    # as a fourth element in the returned list.

    # EXPECTED OUTPUT: [145.2, 163.8, 171.4, 168.9]
    # (exact values vary with hand position; all four elements are floats in degrees)

    return [theta_thumb, theta_index, theta_middle]


# ---------------------------------------------------------------------------
# MediaPipe LIVE_STREAM callback
# ---------------------------------------------------------------------------

# Queue used to pass the latest landmark result from the async callback
# to the main loop without blocking either side.
_result_queue: queue.Queue = queue.Queue(maxsize=1)


def _on_result(
    result: mp_vision.HandLandmarkerResult,
    output_image: mp.Image,  # noqa: F841  (unused but required by signature)
    timestamp_ms: int,       # noqa: F841
) -> None:
    """Callback invoked by MediaPipe on the detection thread."""
    # Keep only the freshest result; discard stale values.
    try:
        _result_queue.get_nowait()
    except queue.Empty:
        pass
    _result_queue.put_nowait(result)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Open camera (with automatic fallback to looping video) ---
    cap = open_camera()

    # --- Build the HandLandmarker ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # --- State ---
    last_result: mp_vision.HandLandmarkerResult | None = None
    last_print_time: float = 0.0
    frame_index: int = 0

    print("Running — press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR frame to an mp.Image and send for async detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000) + frame_index  # must be monotonically increasing
        landmarker.detect_async(mp_image, timestamp_ms)
        frame_index += 1

        # Drain the result queue (non-blocking)
        try:
            last_result = _result_queue.get_nowait()
        except queue.Empty:
            pass

        # --- Draw landmark overlay ---
        if last_result and last_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in last_result.hand_landmarks:
                # Convert normalised landmarks to a legacy NormalizedLandmarkList
                # so drawing_utils can render them with standard connectors.
                landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                landmark_list.landmark.extend([
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z
                    )
                    for lm in hand_landmarks
                ])
                _mp_drawing.draw_landmarks(
                    frame,
                    landmark_list,
                    _mp_hands_connections,
                )

            # --- Print state vector once per second ---
            now = time.time()
            if now - last_print_time >= 1.0:
                # Use the first detected hand
                angles = compute_state_vector(last_result.hand_landmarks[0])
                print(f"[{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}]")
                last_print_time = now

        cv2.imshow("Joint Angles — press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def compute_state_vector(landmarks: list) -> list[float]:
#     def lm(i):
#         return (landmarks[i].x, landmarks[i].y)
#
#     theta_thumb  = _compute_angle(lm(0), lm(1),  lm(2))
#     theta_index  = _compute_angle(lm(0), lm(5),  lm(6))
#     theta_middle = _compute_angle(lm(0), lm(9),  lm(10))
#     theta_ring   = _compute_angle(lm(0), lm(13), lm(14))
#
#     return [theta_thumb, theta_index, theta_middle, theta_ring]
