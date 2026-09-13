#!/usr/bin/env python3
"""
Module 4 Exercise: Add a Second Landmark (Thumb Tip)
======================================================
In 01_hand_to_reacher.py, only landmark 8 (index fingertip) drives the arm.
That gives you control over 2 joints with 2 values (x, y of one fingertip).

Your task: add landmark 4 (thumb tip) as a third control channel, producing
a 3-element action vector:
  action[0] = scale_landmark_to_action(lm8.x)   ← index fingertip x
  action[1] = scale_landmark_to_action(lm8.y)   ← index fingertip y
  action[2] = scale_landmark_to_action(lm4.x)   ← thumb tip x  ← YOU ADD THIS

The ActionSliceWrapper (already set up below) strips action[2] before
sending to Reacher-v5, which only accepts 2 values.  Your extra dimension
is visible in the on-screen overlay but doesn't damage the environment.

Why add more fingers?
  Real robot arms have 6–7 degrees of freedom.  Each additional landmark
  you map gives you one more control channel.  This exercise is one step
  toward full-hand teleoperation.

Start here:  Find the TODO block below (~line 210) and add 3 lines of code.
             The solution is commented out at the bottom of this file.

Expected output: overlay shows three action values, e.g.
  Actions: [-0.32,  0.14,  0.76]

Press 'q' to quit.

Run from the repo root:
  python modules/04_perception_to_action/exercise.py

Uses MediaPipe Tasks API:
  https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)
"""

import queue
import sys
import time
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
# This file lives at modules/04_perception_to_action/exercise.py
# → parents[0] = modules/04_perception_to_action/
# → parents[1] = modules/
# → parents[2] = <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import gymnasium as gym
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.camera import open_camera
from utils.gym_utils import make_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = str(REPO_ROOT / "assets" / "hand_landmarker.task")

_NO_HAND_TIMEOUT_S = 1.0

_LANDMARK_RADIUS = 5
_LANDMARK_COLOR = (0, 255, 0)
_CONNECTION_COLOR = (255, 255, 255)
_CONNECTION_THICKNESS = 2


# ---------------------------------------------------------------------------
# scale_landmark_to_action — provided, no changes needed
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
# ActionSliceWrapper — provided, no changes needed
# ---------------------------------------------------------------------------

class ActionSliceWrapper(gym.ActionWrapper):
    """
    Accept an action of any length and pass only the first ``n`` elements
    to the wrapped environment.

    Why do we need this?
      Reacher-v5 expects exactly 2 torque values.
      Our exercise sends 3 values (index x, index y, thumb x).
      This wrapper silently drops the extra dimension so Reacher doesn't crash.

      In a real system you might use a learned projection instead of slicing,
      but for learning purposes this is the clearest approach.

    Parameters
    ----------
    env : The Reacher-v5 environment to wrap.
    n   : Number of action dimensions the inner environment expects (2 for Reacher).
    """

    def __init__(self, env: gym.Env, n: int) -> None:
        super().__init__(env)
        self._n = n

    def action(self, action: np.ndarray) -> np.ndarray:
        # Keep only the first n elements — discard the rest
        return np.asarray(action, dtype=np.float32)[: self._n]


# ---------------------------------------------------------------------------
# Async landmark callback and result queue
# ---------------------------------------------------------------------------

_result_queue: queue.Queue = queue.Queue(maxsize=1)


def _on_result(
    result: mp.tasks.vision.HandLandmarkerResult,
    output_image: mp.Image,   # noqa: ARG001
    timestamp_ms: int,         # noqa: ARG001
) -> None:
    """Place the latest HandLandmarker result on the single-slot queue."""
    try:
        _result_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        _result_queue.put_nowait(result)
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# Drawing helper — provided, no changes needed
# ---------------------------------------------------------------------------

def _draw_landmarks(
    frame: "cv2.Mat",
    result: mp.tasks.vision.HandLandmarkerResult,
) -> None:
    """Draw hand landmarks and skeleton connections onto *frame* (BGR, in-place)."""
    h, w = frame.shape[:2]
    connections = mp.solutions.hands.HAND_CONNECTIONS

    for hand_landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start_idx, end_idx in connections:
            cv2.line(frame, pts[start_idx], pts[end_idx], _CONNECTION_COLOR, _CONNECTION_THICKNESS)

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
    # Open camera
    # ------------------------------------------------------------------
    try:
        cap = open_camera()
    except RuntimeError as err:
        print(f"[ERROR] Could not open camera or fallback video: {err}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create Reacher-v5 and wrap it so it accepts a 3-element action
    # ------------------------------------------------------------------
    _base_env = make_env("Reacher-v5", render_mode="human")
    env = ActionSliceWrapper(_base_env, n=2)
    obs, _ = env.reset()

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
    # Main loop  — wrapped in try/finally so env/cap always close
    # ------------------------------------------------------------------
    timestamp_ms = 0
    last_hand_time = time.monotonic()
    # Default: 3-element zero action (landmark 8 x, landmark 8 y, landmark 4 x)
    action = np.zeros(3, dtype=np.float32)

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Send frame to HandLandmarker (async)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 1

            # Consume latest landmark result
            try:
                result = _result_queue.get_nowait()
            except queue.Empty:
                result = None

            no_hand_text = None

            if result is not None and result.hand_landmarks:
                lm8 = result.hand_landmarks[0][8]   # index finger tip  — provided

                # TODO START ────────────────────────────────────────────────
                # Add landmark 4 (thumb tip) as a third action dimension.
                #
                # Step 1: Get the thumb tip landmark (index 4)
                #   lm4 = result.hand_landmarks[0][4]
                #
                # Step 2: Scale its x coordinate to [-1, 1]
                #   lm4_action = scale_landmark_to_action(lm4.x)
                #
                # Step 3: Build the 3-element action array
                #   action = np.array([
                #       scale_landmark_to_action(lm8.x),  # joint 1 torque
                #       scale_landmark_to_action(lm8.y),  # joint 2 torque
                #       lm4_action,                       # YOUR new channel
                #   ], dtype=np.float32)
                raise NotImplementedError(
                    "Complete the TODO block starting at this line. "
                    "See # SOLUTION HINT below for guidance."
                )
                # TODO END ──────────────────────────────────────────────────
                # SOLUTION HINT: The ActionSliceWrapper automatically sends
                # only action[:2] to Reacher-v5.  Your action[2] appears in
                # the on-screen overlay but does not affect the simulation.
                # You only need to build the 3-element array.

                last_hand_time = time.monotonic()
                _draw_landmarks(frame, result)

            else:
                if time.monotonic() - last_hand_time > _NO_HAND_TIMEOUT_S:
                    action = np.zeros(3, dtype=np.float32)
                    no_hand_text = "No hand detected \u2014 holding position"

            # EXPECTED OUTPUT: action overlay shows three values, e.g.
            #   Actions: [-0.32,  0.14,  0.76]

            # Step the environment — ActionSliceWrapper forwards only action[:2]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

            # Overlay "no hand" message
            if no_hand_text is not None:
                cv2.putText(
                    frame,
                    no_hand_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Action overlay — shows all three components
            action_text = (
                f"Actions: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            )
            cv2.putText(
                frame,
                action_text,
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Module 4 Exercise \u2014 press q to quit", frame)

            if actual_render_mode == "rgb_array":
                rgb_render = env.render()
                if rgb_render is not None:
                    bgr_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Reacher-v5 Sim", bgr_render)
            elif actual_render_mode is None or actual_render_mode == "None":
                cv2.putText(
                    frame,
                    f"Env torques: [{action[0]:.2f}, {action[1]:.2f}]",
                    (10, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Module 4 Exercise \u2014 press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup always runs — even when NotImplementedError is raised
        env.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ==============================================================================
# SOLUTION (uncomment to run):
# ==============================================================================
# def main() -> None:
#     if not Path(MODEL_PATH).exists():
#         print(f"[ERROR] HandLandmarker model not found: {MODEL_PATH}", file=sys.stderr)
#         print("        Run setup.bat (or setup.sh) to download it.", file=sys.stderr)
#         sys.exit(1)
#
#     try:
#         cap = open_camera()
#     except RuntimeError as err:
#         print(f"[ERROR] Could not open camera or fallback video: {err}", file=sys.stderr)
#         sys.exit(1)
#
#     _base_env = make_env("Reacher-v5", render_mode="human")
#     env = ActionSliceWrapper(_base_env, n=2)
#     obs, _ = env.reset()
#     actual_render_mode = env.unwrapped.render_mode
#
#     base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = mp_vision.HandLandmarkerOptions(
#         base_options=base_options,
#         running_mode=mp_vision.RunningMode.LIVE_STREAM,
#         num_hands=2,
#         result_callback=_on_result,
#     )
#
#     timestamp_ms = 0
#     last_hand_time = time.monotonic()
#     action = np.zeros(3, dtype=np.float32)
#
#     with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
#         while True:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 break
#
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#             landmarker.detect_async(mp_image, timestamp_ms)
#             timestamp_ms += 1
#
#             try:
#                 result = _result_queue.get_nowait()
#             except queue.Empty:
#                 result = None
#
#             no_hand_text = None
#
#             if result is not None and result.hand_landmarks:
#                 lm8 = result.hand_landmarks[0][8]   # index finger tip
#
#                 # ── SOLUTION ────────────────────────────────────────────────
#                 lm4 = result.hand_landmarks[0][4]   # thumb tip
#                 lm4_action = scale_landmark_to_action(lm4.x)
#                 action = np.array(
#                     [
#                         scale_landmark_to_action(lm8.x),
#                         scale_landmark_to_action(lm8.y),
#                         lm4_action,
#                     ],
#                     dtype=np.float32,
#                 )
#                 # ── END SOLUTION ─────────────────────────────────────────────
#
#                 last_hand_time = time.monotonic()
#                 _draw_landmarks(frame, result)
#
#             else:
#                 if time.monotonic() - last_hand_time > _NO_HAND_TIMEOUT_S:
#                     action = np.zeros(3, dtype=np.float32)
#                     no_hand_text = "No hand detected \u2014 holding position"
#
#             obs, reward, terminated, truncated, info = env.step(action)
#             if terminated or truncated:
#                 obs, _ = env.reset()
#
#             if no_hand_text is not None:
#                 cv2.putText(frame, no_hand_text, (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
#
#             action_text = f"Actions: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
#             cv2.putText(frame, action_text, (10, frame.shape[0] - 15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
#
#             cv2.imshow("Module 4 Exercise \u2014 press q to quit", frame)
#
#             if actual_render_mode == "rgb_array":
#                 rgb_render = env.render()
#                 if rgb_render is not None:
#                     bgr_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
#                     cv2.imshow("Reacher-v5 Sim", bgr_render)
#             elif actual_render_mode is None or actual_render_mode == "None":
#                 cv2.putText(
#                     frame,
#                     f"Env torques: [{action[0]:.2f}, {action[1]:.2f}]",
#                     (10, frame.shape[0] - 45),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
#                 )
#                 cv2.imshow("Module 4 Exercise \u2014 press q to quit", frame)
#
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#
#     env.close()
#     cap.release()
#     cv2.destroyAllWindows()
