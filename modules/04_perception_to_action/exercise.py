#!/usr/bin/env python3
"""
Module 4 Exercise: Extend Hand-to-Reacher with a Second Landmark
================================================================
Building on ``01_hand_to_reacher.py``, this exercise adds a second landmark
to create a 3-element action vector.  A thin Gymnasium wrapper strips the
extra dimension before it reaches Reacher-v5, so the environment stays
compatible while you practice constructing richer action representations.

Expected output
---------------
The camera window shows live hand landmarks.  Once you complete the TODO, the
on-screen overlay will include::

    Actions: [<j0>, <j1>, <j2>]   (three values, each in [-1.0, 1.0])

Unmodified, the script raises ``NotImplementedError`` at the TODO block.

Press 'q' to quit.

Pedagogical note
----------------
Collecting demonstrations with more joints than the target environment needs
is common in imitation learning pipelines — a wrapper (or a learned projection)
maps the richer signal down to the actuator space.  This pattern appears in
Behaviour Cloning, ACT, and Diffusion Policy data-collection rigs.
"""

# Uses MediaPipe Tasks API:
# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (Apache-2.0)

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
    """Accept an action of any length and pass only the first ``n`` elements
    to the wrapped environment.

    This lets the exercise send a 3-element action while Reacher-v5 expects 2.
    The extra element is recorded (for display / logging) but not forwarded.

    Parameters
    ----------
    env:
        The environment to wrap.
    n:
        Number of action dimensions the inner environment expects.
    """

    def __init__(self, env: gym.Env, n: int) -> None:
        super().__init__(env)
        self._n = n

    def action(self, action: np.ndarray) -> np.ndarray:  # type: ignore[override]
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
                # Use landmark 4 (thumb tip) to add a third torque component.
                #
                # Steps:
                #   1. Read landmark 4 from result.hand_landmarks[0]:
                #          lm4 = result.hand_landmarks[0][4]
                #   2. Call scale_landmark_to_action on its x coordinate to get
                #      a value in [-1, 1] — call it lm4_action.
                #   3. Build a 3-element numpy array:
                #          action = np.array([
                #              scale_landmark_to_action(lm8.x),
                #              scale_landmark_to_action(lm8.y),
                #              lm4_action,
                #          ], dtype=np.float32)
                raise NotImplementedError(
                    "Complete the TODO block starting at this line. "
                    "See # SOLUTION HINT below for guidance."
                )
                # TODO END ──────────────────────────────────────────────────
                # SOLUTION HINT: Retrieve landmark 4 (thumb tip) from
                # result.hand_landmarks[0][4] and pass its .x attribute to
                # scale_landmark_to_action, then include that value as the
                # third element of the action array alongside the two existing
                # lm8-derived values.  Do not modify anything outside the TODO
                # block — the ActionSliceWrapper already handles sending only
                # the first two elements to the environment.

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
