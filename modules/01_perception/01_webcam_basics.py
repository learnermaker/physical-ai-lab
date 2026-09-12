#!/usr/bin/env python3
"""
Module 1 Demo: Webcam Basics
Displays live camera feed in a window.  On the first frame, prints the
frame shape to stdout so participants can see the resolution.
Press 'q' to quit.

If no webcam is detected, falls back silently to the looping fallback
video (assets/fallback_hand_demo.mp4) — handled entirely by open_camera().
"""

# Uses utils/camera.py from the workshop repo.
import sys
from pathlib import Path

# --- Python path convention (Design § 2.7) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # two levels up from modules/01_perception/
sys.path.insert(0, str(REPO_ROOT))

import cv2
from utils.camera import open_camera


def main() -> None:
    cap = open_camera()

    first_frame = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame:
            print(f"Frame shape: {frame.shape}")  # e.g. (480, 640, 3)
            first_frame = False

        cv2.imshow("Webcam Basics — press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
