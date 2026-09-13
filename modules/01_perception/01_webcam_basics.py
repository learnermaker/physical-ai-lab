#!/usr/bin/env python3
"""
Module 1 Demo 1: Webcam Basics
================================
The very first stage of the Physical AI pipeline is capturing the world
through a camera.  This script opens your webcam and displays the live feed.

What you will see:
  - A window showing your camera feed in real time
  - The frame shape printed to the terminal (e.g. (480, 640, 3))

How it connects to the pipeline:
  Webcam → [THIS FILE] → raw frames → next: hand_tracking.py

If no webcam is detected, the script automatically falls back to
assets/fallback_hand_demo.mp4 — a looping demo video.

Press 'q' to quit.
"""

# Uses utils/camera.py from this repo.
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — lets Python find the utils/ package
# ---------------------------------------------------------------------------
# __file__ is this script.  .parents[2] walks two directories up:
#   modules/01_perception/01_webcam_basics.py
#            ↑ parents[0]
#   modules/
#   ↑ parents[1]
#   physical-ai-workshop/   ← REPO_ROOT (where utils/ lives)
#   ↑ parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # makes `import utils` work

import cv2                          # OpenCV: reads camera, displays windows
from utils.camera import open_camera  # camera helper: webcam with fallback


def main() -> None:
    # ── WHY THIS MATTERS FOR PHYSICAL AI ─────────────────────────────────────
    # Every robot needs eyes. Before you can control an arm, balance a pole, or
    # navigate a warehouse, the system must first perceive the world from raw
    # sensor data. This script is Stage 1 of the pipeline: getting frames off
    # the camera and into Python so the rest of the system can work with them.
    # ─────────────────────────────────────────────────────────────────────────
    cap = open_camera()

    first_frame = True

    while True:
        # cap.read() returns (success_flag, frame).
        # frame is a numpy array of shape (height, width, 3) — 3 = BGR channels.
        # BGR (Blue-Green-Red) is OpenCV's default colour order — not RGB!
        ret, frame = cap.read()
        if not ret:
            break  # video ended or camera disconnected

        if first_frame:
            # frame.shape = (height, width, channels)
            # e.g. (480, 640, 3) means 480 rows × 640 columns × 3 colour channels
            print(f"Frame shape: {frame.shape}")
            #      height ^   ^ width   ^ channels (always 3 for colour)
            first_frame = False

        # Display the frame in a window.
        # The string is the window title.
        cv2.imshow("Webcam Basics — press q to quit", frame)

        # cv2.waitKey(1) pauses for 1 ms and returns the key pressed.
        # & 0xFF masks to the lower 8 bits (required on some platforms).
        # ord("q") is the ASCII code for 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Always release the camera and close windows when done.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # This block only runs when you execute the file directly:
    #   python modules/01_perception/01_webcam_basics.py
    # It does NOT run when another file imports this module.
    main()
