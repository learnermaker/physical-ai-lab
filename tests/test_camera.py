"""
tests/test_camera.py — Unit tests for utils/camera.py

Tests cover Requirements 7.1–7.5:
  7.1  open_camera returns a working capture object
  7.2  webcam unavailable → fallback video used (LoopingVideoCapture)
  7.3  LoopingVideoCapture loops past EOF continuously
  7.4  explicit invalid fallback_path → ValueError
  7.5  fallback video also unavailable → RuntimeError
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure the repo root is on sys.path so `utils` is importable regardless of
# where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.camera import LoopingVideoCapture, open_camera


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_short_video(tmp_path):
    """A real 3-frame BGR video written with cv2.VideoWriter."""
    path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_open_camera_returns_valid_cap(tmp_short_video):
    """When webcam opens successfully, open_camera returns it directly
    (not a LoopingVideoCapture)."""
    with patch("utils.camera.cv2.VideoCapture") as mock_cap_class:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap_class.return_value = mock_instance

        result = open_camera(fallback_path=tmp_short_video)

    assert result is mock_instance, "Expected the webcam capture to be returned"
    assert not isinstance(result, LoopingVideoCapture), (
        "Should NOT return LoopingVideoCapture when webcam succeeds"
    )


def test_open_camera_fallback_on_invalid_index(tmp_short_video):
    """When the webcam index fails, open_camera falls back to LoopingVideoCapture."""
    # Two successive VideoCapture calls:
    #   1st  — webcam (isOpened=False)
    #   2nd  — fallback probe inside the explicit-path validation (isOpened=True)
    #   3rd  — fallback_cap to check whether fallback opens (isOpened=True)
    # LoopingVideoCapture.__init__ also calls cv2.VideoCapture internally.
    webcam_mock = MagicMock()
    webcam_mock.isOpened.return_value = False

    fallback_mock = MagicMock()
    fallback_mock.isOpened.return_value = True

    with patch("utils.camera.cv2.VideoCapture", side_effect=[
        fallback_mock,  # explicit fallback validation probe (step 1)
        webcam_mock,    # webcam attempt (step 2–3)
        fallback_mock,  # fallback_cap openability check (step 5)
        fallback_mock,  # LoopingVideoCapture.__init__ (step 7)
    ]):
        result = open_camera(fallback_path=tmp_short_video)

    assert isinstance(result, LoopingVideoCapture), (
        "Expected LoopingVideoCapture when webcam is unavailable"
    )


def test_open_camera_raises_runtime_if_fallback_missing():
    """When neither webcam nor default fallback can be opened, RuntimeError is raised."""
    failing_mock = MagicMock()
    failing_mock.isOpened.return_value = False

    # open_camera() with no args uses the default fallback path — no upfront
    # validation step, so ALL VideoCapture calls (webcam + fallback_cap) receive
    # the failing mock.
    with patch("utils.camera.cv2.VideoCapture", return_value=failing_mock):
        with pytest.raises(RuntimeError, match="Neither webcam nor fallback video"):
            open_camera()


def test_open_camera_raises_value_error_for_explicit_bad_path():
    """Passing a non-existent explicit fallback_path raises ValueError immediately."""
    bad_path = "/nonexistent/path/that/does/not/exist.mp4"
    with pytest.raises(ValueError, match="fallback_path"):
        open_camera(fallback_path=bad_path)


def test_looping_video_capture_resets_on_eof(tmp_short_video):
    """LoopingVideoCapture wrapping a 3-frame video delivers valid frames on
    the 4th through 6th reads (i.e. it loops correctly past EOF)."""
    cap = LoopingVideoCapture(tmp_short_video)
    try:
        for i in range(6):
            ret, frame = cap.read()
            assert ret is True, f"read #{i+1} returned ret=False"
            assert frame is not None, f"read #{i+1} returned frame=None"
            assert isinstance(frame, np.ndarray), (
                f"read #{i+1} did not return an ndarray"
            )
    finally:
        cap.release()
