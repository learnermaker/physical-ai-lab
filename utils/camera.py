"""
utils/camera.py — Shared camera utilities for the Physical AI Workshop.

Provides:
    LoopingVideoCapture  — drop-in cv2.VideoCapture wrapper that loops a
                           finite video file indefinitely.
    open_camera          — open a webcam with transparent fallback to a
                           looping video file when the webcam is unavailable.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class LoopingVideoCapture:
    """
    Drop-in replacement for cv2.VideoCapture that loops a video file
    indefinitely.

    Only the three methods that downstream workshop code uses are exposed:
    ``read()``, ``isOpened()``, and ``release()``.  No other cv2.VideoCapture
    methods are forwarded — this keeps the interface minimal and testable.

    Parameters
    ----------
    path : str
        Filesystem path to the video file to loop.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap = cv2.VideoCapture(path)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Return the next frame, seeking back to frame 0 on EOF.

        Read logic (exact, per design § 2.1):
        1. Call self._cap.read() → (ret, frame)
        2. If ret is True  → return (True, frame)
        3. If ret is False → seek to frame 0, retry read() once
        4. If retry also fails → return (False, None)  (corrupt / empty file)
        """
        ret, frame = self._cap.read()
        if ret:
            return True, frame

        # EOF (or read failure) — seek back to the start and retry once
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        if ret:
            return True, frame

        # Corrupt or empty file — caller must handle this
        return False, None

    def isOpened(self) -> bool:
        """Delegate to the underlying cv2.VideoCapture."""
        return self._cap.isOpened()

    def release(self) -> None:
        """Release the underlying cv2.VideoCapture."""
        self._cap.release()


def open_camera(
    index: int = 0,
    fallback_path: Optional[str] = None,
) -> "cv2.VideoCapture | LoopingVideoCapture":
    """
    Open a webcam, falling back to a looping video file when the webcam is
    unavailable.

    Parameters
    ----------
    index : int
        OpenCV camera index to try first (default 0).
    fallback_path : str | None
        Path to the fallback video file.  When ``None`` (default), the
        function resolves the path as::

            <repo_root>/assets/fallback_hand_demo.mp4

        which is always correct regardless of the caller's working directory.
        When an explicit path is supplied and it is not a valid video source,
        ``ValueError`` is raised immediately — before the webcam is tried.

    Returns
    -------
    cv2.VideoCapture
        If the webcam at *index* is available.
    LoopingVideoCapture
        If the webcam is unavailable and the fallback video is used instead.

    Raises
    ------
    ValueError
        If *fallback_path* was explicitly supplied but cannot be opened as a
        video source.
    RuntimeError
        If the webcam is unavailable **and** the default fallback asset cannot
        be opened (e.g. the file is missing from the repository clone).
    """
    # ------------------------------------------------------------------
    # Step 1: Resolve / validate fallback_path
    # ------------------------------------------------------------------
    if fallback_path is None:
        # Default: repo-root-relative path, always correct regardless of cwd.
        fallback_path = str(
            Path(__file__).resolve().parent.parent / "assets" / "fallback_hand_demo.mp4"
        )
        _explicit_fallback = False
    else:
        # Caller supplied an explicit path — validate it immediately so the
        # error surfaces as a programming mistake rather than a silent failure
        # later.
        _probe = cv2.VideoCapture(fallback_path)
        _valid = _probe.isOpened()
        _probe.release()
        if not _valid:
            raise ValueError(
                f"fallback_path={fallback_path!r} is not a valid video source"
            )
        _explicit_fallback = True  # noqa: F841  (used only for branching above)

    # ------------------------------------------------------------------
    # Step 2–3: Try the requested webcam index
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap

    # Webcam unavailable — release the failed capture object before continuing.
    cap.release()

    # ------------------------------------------------------------------
    # Step 4: Warn to stderr
    # ------------------------------------------------------------------
    print(
        f"Camera index {index} unavailable — using fallback video: {fallback_path}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Steps 5–7: Open the fallback video
    # ------------------------------------------------------------------
    fallback_cap = cv2.VideoCapture(fallback_path)
    if not fallback_cap.isOpened():
        fallback_cap.release()
        raise RuntimeError("Neither webcam nor fallback video could be opened")

    fallback_cap.release()  # LoopingVideoCapture opens its own handle
    return LoopingVideoCapture(fallback_path)
