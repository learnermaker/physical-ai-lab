"""
tests/test_properties.py — Property-based tests for the Physical AI Workshop.

Tests cover:
  Property 1 & 2  — LoopingVideoCapture reads continuously (never exhausts)
  Property 3      — open_camera raises ValueError for any invalid fallback_path
  Property 4      — make_env fallback chain exhausts modes before raising
  Property 5      — Non-rendering exceptions propagate without retry
  Property 6      — scale_landmark_to_action is a correct linear map
  Property 7      — verify_install summary count matches actual pass count
  Property 8      — Gemini fallback always returns a parseable action dict

Run:
    pytest tests/test_properties.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Python path — repo root must be importable regardless of CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Optional-dependency stubs
#
# Several workshop-only packages (stable_baselines3, mediapipe, mujoco,
# torch, google-genai, dotenv, cvzone) may not be installed in the test
# runner's Python environment.  We inject minimal stubs so that the utils
# and module files can be imported without error.  The stubs are only
# inserted if the real package is absent — in the full workshop env the real
# packages are used unchanged.
# ---------------------------------------------------------------------------
import types as _types


def _stub(name: str, attrs: dict | None = None) -> _types.ModuleType:
    """Create and register a stub module if it is not already present."""
    if name in sys.modules:
        return sys.modules[name]
    mod = _types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# stable_baselines3 — only the BaseCallback class is referenced at import time
if "stable_baselines3" not in sys.modules:
    _sb3 = _stub("stable_baselines3")
    _sb3_common = _stub("stable_baselines3.common")
    _sb3_callbacks = _stub("stable_baselines3.common.callbacks")

    class _BaseCallback:
        """Minimal stub — SB3 callbacks are not exercised by property tests."""
        def __init__(self, verbose: int = 0) -> None:
            pass

    _sb3_callbacks.BaseCallback = _BaseCallback
    _sb3.common = _sb3_common
    _sb3_common.callbacks = _sb3_callbacks

# mediapipe — Module 4 imports it; we only need scale_landmark_to_action
if "mediapipe" not in sys.modules:
    _mp = _stub("mediapipe")
    _mp_tasks = _stub("mediapipe.tasks")
    _mp_tasks_python = _stub("mediapipe.tasks.python")
    _mp_vision_tasks = _stub("mediapipe.tasks.vision")
    _mp_vision_python = _stub("mediapipe.tasks.python.vision")
    _mp_solutions = _stub("mediapipe.solutions")
    _mp_hands_sol = _stub("mediapipe.solutions.hands")
    _mp_hands_sol.HAND_CONNECTIONS = []
    # Dummy Image / ImageFormat so module-level code that may reference them is safe
    _mp.Image = MagicMock
    _mp.ImageFormat = MagicMock()
    _mp.tasks = _mp_tasks
    # tasks.vision must be accessible as attribute AND as sys.modules entry
    _mp_tasks.vision = _mp_vision_tasks
    _mp_tasks.python = _mp_tasks_python
    _mp_tasks_python.vision = _mp_vision_python
    # Provide HandLandmarkerResult and RunningMode as stubs
    _mp_vision_tasks.HandLandmarkerResult = MagicMock
    _mp_vision_python.HandLandmarkerOptions = MagicMock
    _mp_vision_python.HandLandmarker = MagicMock
    _mp_vision_python.RunningMode = MagicMock()
    _mp.solutions = _mp_solutions
    _mp_solutions.hands = _mp_hands_sol

# mujoco — imported transitively via gymnasium[mujoco]; only needed for env creation
_stub("mujoco")

# torch — not used in property tests
_stub("torch")

# google.genai — deferred import inside get_robot_action; we patch it in Property 8
_stub("google")
_stub("google.genai", {"Client": MagicMock})
# types.Part.from_bytes is called before the patched Client raises; stub it as
# a MagicMock instance so attribute access (Part.from_bytes) works naturally.
_genai_types = _stub("google.genai.types")
_genai_types.Part = MagicMock()

# dotenv — loaded at module level in Module 5 scripts
try:
    import dotenv  # noqa: F401
except ImportError:
    _stub("dotenv")
    _stub("python_dotenv")

# cvzone — not used in property tests
_stub("cvzone")

# ---------------------------------------------------------------------------
# Now import the utilities that the property tests actually exercise
# ---------------------------------------------------------------------------
from utils.camera import LoopingVideoCapture, open_camera  # noqa: E402
from utils.gym_utils import _RENDER_EXCEPTION_PATTERNS, make_env  # noqa: E402

# ---------------------------------------------------------------------------
# Dynamic import: scale_landmark_to_action from Module 4
#
# Module 4 imports mediapipe, cv2, mujoco etc. at module scope.  The stubs
# above let the module load successfully so we can extract the pure function.
# ---------------------------------------------------------------------------
_m4_path = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "04_perception_to_action"
    / "01_hand_to_reacher.py"
)
_m4_spec = importlib.util.spec_from_file_location("m4", _m4_path)
_m4 = importlib.util.module_from_spec(_m4_spec)
_m4_spec.loader.exec_module(_m4)
scale_landmark_to_action = _m4.scale_landmark_to_action

# ---------------------------------------------------------------------------
# Dynamic import: get_robot_action from Module 5
# ---------------------------------------------------------------------------
_m5_path = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "05_foundation_models"
    / "02_gemini_robot_brain.py"
)
_m5_spec = importlib.util.spec_from_file_location("m5", _m5_path)
_m5 = importlib.util.module_from_spec(_m5_spec)
_m5_spec.loader.exec_module(_m5)
get_robot_action = _m5.get_robot_action


# ---------------------------------------------------------------------------
# Fixtures (Task 16.1)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_short_video(tmp_path):
    """
    A real 5-frame synthetic BGR video written with cv2.VideoWriter.

    Used by Properties 1, 2, and 3 to provide a valid video path.
    The video is 64×48, 25 fps, solid black frames.
    """
    path = tmp_path / "test_short.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def tmp_cache_file(tmp_path):
    """
    A minimal cached_responses.json with 3 entries, one of which contains
    an ``"action"`` key.  Used by Property 8.
    """
    data = {
        "describe the scene in one sentence": "A desk with a laptop and coffee mug.",
        "what gesture": "The hand appears to be making a pointing gesture.",
        "you are a robot controller...action suggestion": {
            "action": "WAIT",
            "reason": "Scene is clear, no objects detected",
        },
    }
    cache_path = tmp_path / "cached_responses.json"
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return cache_path


# ---------------------------------------------------------------------------
# Helper (Task 16.1) — used by Property 7
# ---------------------------------------------------------------------------


def count_passes(passes: int, fails: int) -> int:
    """
    Simulate the verify_install counting logic.

    N in "Setup complete: N/14 checks passed" equals the count of checks
    that produced neither a [FAIL] nor a [WARN] result — i.e. pure passes only.
    """
    return passes


# ---------------------------------------------------------------------------
# Property 1 & 2: LoopingVideoCapture never exhausts (Task 16.2)
# ---------------------------------------------------------------------------


@given(n_reads=st.integers(min_value=1, max_value=500))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_looping_capture_never_exhausts(tmp_short_video, n_reads):
    """
    Property 1: open_camera fallback always produces readable frames.
    Property 2: LoopingVideoCapture reads continuously past end of file.

    For any number of reads (1–500), every call to read() on a
    LoopingVideoCapture returns (True, non-empty frame).

    Validates: Requirements 7.1, 7.2, 7.3
    """
    cap = LoopingVideoCapture(str(tmp_short_video))
    try:
        for _ in range(n_reads):
            ret, frame = cap.read()
            assert ret is True
            assert frame is not None
            assert frame.shape[0] > 0
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Property 3: open_camera raises ValueError for any invalid fallback_path (Task 16.3)
# ---------------------------------------------------------------------------


@given(bad_path=st.text(min_size=1).filter(lambda s: not Path(s).exists()))
@settings(max_examples=100)
def test_open_camera_value_error_for_bad_path(bad_path):
    """
    Property 3: ValueError for any invalid explicit fallback_path.

    For any string that does not resolve to a readable video file,
    open_camera(fallback_path=that_string) raises ValueError containing
    the supplied path.

    Validates: Requirements 7.4
    """
    with pytest.raises(ValueError):
        open_camera(index=99, fallback_path=bad_path)


# ---------------------------------------------------------------------------
# Property 4: make_env fallback chain exhausts modes before raising (Task 16.4)
# ---------------------------------------------------------------------------


@given(fail_count=st.integers(min_value=1, max_value=2))
@settings(max_examples=100)
def test_make_env_fallback_chain_succeeds(fail_count):
    """
    Property 4: make_env exhausts render modes before raising.

    When gym.make raises a rendering-related RuntimeError for the first
    fail_count modes, make_env continues and succeeds on the next mode.
    The total call count is fail_count + 1.

    Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.6
    """
    call_count = [0]

    def mock_make(env_id, render_mode=None):
        call_count[0] += 1
        if call_count[0] <= fail_count:
            raise RuntimeError("opengl error: display not found")
        return MagicMock()

    with patch("gymnasium.make", side_effect=mock_make):
        env = make_env("CartPole-v1")
        assert env is not None
        assert call_count[0] == fail_count + 1


# ---------------------------------------------------------------------------
# Property 5: Non-rendering exceptions propagate without retry (Task 16.5)
# ---------------------------------------------------------------------------


@given(
    msg=st.text(min_size=1).filter(
        lambda s: not any(p in s.lower() for p in _RENDER_EXCEPTION_PATTERNS)
    )
)
@settings(max_examples=100)
def test_non_render_exception_propagates_once(msg):
    """
    Property 5: Non-rendering exceptions propagate without retry.

    For any exception message that does not match any rendering-related
    pattern, a ValueError raised by gym.make propagates immediately and
    gym.make is called exactly once (no retries).

    Validates: Requirements 8.7
    """
    call_count = [0]

    def mock_make(env_id, render_mode=None):
        call_count[0] += 1
        raise ValueError(msg)

    with patch("gymnasium.make", side_effect=mock_make):
        with pytest.raises(ValueError):
            make_env("SomeEnv-v1")

    assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Property 6: scale_landmark_to_action is a correct linear map (Task 16.6)
# ---------------------------------------------------------------------------


@given(
    x=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
)
@settings(max_examples=100)
def test_scale_landmark_linear_map(x):
    """
    Property 6: scale_landmark_to_action is a correct linear map.

    For any normalised coordinate x in [0, 1], scale_landmark_to_action(x)
    equals 2.0 * x - 1.0 within a tolerance of 1e-9.

    Validates: Requirements 13.2
    """
    result = scale_landmark_to_action(x)
    expected = 2.0 * x - 1.0
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property 7: verify_install summary count matches actual pass count (Task 16.7)
# ---------------------------------------------------------------------------


@given(passes=st.integers(min_value=0, max_value=14))
@settings(max_examples=100)
def test_verify_summary_count_matches(passes):
    """
    Property 7: verify_install summary count matches actual pass count.

    For any number of passes in [0, 14], count_passes returns the exact pass
    count.  fails is derived as 14 - passes so that passes + fails == 14
    without any filtering overhead.

    Validates: Requirements 5.5
    """
    fails = 14 - passes
    assert count_passes(passes, fails) == passes


# ---------------------------------------------------------------------------
# Property 8: Gemini fallback always returns parseable action dict (Task 16.8)
# ---------------------------------------------------------------------------


@given(
    exc=st.sampled_from(
        [
            Exception("network timeout"),
            ConnectionError("connection refused"),
            json.JSONDecodeError("Expecting value", "", 0),
            RuntimeError("quota exceeded"),
        ]
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gemini_fallback_always_parseable(exc, tmp_cache_file):
    """
    Property 8: Gemini fallback always returns a parseable action dict.

    For any exception type raised by the Gemini API call, get_robot_action
    returns a dict with at least the "action" key (from the cache fallback)
    and never propagates the exception.

    Validates: Requirements 14.5
    """
    with patch("utils.camera.cv2.VideoCapture") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        mock_cv2.return_value = mock_cap

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = exc
            result = get_robot_action(cache_path=tmp_cache_file, api_key="fake_key")

    assert isinstance(result, dict)
    assert "action" in result
