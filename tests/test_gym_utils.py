"""
tests/test_gym_utils.py — Unit tests for utils/gym_utils.py

Tests cover Requirements 8.1–8.7:
  8.1  make_env returns the environment when the requested render mode succeeds
  8.2  make_env falls back from "human" to "rgb_array" on a render error
  8.3  make_env falls back from "rgb_array" to None on a render error
  8.4  make_env raises RuntimeError when all three modes fail with render errors
  8.5  make_env propagates non-rendering exceptions immediately after one call
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure the repo root is on sys.path so `utils` is importable regardless of
# where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Stub stable_baselines3 if not installed — gym_utils imports BaseCallback
# at module scope, so the stub must exist before the import below.
# ---------------------------------------------------------------------------
if "stable_baselines3" not in sys.modules:
    _sb3 = types.ModuleType("stable_baselines3")
    _sb3_common = types.ModuleType("stable_baselines3.common")
    _sb3_callbacks = types.ModuleType("stable_baselines3.common.callbacks")

    class _BaseCallback:
        """Minimal stub for SB3 BaseCallback."""
        def __init__(self, verbose: int = 0) -> None:
            pass

    _sb3_callbacks.BaseCallback = _BaseCallback
    _sb3.common = _sb3_common
    _sb3_common.callbacks = _sb3_callbacks
    sys.modules["stable_baselines3"] = _sb3
    sys.modules["stable_baselines3.common"] = _sb3_common
    sys.modules["stable_baselines3.common.callbacks"] = _sb3_callbacks

from utils.gym_utils import _RENDER_EXCEPTION_PATTERNS, make_env  # noqa: E402

# ---------------------------------------------------------------------------
# Shared render-error factory
# ---------------------------------------------------------------------------

_RENDER_ERROR = RuntimeError("opengl error: display not found")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_make_env_human_mode_succeeds():
    """
    Requirement 8.1: When gym.make succeeds on the first call (human mode),
    make_env returns the environment and calls gym.make exactly once with
    render_mode="human".
    """
    mock_env = MagicMock()
    with patch("gymnasium.make", return_value=mock_env) as mock_make:
        result = make_env("CartPole-v1", render_mode="human")

    assert result is mock_env
    mock_make.assert_called_once_with("CartPole-v1", render_mode="human")


def test_make_env_falls_back_to_rgb_array():
    """
    Requirements 8.2, 8.3: When gym.make raises a rendering error for
    render_mode="human", make_env retries with render_mode="rgb_array" and
    returns the environment on success.  gym.make is called exactly twice.
    """
    mock_env = MagicMock()
    call_count = [0]

    def side_effect(env_id, render_mode=None):
        call_count[0] += 1
        if render_mode == "human":
            raise RuntimeError("opengl error: display not available")
        return mock_env

    with patch("gymnasium.make", side_effect=side_effect):
        result = make_env("CartPole-v1")

    assert result is mock_env
    assert call_count[0] == 2


def test_make_env_falls_back_to_none():
    """
    Requirements 8.4, 8.5: When gym.make raises rendering errors for both
    "human" and "rgb_array", make_env retries with render_mode=None and
    returns the environment on success.  gym.make is called exactly three times.
    """
    mock_env = MagicMock()
    call_count = [0]

    def side_effect(env_id, render_mode=None):
        call_count[0] += 1
        if render_mode in ("human", "rgb_array"):
            raise RuntimeError("glfw error: no display")
        return mock_env

    with patch("gymnasium.make", side_effect=side_effect):
        result = make_env("CartPole-v1")

    assert result is mock_env
    assert call_count[0] == 3


def test_make_env_raises_runtime_when_all_fail():
    """
    Requirement 8.6: When all three render modes fail with rendering-related
    exceptions, make_env raises RuntimeError (not the original exception).
    gym.make must have been called exactly three times.
    """
    call_count = [0]

    def side_effect(env_id, render_mode=None):
        call_count[0] += 1
        raise RuntimeError("egl error: no display server")

    with patch("gymnasium.make", side_effect=side_effect):
        with pytest.raises(RuntimeError):
            make_env("CartPole-v1")

    assert call_count[0] == 3


def test_make_env_propagates_non_render_exception():
    """
    Requirement 8.7: A ValueError whose message does not match any pattern in
    _RENDER_EXCEPTION_PATTERNS propagates immediately.  gym.make is called
    exactly once — no fallback retries occur.
    """
    # Verify our test message genuinely contains no render keyword.
    non_render_msg = "No registered env with id CartPole-X"
    assert not any(p in non_render_msg.lower() for p in _RENDER_EXCEPTION_PATTERNS)

    call_count = [0]

    def side_effect(env_id, render_mode=None):
        call_count[0] += 1
        raise ValueError(non_render_msg)

    with patch("gymnasium.make", side_effect=side_effect):
        with pytest.raises(ValueError, match="No registered env"):
            make_env("CartPole-X")

    assert call_count[0] == 1
