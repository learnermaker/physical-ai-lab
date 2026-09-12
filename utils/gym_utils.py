"""
utils/gym_utils.py — Shared Gymnasium utilities for the Physical AI Workshop.

Provides:
    _RENDER_EXCEPTION_PATTERNS  — tuple of lowercase substrings that identify
                                  rendering-related exceptions.
    _is_render_error             — predicate that classifies an exception as
                                  rendering-related.
    make_env                     — create a Gymnasium environment with a
                                  transparent render-mode fallback chain
                                  (human → rgb_array → None).
    LivePlotCallback             — SB3 callback for live matplotlib reward plot
                                  with Agg fallback for headless environments.
    StreamingCallback            — SB3 callback that pushes training progress
                                  events to a queue.Queue for SSE streaming.
"""

from __future__ import annotations

import queue
from pathlib import Path

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Rendering exception detection
# ---------------------------------------------------------------------------

_RENDER_EXCEPTION_PATTERNS: tuple[str, ...] = (
    "opengl",
    "glfw",
    "display",
    "egl",
    "pyglet",
    "gl error",
    "render",
    "framebuffer",
    "glx",
    # Additional patterns for Windows no-GPU environments
    "access violation",  # GLFW crash on machines without OpenGL drivers
    "gladloadgl",        # MuJoCo gladLoadGL error (no OpenGL)
    "fatalerror",        # MuJoCo FatalError wrapper
    "wgl",               # Windows OpenGL layer failures
)


def _is_render_error(exc: Exception) -> bool:
    """
    Return True when *exc* is a rendering-related exception.

    Accepts ``ImportError``, ``OSError``, ``RuntimeError``, and ``AttributeError``
    instances whose string representation contains at least one keyword from
    ``_RENDER_EXCEPTION_PATTERNS``.

    Also catches any exception whose *type name* contains "fatalerror" —
    this covers ``mujoco.FatalError`` (raised as ``FatalError: gladLoadGL error``
    on machines without OpenGL) without requiring a direct import of mujoco.

    Any exception not matched by these rules (e.g. ``ValueError`` from an
    unknown env_id) is treated as a programming error and propagated without
    retry.
    """
    msg = str(exc).lower()
    type_name = type(exc).__name__.lower()
    keyword_match = any(pattern in msg for pattern in _RENDER_EXCEPTION_PATTERNS)
    type_match = "fatalerror" in type_name or "gladloadgl" in msg
    return (
        isinstance(exc, (ImportError, OSError, RuntimeError, AttributeError))
        and keyword_match
    ) or type_match


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_id: str, render_mode: str = "human") -> gym.Env:
    """
    Create a Gymnasium environment with a transparent render-mode fallback.

    Attempts render modes in the order ``["human", "rgb_array", None]``.
    The first mode that succeeds is used.  If a mode raises a
    rendering-related exception (identified by ``_is_render_error``), a
    progress message is printed to ``stdout`` and the next mode is tried.
    Any non-rendering exception propagates immediately without retry.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g. ``"CartPole-v1"``).  Must be
        registered in the Gymnasium registry; unrecognised IDs raise
        whatever exception ``gymnasium.make`` normally raises (typically
        ``gymnasium.error.Error`` or a subclass) — this is propagated
        without retry.
    render_mode : str
        Requested render mode.  Defaults to ``"human"``.  When the
        requested mode fails, the fallback chain continues from
        ``"rgb_array"`` → ``None`` regardless of which mode was requested.

    Returns
    -------
    gym.Env
        The created environment.

    Raises
    ------
    RuntimeError
        If all three render modes (``"human"``, ``"rgb_array"``, ``None``)
        fail with rendering-related exceptions.
    (propagates)
        Any non-rendering exception raised by ``gymnasium.make`` is
        re-raised immediately after the single ``gym.make`` call that
        triggered it.

    Notes
    -----
    When the requested ``render_mode`` succeeds, the message printed is::

        [make_env] Created 'CartPole-v1' with render_mode='human'

    When a fallback mode succeeds, the message printed is::

        [make_env] render_mode='human' failed — using 'rgb_array'
    """
    requested_mode = render_mode
    modes = ["human", "rgb_array", None]

    for mode in modes:
        env = None
        try:
            env = gym.make(env_id, render_mode=mode)
            # Some environments (e.g. Reacher-v5 with 'human' mode on a
            # no-OpenGL machine) crash not during gym.make() but during the
            # first reset() call when the renderer is actually initialised.
            # We probe with a throw-away reset here so that the fallback chain
            # catches those failures too.
            obs, _ = env.reset()
            if mode != requested_mode:
                print(
                    f"[make_env] render_mode={requested_mode!r} failed — using {mode!r}"
                )
            else:
                print(f"[make_env] Created {env_id!r} with render_mode={mode!r}")
            return env
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if _is_render_error(exc):
                print(f"[make_env] render_mode={mode!r} failed: {exc}")
                continue
            raise  # non-rendering exception — propagate immediately

    raise RuntimeError(f"make_env: all render modes failed for {env_id!r}")


# ---------------------------------------------------------------------------
# SB3 Callbacks
# ---------------------------------------------------------------------------

# Adapted from SB3 Callbacks documentation:
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html (MIT)

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402


class LivePlotCallback(BaseCallback):
    """
    Matplotlib live reward plot updated every check_freq training steps.
    Falls back to file output (Agg backend) when a display is unavailable.
    """

    def __init__(
        self,
        check_freq: int = 1000,
        save_path: str | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.check_freq = check_freq
        # Resolve save_path relative to repo root so it works regardless of CWD.
        if save_path is None:
            self.save_path = str(
                Path(__file__).resolve().parent.parent / "models" / "training_curve.png"
            )
        else:
            self.save_path = save_path

        self._file_mode = False
        self._fig = None
        self._ax = None

        # Attempt to enable interactive mode. On headless systems (no display
        # server) this raises; we catch broadly because matplotlib may raise
        # different exception types depending on the backend and OS.
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.ion()
            self._plt = plt
        except Exception:  # noqa: BLE001
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415

            self._plt = plt
            self._file_mode = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_figure(self) -> None:
        """Create the figure and axes on first use."""
        if self._fig is None:
            self._fig, self._ax = self._plt.subplots(figsize=(8, 4))
            self._ax.set_xlabel("Episode")
            self._ax.set_ylabel("Reward")
            self._ax.set_title("Training Reward")

    def _update_plot(self) -> None:
        """Redraw the reward plot from the current ep_info_buffer."""
        buf = self.model.ep_info_buffer
        if not buf:
            return

        self._ensure_figure()
        rewards = [ep["r"] for ep in buf]
        episodes = list(range(len(rewards)))

        self._ax.cla()
        self._ax.set_xlabel("Episode")
        self._ax.set_ylabel("Reward")
        self._ax.set_title("Training Reward")
        self._ax.plot(episodes, rewards, alpha=0.4, color="steelblue", label="Episode reward")

        # Running mean over last 100 episodes.
        if len(rewards) >= 2:
            window = min(100, len(rewards))
            running_mean = [
                float(np.mean(rewards[max(0, i - window + 1) : i + 1]))
                for i in range(len(rewards))
            ]
            self._ax.plot(
                episodes,
                running_mean,
                color="darkorange",
                linewidth=2,
                label=f"Mean (last {window})",
            )
        self._ax.legend(loc="upper left")

        self._fig.canvas.draw()

        if self._file_mode:
            self._fig.savefig(self.save_path, bbox_inches="tight")
        else:
            self._plt.pause(0.001)

    # ------------------------------------------------------------------
    # BaseCallback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            self._update_plot()
        return True

    def _on_training_end(self) -> None:
        # _update_plot handles the final save/draw (including file mode save).
        self._update_plot()
        if self._file_mode:
            print(f"[LivePlotCallback] Training curve saved to {self.save_path!r}")
        else:
            self._plt.show()


class StreamingCallback(BaseCallback):
    """
    SB3 callback that pushes training progress events to a queue.Queue
    for consumption by the FastAPI SSE training progress endpoint.
    """

    def __init__(
        self,
        job_queue: queue.Queue,
        check_freq: int = 1000,
        model_save_path: str = "models/ppo-CartPole-trained.zip",
    ) -> None:
        super().__init__(verbose=0)
        self.job_queue = job_queue
        self.check_freq = check_freq
        self.model_save_path = model_save_path

    def _on_step(self) -> bool:
        """Every check_freq steps push mean reward to the queue."""
        if self.num_timesteps % self.check_freq == 0:
            buf = self.model.ep_info_buffer
            if buf:
                mean_reward = float(np.mean([ep["r"] for ep in buf]))
                self.job_queue.put(
                    {
                        "timestep": self.num_timesteps,
                        "mean_reward": mean_reward,
                    }
                )
        return True

    def _on_training_end(self) -> None:
        """Save the model and signal completion."""
        self.model.save(self.model_save_path)
        self.job_queue.put({"done": True, "model_path": self.model_save_path})
