"""Compose a frame with a layout, on a cadence, reusing one buffer.

This is the part that used to be inside ``VisionManager._update_display_frame``:
decide whether this iteration needs a new composed frame, copy, draw, keep the
result.  It moved out because it has nothing to do with vision - it takes a
frame and a state, and returns a frame.

The buffer is reused across rebuilds.  That is not an optimisation detail: the
LCD sink skips a frame when its generation has not changed, and a fresh buffer
every tick would defeat that.  ``generation`` is the contract with the sink -
object identity is not, because it silently breaks the moment the composer
reallocates.

Skips are also why the display does not cost a full 640x480 copy per vision
iteration: a sensor frame that has not advanced, or a rebuild inside the
cadence window, returns the previous composition untouched.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from framework.hal.interface.frame import frame_size
from framework.ui.layout import Layout
from framework.ui.measurer import CanvasMeasurer
from framework.ui.scene import Scene
from framework.ui.space import Constraints

__all__ = ["Composed", "HudComposer"]

logger = logging.getLogger(__name__)

#: The device renders at ~30fps; drawing faster than the panel presents means
#: paying for copies nobody sees.
DEFAULT_INTERVAL_S = 1.0 / 30.0

#: A very small requested interval must not turn the loop into a copy loop.
MIN_INTERVAL_S = 1.0 / 100.0


def _resolve_interval(seconds: float) -> float:
    """A rebuild window in seconds; 0 disables the window entirely.

    ``0`` means "rebuild whenever the sensor frame changed", which is what the
    panel wants when it presents every tick.  Any positive value is floored at
    ``MIN_INTERVAL_S`` so a 1 ms request cannot turn the vision loop into a
    copy loop.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    if value <= 0:
        return 0.0
    return max(value, MIN_INTERVAL_S)


@dataclass(frozen=True, slots=True)
class Composed:
    """One composition: the pixels, what was drawn, and which generation."""

    frame: Any
    scene: Scene
    generation: int


class HudComposer:
    """Turns ``(frame, state)`` into a composed frame at a bounded cadence."""

    def __init__(self, canvas_factory: Any = None, layout: Optional[Layout] = None,
                 interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self._canvas_factory = canvas_factory
        self._layout = layout
        self._interval_s = _resolve_interval(interval_s)
        self._enabled = True

        self._frame: Any = None
        self._scene = Scene()
        self._generation = 0
        self._last_serial: Any = None
        self._last_at = 0.0
        self._error_logged = False

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #

    @property
    def generation(self) -> int:
        """The number of successful compositions so far."""
        return self._generation

    @property
    def frame(self) -> Any:
        """The most recently composed frame, or None before the first one."""
        return self._frame

    def set_enabled(self, enabled: bool) -> None:
        """Stop composing entirely; ``frame`` keeps its last value."""
        self._enabled = bool(enabled)

    def set_layout(self, layout: Layout) -> None:
        self._layout = layout
        self.reset()

    def set_canvas_factory(self, factory: Any) -> None:
        self._canvas_factory = factory
        self.reset()

    def set_interval_ms(self, interval_ms: Optional[float]) -> None:
        """Retune the rebuild window; None keeps the current one."""
        if interval_ms is None:
            return
        try:
            seconds = float(interval_ms) / 1000.0
        except (TypeError, ValueError):
            return
        seconds = float(interval_ms) / 1000.0
        self._interval_s = _resolve_interval(seconds)

    def reset(self) -> None:
        """Force the next call to rebuild, e.g. after a camera reconnect."""
        self._last_serial = None
        self._last_at = 0.0

    # ------------------------------------------------------------------ #
    # composition
    # ------------------------------------------------------------------ #

    def compose(self, source: Any, state: Any,
                serial: Any = None) -> Optional[Composed]:
        """Compose *source*, or return the previous result when nothing moved.

        Args:
            source: The frame to copy; must support ``copy()``.
            state: What the layout draws (``ViewState``).
            serial: A counter that changes when the sensor produced a new
                frame.  None means "always rebuild" (platforms without one).
        """
        if not self._enabled or source is None:
            return None

        if self._frame is not None:
            if serial is not None and serial == self._last_serial:
                return self._current()
            if time.monotonic() - self._last_at < self._interval_s:
                self._last_serial = serial if serial is not None else -1
                return self._current()

        self._last_serial = serial if serial is not None else -1
        self._last_at = time.monotonic()
        self._rebuild(source, state)
        return self._current()

    def _rebuild(self, source: Any, state: Any) -> None:
        frame = source.copy()
        scene = Scene()

        if self._canvas_factory is not None and self._layout is not None:
            try:
                canvas = self._canvas_factory(frame)
                scene = self._layout.scene(
                    state, Constraints.of(frame_size(frame)), CanvasMeasurer(canvas)
                )
                scene.render(canvas)
            except Exception:
                # A broken HUD must not reach the vision loop's guard, which
                # would sleep a second and drop result delivery to 1 Hz.  The
                # frame is still published; only the overlay is missing.
                if not self._error_logged:
                    logger.warning("HUD render failed; continuing without it",
                                   exc_info=True)
                    self._error_logged = True
                scene = Scene()

        self._frame = frame
        self._scene = scene
        self._generation += 1

    def current(self) -> Optional[Composed]:
        """The latest composition, without composing a new one."""
        return self._current()

    def _current(self) -> Optional[Composed]:
        if self._frame is None:
            return None
        return Composed(self._frame, self._scene, self._generation)
