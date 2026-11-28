"""Turn pointer commands into press / drag / tap events over a Scene.

The three commands (``UI_PRESS`` / ``UI_MOVE`` / ``UI_RELEASE``) arrive from
any InputSource - a cv2 window's mouse callbacks, or the touchscreen read as a
pressed/not-pressed state.  Nothing here knows which, and that is the point:
the same hit-testing and the same "was this a tap" rule serve both.

Hit boxes come from the Scene the layout just produced, so anything a layout
declares as clickable is clickable - a button, a detection box, a row.  The
old code searched a fixed button list, which meant nothing else could ever be
clicked no matter how it was drawn.

A press that drifts further than ``slop``, or lasts longer than ``max_hold_s``,
is a drag, never a tap: a resting finger must not trigger an action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from framework.cmd_queue import Cmd
from framework.ui.scene import Scene

__all__ = ["Pointer", "PointerEvent", "PRESS", "MOVE", "RELEASE"]

PRESS = "UI_PRESS"
MOVE = "UI_MOVE"
RELEASE = "UI_RELEASE"


@dataclass(frozen=True, slots=True)
class PointerEvent:
    """What a pointer command meant.

    ``kind`` is ``press`` / ``drag`` / ``tap`` / ``cancel``.  ``id`` is the hit
    box the gesture started on, empty when the press landed on nothing.
    """

    kind: str
    id: str = ""
    x: int = 0
    y: int = 0


class Pointer:
    """Press / move / release over the hit boxes of the current Scene."""

    def __init__(self, slop: int = 12, max_hold_s: float = 0.6) -> None:
        self._slop = int(slop)
        self._max_hold_s = float(max_hold_s)
        self._scene: Optional[Scene] = None
        self._target: Optional[str] = None
        self._start: Tuple[int, int] = (0, 0)
        self._started_at = 0.0
        self._moved = False

    def attach(self, scene: Optional[Scene]) -> None:
        """Point at the Scene the next commands are tested against."""
        self._scene = scene

    @property
    def target(self) -> Optional[str]:
        """The hit box currently held down, if any."""
        return self._target

    def apply(self, cmd: Cmd, now: Optional[float] = None) -> Optional[PointerEvent]:
        """Interpret one command; None when it changes nothing."""
        moment = time.monotonic() if now is None else now
        payload: Dict[str, Any] = cmd.payload or {}
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))

        if cmd.kind == PRESS:
            box = self._scene.hit_test(x, y) if self._scene is not None else None
            self._target = box.id if box is not None else None
            self._start = (x, y)
            self._started_at = moment
            self._moved = False
            return PointerEvent("press", self._target or "", x, y)

        if self._target is None:
            # A move or release with nothing held: nothing to report.  Keeping
            # this quiet is what lets a caller feed every command blindly.
            return None

        if cmd.kind == MOVE:
            dx, dy = x - self._start[0], y - self._start[1]
            if not self._moved and (abs(dx) > self._slop or abs(dy) > self._slop):
                self._moved = True
            return PointerEvent("drag", self._target, x, y) if self._moved else None

        if cmd.kind == RELEASE:
            target = self._target
            self._target = None
            if self._moved or (moment - self._started_at) > self._max_hold_s:
                return PointerEvent("cancel", target, x, y)
            landed = self._scene.hit_test(x, y) if self._scene is not None else None
            if landed is not None and landed.id == target:
                return PointerEvent("tap", target, x, y)
            return PointerEvent("cancel", target, x, y)

        return None
