"""Panel space: the logical screen every UI draws into, and its input mapping.

Why a panel space instead of the camera frame:

* the panel is the only surface a user ever sees, and it is the same size on
  both ends of a port (640x480 / 552x368), so layout constants stay meaningful;
* the camera frame changes with the sensor mode (1280x352 here), so drawing in
  frame space makes every label's size depend on the sensor configuration;
* ``Display.show(fit=FIT_CONTAIN)`` already scales the composed image once.
  Composing at panel size means the UI is drawn at the size it is shown at, so
  glyphs are not rendered 2x and then halved by the panel scaler.

The panel keeps no pixels - it only answers "where does a frame land" and
"where did the finger land". Pixel buffers belong to the Canvas backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in panel pixels."""

    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True, slots=True)
class TouchEvent:
    """One raw pointer event, already expressed in panel coordinates.

    ``kind`` is ``"press"`` / ``"move"`` / ``"release"``. Touchscreens only
    produce these three; a mouse additionally has hover, which must never be
    load-bearing in UI logic or the same code stops working on the panel.
    """

    kind: str
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Gesture:
    """A completed interaction: a tap, or a drag that can be ignored as one."""

    kind: str          # "tap" | "drag"
    x: int
    y: int


class Panel:
    """Logical screen geometry: frame placement and window-to-panel mapping."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def fit(self, src_w: int, src_h: int) -> Rect:
        """Letterbox *src_w* x *src_h* into the panel, preserving aspect.

        Same rule as ``Display.show(fit=FIT_CONTAIN)``: scale by the tighter
        axis and centre the result. Integer coordinates keep the panel edges
        covered exactly once.
        """
        if src_w <= 0 or src_h <= 0:
            return Rect(0, 0, self.width, self.height)
        k = min(self.width / src_w, self.height / src_h)
        w = max(1, min(self.width, int(round(src_w * k))))
        h = max(1, min(self.height, int(round(src_h * k))))
        return Rect((self.width - w) // 2, (self.height - h) // 2, w, h)

    def to_panel(self, wx: int, wy: int, scale: float,
                 offset: Point = (0, 0)) -> Point:
        """Window pixels -> panel pixels (divide by the present scale)."""
        if scale <= 0:
            return 0, 0
        px = int((wx - offset[0]) / scale)
        py = int((wy - offset[1]) / scale)
        return px, py

    def to_frame(self, px: int, py: int, src_w: int, src_h: int,
                 fit: Optional[Rect] = None) -> Optional[Point]:
        """Panel pixel -> camera-frame pixel, or None outside the image area.

        This replaces a per-call ``resize_map_pos_reverse``: the mapping is a
        rectangle, so two subtractions and one division are enough.
        """
        r = fit or self.fit(src_w, src_h)
        if not r.contains(px, py):
            return None
        fx = int((px - r.x) * src_w / r.w)
        fy = int((py - r.y) * src_h / r.h)
        return min(fx, src_w - 1), min(fy, src_h - 1)


class TapGate:
    """Turn press/move/release into a tap, with touch-only semantics.

    A press that drifts further than ``slop`` or lasts longer than
    ``max_hold_s`` is reported as a drag and never as a tap, so scrolling or a
    resting finger cannot trigger a button. Both numbers are in panel pixels
    and seconds, i.e. independent of the window scale.
    """

    def __init__(self, slop: int = 12, max_hold_s: float = 0.6) -> None:
        self._slop = int(slop)
        self._max_hold_s = float(max_hold_s)
        self._start: Optional[Point] = None
        self._started_at = 0.0
        self._moved = False

    def feed(self, event: TouchEvent,
             now: Optional[float] = None) -> Optional[Gesture]:
        t = time.monotonic() if now is None else now
        if event.kind == "press":
            self._start = (event.x, event.y)
            self._started_at = t
            self._moved = False
            return None
        if self._start is None:
            return None
        if event.kind == "move":
            dx = event.x - self._start[0]
            dy = event.y - self._start[1]
            if abs(dx) > self._slop or abs(dy) > self._slop:
                self._moved = True
            return None
        if event.kind == "release":
            start, self._start = self._start, None
            if self._moved or (t - self._started_at) > self._max_hold_s:
                return Gesture("drag", start[0], start[1])
            return Gesture("tap", event.x, event.y)
        return None
