"""Hit-testable UI controls, independent of any drawing backend.

The device tuner locks a slider on press and follows the finger even when it
leaves the track; a tap-then-release button must not. Keeping both behaviours
here - as geometry plus a small state machine - is what lets the same page run
on a touchscreen and on a mouse without a per-platform fork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from framework.ui.panel import Rect


@dataclass
class Button:
    """A press/release target; the action itself belongs to the caller."""

    label: str
    rect: Rect

    def hit(self, x: int, y: int) -> bool:
        return self.rect.contains(x, y)


@dataclass
class Slider:
    """A horizontal track with a value in ``[lo, hi]``, latched on press.

    ``rect`` is the whole row: ``track_inset`` px of label on the left and
    ``readout_w`` px of value on the right are not draggable, so a press on the
    text cannot jump the value.
    """

    key: str
    rect: Rect
    lo: float
    hi: float
    value: float = 0.0
    step: float = 1.0
    label: str = ""
    track_inset: int = 0
    readout_w: int = 48
    dragging: bool = field(default=False, init=False)

    KNOB = 7

    @property
    def track(self) -> Rect:
        x = self.rect.x + self.track_inset
        w = self.rect.w - self.track_inset - self.readout_w - self.KNOB
        return Rect(x, self.rect.y + self.rect.h // 2 - self.KNOB, max(1, w),
                    self.KNOB * 2)

    def hit(self, x: int, y: int) -> bool:
        """Track plus a vertical margin: a fingertip is wider than 14 px."""
        t = self.track
        return (t.x - self.KNOB <= x <= t.right + self.KNOB
                and t.y - 8 <= y <= t.bottom + 8)

    def begin(self, x: int, y: int) -> bool:
        if not self.hit(x, y):
            return False
        self.dragging = True
        self._set_from_x(x)
        return True

    def drag(self, x: int) -> bool:
        if not self.dragging:
            return False
        self._set_from_x(x)
        return True

    def end(self) -> None:
        self.dragging = False

    def _set_from_x(self, x: int) -> None:
        t = self.track
        frac = (x - t.x) / max(1, t.w)
        frac = min(max(frac, 0.0), 1.0)
        raw = self.lo + frac * (self.hi - self.lo)
        if self.step > 0:
            raw = round(raw / self.step) * self.step
        self.value = min(max(raw, self.lo), self.hi)

    def knob_x(self) -> int:
        t = self.track
        span = self.hi - self.lo
        frac = 0.0 if span == 0 else (self.value - self.lo) / span
        return int(t.x + frac * t.w)

    def readout(self) -> str:
        if self.step >= 1:
            return str(int(round(self.value)))
        return f"{self.value:.2f}"


def find_button(buttons, x: int, y: int) -> Optional[Button]:
    for b in buttons:
        if b.hit(x, y):
            return b
    return None


def find_slider(sliders, x: int, y: int) -> Optional[Slider]:
    for s in sliders:
        if s.hit(x, y):
            return s
    return None
