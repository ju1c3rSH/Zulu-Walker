"""Hold on-screen UI state: button geometry and interaction gates.

Pure logic, with no pixel APIs, so it is unit-testable off device and reusable
by any backend that produces frame coordinates. Project orchestration lives in
``app``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitRegion:
    """Axis-aligned button rectangle in frame coordinates."""

    x: int
    y: int
    w: int
    h: int

    def contains(self, px: float, py: float, margin: int = 4) -> bool:
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.h + margin
        )


def bottom_left_region(frame_h: int, size: int, margin: int) -> HitRegion:
    """Return a standard icon slot pinned to the bottom-left frame edge."""
    return HitRegion(x=margin, y=frame_h - size - 8, w=size, h=size)


class CooldownGate:
    """Rate-limits repeated triggers (e.g. the LED toggle's 0.3s debounce)."""

    def __init__(self, cooldown_s: float) -> None:
        self._cooldown = cooldown_s
        self._last = 0.0  # monotonic ts of last allowed trigger

    def allow(self, now: float) -> bool:
        if now - self._last >= self._cooldown:
            self._last = now
            return True
        return False
