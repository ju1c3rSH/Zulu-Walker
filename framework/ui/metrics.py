"""Design tokens derived from the surface size.

This is the ``dimens.xml`` of the HUD: sizes and spacings, no positions.  The
reference surface is 640x480, which reproduces the constants the device layout
was originally hand-tuned at; every other size scales from there.

Keeping these out of the layout functions means a layout reads ``m.text_mid``
instead of ``18``, so one change here rescales the whole screen - and a
third-party layout gets the same proportions for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

REFERENCE = (640, 480)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Text sizes and spacings for one surface size."""

    width: int
    height: int

    # Vertical bands.
    bar_h: int
    line2_h: int
    alert_h: int

    # Text sizes.
    text_big: int
    text_mid: int
    text_small: int

    # Spacing.
    pad: int
    gap: int

    # Line weights.
    halo: int
    stroke: int

    @classmethod
    def of(cls, size: Tuple[int, int]) -> "Metrics":
        width, height = int(size[0]), int(size[1])
        return cls(
            width=width,
            height=height,
            bar_h=_scale(height, 30, 480),
            line2_h=_scale(height, 24, 480),
            alert_h=_scale(height, 20, 480),
            text_big=max(14, _scale(height, 22, 480)),
            text_mid=max(12, _scale(height, 18, 480)),
            text_small=max(10, _scale(height, 15, 480)),
            pad=max(4, _scale(width, 6, 640)),
            gap=max(6, _scale(width, 12, 640)),
            halo=max(2, _scale(height, 4, 480)),
            stroke=max(1, _scale(height, 2, 480)),
        )

    @property
    def header_h(self) -> int:
        """Total height of the two-line header bar."""
        return self.bar_h + self.line2_h


def _scale(value: int, base: int, reference: int) -> int:
    """``value`` scaled from the reference axis, rounded to whole pixels."""
    return int(round(value * base / reference))
