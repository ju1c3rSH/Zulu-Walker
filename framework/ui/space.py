"""The area a layout may use, and the operations that carve it up.

Jetpack Compose carries ``Constraints(minWidth, maxWidth, minHeight,
maxHeight)``; this one carries only the maximum.  Every element here sizes to
its content and the root is fixed by the canvas, so a minimum would be a field
nothing reads.

The point of slicing rather than computing offsets is that ``x = w - 240``
encodes two unrelated decisions in one number: how wide the right cluster is,
and how much gap sits before it.  Adding a field then means re-deriving every
constant.  With slices, adding a field costs one more ``take_right``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class Constraints:
    """An area in panel pixels that a layout may fill."""

    x: int
    y: int
    max_width: int
    max_height: int

    @classmethod
    def of(cls, size: Tuple[int, int]) -> "Constraints":
        """The whole surface, as a layout's root constraint."""
        width, height = size
        return cls(0, 0, int(width), int(height))

    @property
    def right(self) -> int:
        return self.x + self.max_width

    @property
    def bottom(self) -> int:
        return self.y + self.max_height

    def inset(self, left: int = 0, top: int = 0,
              right: int = 0, bottom: int = 0) -> "Constraints":
        """Shrink on each side, the way a container applies padding."""
        return Constraints(
            self.x + left,
            self.y + top,
            max(0, self.max_width - left - right),
            max(0, self.max_height - top - bottom),
        )

    def take_left(self, width: int, gap: int = 0) -> Tuple["Constraints", "Constraints"]:
        """Reserve *width* on the left; return ``(taken, remaining)``."""
        taken_w = min(max(0, int(width)), self.max_width)
        taken = Constraints(self.x, self.y, taken_w, self.max_height)
        remaining = Constraints(
            self.x + taken_w + gap,
            self.y,
            max(0, self.max_width - taken_w - gap),
            self.max_height,
        )
        return taken, remaining

    def take_right(self, width: int, gap: int = 0) -> Tuple["Constraints", "Constraints"]:
        """Reserve *width* on the right; return ``(taken, remaining)``."""
        taken_w = min(max(0, int(width)), self.max_width)
        taken = Constraints(self.right - taken_w, self.y, taken_w, self.max_height)
        remaining = Constraints(
            self.x,
            self.y,
            max(0, self.max_width - taken_w - gap),
            self.max_height,
        )
        return taken, remaining

    def take_top(self, height: int, gap: int = 0) -> Tuple["Constraints", "Constraints"]:
        """Reserve *height* at the top; return ``(taken, remaining)``."""
        taken_h = min(max(0, int(height)), self.max_height)
        taken = Constraints(self.x, self.y, self.max_width, taken_h)
        remaining = Constraints(
            self.x,
            self.y + taken_h + gap,
            self.max_width,
            max(0, self.max_height - taken_h - gap),
        )
        return taken, remaining

    def fits(self, width: int, height: int) -> bool:
        """Whether a box of this size stays inside the area."""
        return width <= self.max_width and height <= self.max_height
