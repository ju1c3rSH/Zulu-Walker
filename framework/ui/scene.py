"""What a layout produces: primitives, their hit boxes, and their bounds.

A ``Scene`` is the whole output of a layout - pure data, no backend, no
pixels.  That is what makes a layout testable off-device and what lets the
same layout be drawn by ``CvCanvas`` on a laptop and ``MaixCanvas`` on the
board.

``Measured`` exists because a layout has to know a label's size before it can
place it: it is Compose's ``Placeable``, kept as a value so a layout stays a
pure function instead of a chain of closures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Optional, Tuple

from framework.ui.overlay import Circle, Line, Overlay, Primitive, Rect, Text
from framework.ui.space import Constraints

__all__ = ["HitBox", "Measured", "Scene", "LayoutOverflow"]


class LayoutOverflow(Exception):
    """A layout produced content outside the area it was given."""


@dataclass(frozen=True, slots=True)
class HitBox:
    """A clickable region, in panel pixels.

    ``id`` is namespaced by the layout (``"button:EXIT"``, ``"det:3"``); the
    pointer never interprets it, the caller does.
    """

    id: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


@dataclass(frozen=True, slots=True)
class Measured:
    """A primitive plus its measured size, before placement."""

    primitive: Primitive
    width: int
    height: int

    def place(self, x: int, y: int) -> Primitive:
        """Move the primitive so its top-left corner lands on ``(x, y)``."""
        p = self.primitive
        if isinstance(p, (Rect, Text, Circle)):
            return replace(p, x=int(x), y=int(y))
        if isinstance(p, Line):
            dx, dy = int(x) - p.x1, int(y) - p.y1
            return replace(p, x1=p.x1 + dx, y1=p.y1 + dy,
                           x2=p.x2 + dx, y2=p.y2 + dy)
        return p


@dataclass(frozen=True, slots=True)
class Scene:
    """One frame's worth of drawing, in panel coordinates."""

    primitives: Tuple[Primitive, ...] = ()
    hits: Tuple[HitBox, ...] = ()
    bounds: Optional[Constraints] = None

    def render(self, canvas: Any) -> None:
        """Draw every primitive, in order, onto *canvas*."""
        overlay = Overlay()
        overlay.add(*self.primitives)
        overlay.render(canvas)

    def hit_test(self, x: int, y: int) -> Optional[HitBox]:
        """The topmost hit box containing the point; later entries win."""
        for box in reversed(self.hits):
            if box.contains(x, y):
                return box
        return None

    def validate(self, measure: Any = None) -> None:
        """Raise when any primitive falls outside ``bounds``.

        Compose throws when ``layout(w, h)`` exceeds its constraints; the same
        check here turns a label that silently overruns the screen into a test
        failure while the layout is being written.

        Pass a ``Measurer`` to also bound text: a primitive only stores its
        anchor, so without one, text can only be checked at its origin.
        """
        if self.bounds is None:
            return
        outside: List[Primitive] = []
        for p in self.primitives:
            right, bottom = _extent(p, measure)
            if right > self.bounds.right or bottom > self.bounds.bottom:
                outside.append(p)
        if outside:
            raise LayoutOverflow(
                f"{len(outside)} primitive(s) exceed "
                f"{self.bounds.max_width}x{self.bounds.max_height}"
            )


def _extent(p: Primitive, measure: Any = None) -> Tuple[int, int]:
    """The bottom-right corner a primitive occupies."""
    if isinstance(p, Rect):
        return p.x + p.w, p.y + p.h
    if isinstance(p, Circle):
        return p.x + p.r, p.y + p.r
    if isinstance(p, Line):
        return max(p.x1, p.x2), max(p.y1, p.y2)
    if isinstance(p, Text):
        if measure is not None:
            width, height = measure.text_size(p.text, p.size)
            return p.x + width, p.y + height
        return p.x, p.y
    return 0, 0


def stack(items: Iterable[Primitive]) -> Tuple[Primitive, ...]:
    """Convenience for the common ``tuple(...)`` of layout output."""
    return tuple(items)
