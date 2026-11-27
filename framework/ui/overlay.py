"""Drawing primitives that carry no rendering backend.

The vision pipeline describes what the display shows; a Canvas decides how
that becomes pixels.  Keeping the description as data is what lets the same
HUD run on the device's ``maix.Image`` and on an OpenCV buffer, and what
makes it testable without a camera: a test asserts the primitive list, not
the pixels.

Order matters.  Primitives are drawn in insertion order, so a later one
paints over an earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from framework.ui.canvas import Canvas, Rgb

__all__ = [
    "Circle",
    "Line",
    "Overlay",
    "Primitive",
    "Rect",
    "Text",
]


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle; ``fill`` ignores ``width``."""

    x: int
    y: int
    w: int
    h: int
    color: Rgb
    width: int = 1
    fill: bool = False


@dataclass(frozen=True, slots=True)
class Line:
    """Straight segment between two points."""

    x1: int
    y1: int
    x2: int
    y2: int
    color: Rgb
    width: int = 1


@dataclass(frozen=True, slots=True)
class Circle:
    """Circle centred on (x, y)."""

    x: int
    y: int
    r: int
    color: Rgb
    width: int = 1
    fill: bool = False


@dataclass(frozen=True, slots=True)
class Text:
    """Text whose box is anchored at (x, y).

    ``size`` is the line height.  ``align`` / ``valign`` name which edge of
    the text box (x, y) refers to, so a caller can place a label by its right
    edge without measuring it first: measuring needs the backend, and a
    primitive must stay a pure value.
    """

    x: int
    y: int
    text: str
    size: int
    color: Rgb
    align: str = "left"
    valign: str = "top"


Primitive = Rect | Line | Circle | Text


class Overlay:
    """An ordered list of primitives, drawn in insertion order."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: List[Primitive] = []

    def add(self, *items: Primitive) -> None:
        """Append primitives; they paint after everything already present."""
        self._items.extend(items)

    def clear(self) -> None:
        """Drop every primitive, keeping the list object for reuse."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Tuple[Primitive, ...]:
        """The primitives, for tests and debuggers."""
        return tuple(self._items)

    def render(self, canvas: Canvas) -> None:
        """Draw every primitive onto *canvas* in order."""
        for item in self._items:
            if isinstance(item, Rect):
                canvas.rect(
                    item.x, item.y, item.w, item.h, item.color,
                    fill=item.fill, width=item.width,
                )
            elif isinstance(item, Line):
                canvas.line(
                    item.x1, item.y1, item.x2, item.y2, item.color,
                    width=item.width,
                )
            elif isinstance(item, Circle):
                canvas.circle(
                    item.x, item.y, item.r, item.color,
                    fill=item.fill, width=item.width,
                )
            elif isinstance(item, Text):
                tx, ty = item.x, item.y
                if item.align != "left" or item.valign != "top":
                    width, height = canvas.text_size(item.text, item.size)
                    if item.align == "right":
                        tx -= width
                    elif item.align == "center":
                        tx -= width // 2
                    if item.valign == "bottom":
                        ty -= height
                    elif item.valign == "middle":
                        ty -= height // 2
                canvas.text(tx, ty, item.text, item.size, item.color)
            else:
                raise TypeError(f"unknown primitive: {item!r}")
