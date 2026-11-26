"""Platform-neutral drawing protocol shared by the panel UI on every target.

Coordinate space: *panel space* - origin top-left, x right, y down, unit = one
pixel of the logical screen (640x480 on MaixCAM2, 552x368 on MaixCAM-Pro).
The camera frame is a layer inside that space, not the space itself, so layout
code never needs to know the sensor resolution.

Color entry point: always ``(r, g, b)`` ints. Backends convert to their native
color type; a tuple must never cross a C++ boundary (MaixPy raises TypeError
there, and the callers' broad ``except`` turns it into a silently blank UI).

Text: ``size`` is the requested line height in pixels. Each backend calibrates
its own font scale against that number, so the same layout code produces
visually comparable text on both ends even though the glyphs differ.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

Rgb = Tuple[int, int, int]
Point = Tuple[int, int]
Size = Tuple[int, int]


@runtime_checkable
class Canvas(Protocol):
    """The seven frozen primitives (D2) plus text metrics.

    Backends own a pixel buffer and mutate it in place; a Canvas is never
    shared between threads, and its native handle never leaves the backend.
    """

    def size(self) -> Size:
        """Width and height of the drawing surface, in panel pixels."""
        ...

    def rect(self, x: int, y: int, w: int, h: int, color: Rgb,
             fill: bool = False, width: int = 1) -> None:
        """Axis-aligned rectangle; ``fill`` ignores ``width``."""
        ...

    def line(self, x1: int, y1: int, x2: int, y2: int, color: Rgb,
             width: int = 1) -> None:
        """Straight segment between two points."""
        ...

    def circle(self, x: int, y: int, r: int, color: Rgb,
               fill: bool = False, width: int = 1) -> None:
        """Circle centred on (x, y)."""
        ...

    def text(self, x: int, y: int, text: str, size: int, color: Rgb) -> None:
        """Draw *text* with its top-left corner at (x, y)."""
        ...

    def text_size(self, text: str, size: int) -> Size:
        """Measured (width, height) of *text* at the same ``size`` as text()."""
        ...


def scale_for(size: int, reference_px: int) -> float:
    """Font scale for a requested line height.

    Args:
        size: Requested line height in pixels.
        reference_px: Line height the backend's font produces at scale 1.0.

    Returns:
        Multiplier clamped away from zero; a zero scale renders nothing.
    """
    return max(float(size) / float(reference_px), 0.05)


def fit_size(canvas: Canvas, text: str, max_width: int, size: int,
             min_size: int = 10) -> int:
    """Largest font size <= *size* whose rendered width fits *max_width*.

    Layout in this project comes from a 552x368 device screen; a label that
    fits there can still overrun a 90 px desktop button. Measuring and
    shrinking is cheaper than clipping, and it keeps the text readable instead
    of cutting the last glyph off.
    """
    s = int(size)
    while s > min_size and canvas.text_size(text, s)[0] > max_width:
        s -= 1
    return s


def draw_fitted(canvas: Canvas, rect, text: str, size: int, color: Rgb,
                pad: int = 4, min_size: int = 10) -> None:
    """Centre *text* inside *rect*, shrinking it when the box is too narrow.

    ``rect`` is ``framework.ui.panel.Rect``; it is duck-typed here so this
    module stays free of a back-reference to the geometry layer.
    """
    s = fit_size(canvas, text, rect.w - 2 * pad, size, min_size)
    tw, th = canvas.text_size(text, s)
    canvas.text(rect.x + max(pad, (rect.w - tw) // 2),
                rect.y + max(0, (rect.h - th) // 2), text, s, color)
