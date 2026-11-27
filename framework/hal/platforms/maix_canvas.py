"""Canvas backend that draws onto a ``maix.image.Image``.

The device pipeline paints the HUD straight onto the captured frame, so this
backend owns the firmware calls.  Two rules from
``docs/architecture/maixpy_drawing_backend.md`` shape it:

* colours cross the C++ boundary as ``image.Color`` objects, never as tuples.
  A tuple makes MaixPy raise TypeError, and an outer ``except`` would turn
  that into a silently blank HUD.
* ``size`` is a line height in pixels, like every other Canvas, and each
  backend calibrates its own glyph scale against it.

Text is drawn with ``draw_string`` per frame rather than through a sprite
cache.  The sprites the old pipeline kept existed to avoid re-rasterising a
black header bar; that bar is a layout concern and now draws as an opaque
``Rect`` followed by text, which costs one extra rasterisation per header
fragment and keeps this backend free of a cache whose invalidation rules
nobody can check without a device.
"""

from __future__ import annotations

from typing import Any, Dict

from framework.ui.canvas import Rgb, Size, scale_for

__all__ = ["MaixCanvas"]

# Line height ``draw_string(scale=1.0)`` produces on MaixCAM firmware.
_UNIT_PX = 15

# Stroke width follows the glyph size: the drawing code this replaced used
# thickness 1 at scale <= 1.2 and thickness 2 for the larger list text.
_THICK_PX = 24

# Colour objects are immutable and cheap to share, so they live at module
# level: a Canvas is built per frame, and a per-instance cache would rebuild
# the same four or five objects every time.
_COLORS: Dict[Rgb, Any] = {}


class MaixCanvas:
    """Draws onto a ``maix.image.Image``; never allocates a buffer."""

    def __init__(self, image: Any) -> None:
        self._img = image

    # ------------------------------------------------------------------ #
    # Canvas
    # ------------------------------------------------------------------ #

    def size(self) -> Size:
        return int(self._img.width()), int(self._img.height())

    def rect(self, x: int, y: int, w: int, h: int, color: Rgb,
             fill: bool = False, width: int = 1) -> None:
        if w <= 0 or h <= 0:
            return
        self._img.draw_rect(
            int(x), int(y), int(w), int(h),
            color=self._color(color),
            thickness=-1 if fill else max(1, int(width)),
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, color: Rgb,
             width: int = 1) -> None:
        self._img.draw_line(
            int(x1), int(y1), int(x2), int(y2),
            color=self._color(color),
            thickness=max(1, int(width)),
        )

    def circle(self, x: int, y: int, r: int, color: Rgb,
               fill: bool = False, width: int = 1) -> None:
        if r <= 0:
            return
        self._img.draw_circle(
            int(x), int(y), int(r),
            color=self._color(color),
            thickness=-1 if fill else max(1, int(width)),
        )

    def text(self, x: int, y: int, text: str, size: int, color: Rgb) -> None:
        if not text:
            return
        self._img.draw_string(
            int(x), int(y), text,
            color=self._color(color),
            scale=scale_for(size, _UNIT_PX),
            thickness=self._thickness(size),
        )

    def text_size(self, text: str, size: int) -> Size:
        if not text:
            return 0, 0
        width, height = self._img.string_size(
            text, scale=scale_for(size, _UNIT_PX), thickness=self._thickness(size)
        )
        return int(width), int(height)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _thickness(size: int) -> int:
        """Stroke width for a line height; keeps text legible at a distance."""
        return 2 if size >= _THICK_PX else 1

    def _color(self, rgb: Rgb) -> Any:
        """The firmware colour object for one RGB triple, built once."""
        cached = _COLORS.get(rgb)
        if cached is None:
            from maix import image as maix_image

            cached = maix_image.Color.from_rgb(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            _COLORS[rgb] = cached
        return cached
