"""Text measurement for layout functions.

A layout has to know how wide a label is before it can decide where the next
one starts - that is the one thing the old ``x = w - 240`` constants were
standing in for.  Measurement needs a backend, so it enters the layout as an
argument instead of the layout reaching for a canvas.

On the device ``string_size`` is a firmware call; a per-frame LRU keeps a
static label from paying for itself every frame.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, Tuple

from framework.ui.canvas import Canvas


class Measurer(Protocol):
    """Whatever a layout can ask about text size."""

    def text_size(self, text: str, size: int) -> Tuple[int, int]:
        """The rendered ``(width, height)`` of *text* at *size*."""
        ...

    def fit(self, text: str, max_width: int, size: int,
            min_size: int = 10) -> int:
        """The largest size <= *size* whose width still fits *max_width*."""
        ...


class CanvasMeasurer:
    """A Measurer backed by a Canvas, with a small cache.

    The cache is per-frame in practice: a layout measures each label once per
    render, and the same label (``FPS:``, ``LINK:OK``) appears every frame.
    """

    def __init__(self, canvas: Canvas, capacity: int = 64) -> None:
        self._canvas = canvas
        self._capacity = max(1, int(capacity))
        self._cache: "OrderedDict[Tuple[str, int], Tuple[int, int]]" = OrderedDict()

    def text_size(self, text: str, size: int) -> Tuple[int, int]:
        key = (text, int(size))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        measured = self._canvas.text_size(text, int(size))
        measured = (int(measured[0]), int(measured[1]))
        self._cache[key] = measured
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return measured

    def fit(self, text: str, max_width: int, size: int,
            min_size: int = 10) -> int:
        size = int(size)
        while size > min_size and self.text_size(text, size)[0] > max_width:
            size -= 1
        return size
