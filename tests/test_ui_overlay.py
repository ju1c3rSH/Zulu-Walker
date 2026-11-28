"""Unit tests for the backend-neutral drawing primitives.

Run: python -m pytest tests/test_ui_overlay.py -q

The point of the Overlay is that what the HUD shows can be asserted without
a camera, a display or a firmware: the primitive list is data, and the
renderer is a pure dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from framework.ui.overlay import Circle, Line, Overlay, Rect, Text


class RecordingCanvas:
    """A Canvas that records calls instead of drawing."""

    def __init__(self) -> None:
        self.calls = []

    def size(self):
        return 640, 480

    def rect(self, x, y, w, h, color, fill=False, width=1):
        self.calls.append(("rect", x, y, w, h, color, fill, width))

    def line(self, x1, y1, x2, y2, color, width=1):
        self.calls.append(("line", x1, y1, x2, y2, color, width))

    def circle(self, x, y, r, color, fill=False, width=1):
        self.calls.append(("circle", x, y, r, color, fill, width))

    def text(self, x, y, text, size, color):
        self.calls.append(("text", x, y, text, size, color))

    def text_size(self, text, size):
        return len(text) * size // 2, size


def test_an_empty_overlay_draws_nothing():
    canvas = RecordingCanvas()
    Overlay().render(canvas)
    assert canvas.calls == []


def test_primitives_are_drawn_in_insertion_order():
    overlay = Overlay()
    overlay.add(Rect(1, 2, 3, 4, (10, 20, 30), fill=True))
    overlay.add(Text(5, 6, "hi", 15, (255, 255, 255)))
    canvas = RecordingCanvas()

    overlay.render(canvas)

    assert canvas.calls == [
        ("rect", 1, 2, 3, 4, (10, 20, 30), True, 1),
        ("text", 5, 6, "hi", 15, (255, 255, 255)),
    ]


def test_every_primitive_type_reaches_the_matching_canvas_method():
    overlay = Overlay()
    overlay.add(
        Line(0, 1, 2, 3, (1, 1, 1), width=2),
        Circle(4, 5, 6, (2, 2, 2), fill=False, width=3),
    )
    canvas = RecordingCanvas()

    overlay.render(canvas)

    assert canvas.calls == [
        ("line", 0, 1, 2, 3, (1, 1, 1), 2),
        ("circle", 4, 5, 6, (2, 2, 2), False, 3),
    ]


def test_clear_keeps_the_list_reusable():
    overlay = Overlay()
    overlay.add(Rect(0, 0, 1, 1, (0, 0, 0)))
    assert len(overlay) == 1

    overlay.clear()
    assert len(overlay) == 0

    overlay.add(Text(0, 0, "x", 12, (0, 0, 0)))
    assert len(overlay) == 1


def test_an_unknown_primitive_is_refused_rather_than_skipped():
    overlay = Overlay()
    overlay.add("not a primitive")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        overlay.render(RecordingCanvas())
