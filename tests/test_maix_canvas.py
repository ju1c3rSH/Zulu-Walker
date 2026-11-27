"""Unit tests for the maix.Image canvas backend.

Run: python -m pytest tests/test_maix_canvas.py -q

The firmware is absent on a development machine, so a stand-in ``maix.image``
records the calls.  What is under test is the boundary itself: colours must
cross as ``Color`` objects (a tuple silently blanks the HUD on device), the
coordinate arguments must be ints, and ``size`` must become a glyph scale.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.hal.platforms.maix_canvas import MaixCanvas


class FakeColor:
    """Stand-in for ``maix.image.Color``."""

    @staticmethod
    def from_rgb(r, g, b):
        return ("color", r, g, b)


class FakeImage:
    """Records every draw call the canvas makes."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._w = width
        self._h = height
        self.calls = []

    def width(self):
        return self._w

    def height(self):
        return self._h

    def draw_rect(self, x, y, w, h, color, thickness):
        self.calls.append(("rect", x, y, w, h, color, thickness))

    def draw_line(self, x1, y1, x2, y2, color, thickness):
        self.calls.append(("line", x1, y1, x2, y2, color, thickness))

    def draw_circle(self, x, y, r, color, thickness):
        self.calls.append(("circle", x, y, r, color, thickness))

    def draw_string(self, x, y, text, color, scale, thickness):
        self.calls.append(("text", x, y, text, color, scale, thickness))

    def string_size(self, text, scale, thickness):
        return len(text) * 8, 15


@pytest.fixture()
def fake_firmware(monkeypatch):
    module = types.ModuleType("maix.image")
    module.Color = FakeColor
    parent = types.ModuleType("maix")
    parent.image = module
    monkeypatch.setitem(sys.modules, "maix", parent)
    monkeypatch.setitem(sys.modules, "maix.image", module)
    return module


def test_size_comes_from_the_image(fake_firmware):
    assert MaixCanvas(FakeImage(320, 240)).size() == (320, 240)


def test_a_filled_rect_maps_to_thickness_minus_one(fake_firmware):
    image = FakeImage()
    MaixCanvas(image).rect(1, 2, 3, 4, (10, 20, 30), fill=True)

    assert image.calls == [("rect", 1, 2, 3, 4, ("color", 10, 20, 30), -1)]


def test_a_stroked_rect_keeps_its_width(fake_firmware):
    image = FakeImage()
    MaixCanvas(image).rect(0, 0, 5, 5, (1, 2, 3), width=3)

    assert image.calls[-1][-1] == 3


def test_a_degenerate_rect_draws_nothing(fake_firmware):
    image = FakeImage()
    canvas = MaixCanvas(image)

    canvas.rect(0, 0, 0, 5, (0, 0, 0))
    canvas.circle(0, 0, 0, (0, 0, 0))

    assert image.calls == []


def test_colours_cross_the_boundary_as_color_objects(fake_firmware):
    """A tuple would raise TypeError inside the firmware binding."""
    image = FakeImage()
    canvas = MaixCanvas(image)

    canvas.rect(0, 0, 1, 1, (4, 5, 6))
    canvas.line(0, 0, 1, 1, (4, 5, 6))
    canvas.circle(1, 1, 2, (4, 5, 6))
    canvas.text(0, 0, "x", 15, (4, 5, 6))

    colors = {
        call[{"rect": 5, "line": 5, "circle": 4, "text": 4}[call[0]]]
        for call in image.calls
    }
    assert colors == {("color", 4, 5, 6)}


def test_text_uses_the_scale_calibrated_for_the_line_height(fake_firmware):
    image = FakeImage()
    MaixCanvas(image).text(2, 3, "FPS", 15, (255, 255, 255))

    kind, x, y, text, _color, scale, _thickness = image.calls[0]
    assert (kind, x, y, text) == ("text", 2, 3, "FPS")
    assert scale == pytest.approx(1.0)


def test_text_size_reports_the_firmware_measurement(fake_firmware):
    canvas = MaixCanvas(FakeImage())
    assert canvas.text_size("abcd", 15) == (32, 15)


def test_empty_text_is_not_drawn(fake_firmware):
    image = FakeImage()
    canvas = MaixCanvas(image)

    canvas.text(0, 0, "", 15, (0, 0, 0))

    assert image.calls == []
    assert canvas.text_size("", 15) == (0, 0)
