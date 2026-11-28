"""Unit tests for the platform-neutral frame accessors.

Run: python -m pytest tests/test_frame_view.py -q

Three frame shapes reach the pipeline, and before this module each consumer
guessed: the display step knew about maix Images and ndarrays, the inference
processor assumed a maix Image, and both were wrong on the platform the other
one ran on.  These tests pin the single place that knows the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.hal.interface import frame_for_display, frame_size


class FakeMaixImage:
    """Stand-in for ``maix.image.Image``: size exposed as methods."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._w, self._h = width, height

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h


def test_a_native_image_is_measured_through_its_methods():
    assert frame_size(FakeMaixImage(320, 240)) == (320, 240)


def test_an_ndarray_is_measured_through_its_shape():
    assert frame_size(np.zeros((480, 640, 3), dtype=np.uint8)) == (640, 480)


def test_a_lazy_proxy_is_measured_without_converting_it():
    """The proxy exists so metadata reads stay free; measuring must not convert."""
    from framework.hal.platforms.lazy_frame import LazyFrame

    conversions = []

    def convert(_raw):
        conversions.append(1)
        return np.zeros((720, 1280, 3), np.uint8)

    frame = LazyFrame(object(), convert, shape_hw=(720, 1280))

    assert frame_size(frame) == (1280, 720)
    assert conversions == []


def test_an_unknown_object_reports_no_size_rather_than_raising():
    """The vision thread must be able to skip, not crash."""
    assert frame_size(object()) == (0, 0)
    assert frame_size(SimpleNamespace(shape=None)) == (0, 0)


def test_a_grey_frame_still_reports_its_width_and_height():
    assert frame_size(np.zeros((480, 640), dtype=np.uint8)) == (640, 480)


def test_the_native_frame_wins_over_the_ndarray_view():
    camera = SimpleNamespace(last_raw=FakeMaixImage(100, 50),
                             last_frame=np.zeros((50, 100, 3), np.uint8))

    assert isinstance(frame_for_display(camera), FakeMaixImage)


def test_a_platform_without_native_frames_falls_back_to_the_ndarray():
    camera = SimpleNamespace(last_raw=None,
                             last_frame=np.zeros((50, 100, 3), np.uint8))

    frame = frame_for_display(camera)

    assert isinstance(frame, np.ndarray)
    assert frame_size(frame) == (100, 50)


def test_no_frame_at_all_reads_as_none():
    assert frame_for_display(SimpleNamespace(last_raw=None)) is None
    assert frame_for_display(None) is None
