"""Composition cadence and buffer reuse.

Run: python -m pytest tests/test_hud_composer.py -q

The composer replaced ``VisionManager._update_display_frame``.  These tests
cover what that method actually decided: whether this iteration needs a new
composed frame, and which frame the caller sees when it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.ui.composer import HudComposer
from framework.ui.overlay import Rect
from framework.ui.scene import Scene
from framework.ui.space import Constraints


class RecordingCanvas:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def size(self):
        return self.frame.shape[1], self.frame.shape[0]

    def rect(self, x, y, w, h, color, fill=False, width=1):
        self.calls.append("rect")

    def line(self, x1, y1, x2, y2, color, width=1):
        self.calls.append("line")

    def circle(self, x, y, r, color, fill=False, width=1):
        self.calls.append("circle")

    def text(self, x, y, text, size, color):
        self.calls.append("text")

    def text_size(self, text, size):
        return len(text) * size // 2, size


class FakeLayout:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def scene(self, state, constraints, measure) -> Scene:
        self.calls += 1
        return Scene((Rect(0, 0, 16, 16, (1, 2, 3), fill=True),), (), constraints)


class ExplodingLayout:
    name = "boom"

    def scene(self, state, constraints, measure):
        raise RuntimeError("layout lost its surface")


def _frame(h=48, w=64):
    return np.zeros((h, w, 3), np.uint8)


def _composer(layout=None, interval_s=0.0):
    return HudComposer(RecordingCanvas, layout or FakeLayout(), interval_s=interval_s)


def test_the_first_composition_copies_the_frame():
    composer = _composer()
    source = _frame()

    composed = composer.compose(source, object())

    assert composed is not None
    assert composed.frame is not source
    assert composed.generation == 1


def test_an_unchanged_serial_does_not_rebuild():
    composer = _composer()
    source = _frame()

    first = composer.compose(source, object(), serial=7)
    second = composer.compose(source, object(), serial=7)

    assert second.generation == first.generation
    assert second.frame is first.frame


def test_a_new_serial_rebuilds():
    composer = _composer()
    source = _frame()

    first = composer.compose(source, object(), serial=1)
    second = composer.compose(source, object(), serial=2)

    assert second.generation > first.generation


def test_the_rebuild_window_skips_a_fresh_frame():
    """Inside the cadence window the previous composition stands."""
    composer = _composer(interval_s=10.0)
    source = _frame()

    first = composer.compose(source, object(), serial=1)
    second = composer.compose(source, object(), serial=2)

    assert second.generation == first.generation


def test_a_disabled_composer_produces_nothing():
    composer = _composer()
    composer.set_enabled(False)

    assert composer.compose(_frame(), object()) is None


def test_no_source_frame_produces_nothing():
    composer = _composer()

    assert composer.compose(None, object()) is None


def test_a_broken_layout_does_not_escape():
    """The vision loop's guard would otherwise sleep a second per frame."""
    composer = _composer(ExplodingLayout())

    composed = composer.compose(_frame(), object())

    assert composed is not None
    assert composed.generation == 1


def test_the_layout_receives_the_frame_size_as_its_constraint():
    seen = []

    class Recording(FakeLayout):
        def scene(self, state, constraints, measure):
            seen.append((constraints.max_width, constraints.max_height))
            return super().scene(state, constraints, measure)

    composer = _composer(Recording())
    composer.compose(_frame(h=48, w=64), object())

    assert seen == [(64, 48)]


def test_reset_forces_the_next_call_to_rebuild():
    composer = _composer()
    source = _frame()

    first = composer.compose(source, object(), serial=5)
    composer.reset()
    second = composer.compose(source, object(), serial=5)

    assert second.generation > first.generation


def test_without_a_canvas_factory_the_frame_is_still_copied():
    composer = HudComposer(None, FakeLayout())
    source = _frame()

    composed = composer.compose(source, object())

    assert composed is not None
    assert composed.frame is not source
    assert isinstance(composed.scene, Scene)


def test_the_interval_can_be_retuned_at_runtime():
    composer = _composer(interval_s=0.0)
    composer.set_interval_ms(5000)

    first = composer.compose(_frame(), object(), serial=1)
    second = composer.compose(_frame(), object(), serial=2)

    assert second.generation == first.generation
