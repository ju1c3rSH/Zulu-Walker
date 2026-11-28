"""Pointer state machine: press / drag / tap over a Scene's hit boxes.

Run: python -m pytest tests/test_pointer.py -q

The rule that matters is the tap gate: a press that drifts or lingers is not
a tap, so a resting finger or a swipe across a button cannot fire it.  The
same gate has to work for a mouse and a touchscreen, which is why it lives on
commands rather than on either device's event type.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.cmd_queue import Cmd
from framework.ui.pointer import Pointer
from framework.ui.scene import HitBox, Scene

BOX = HitBox("button:EXIT", 10, 100, 80, 40)


def _pointer() -> Pointer:
    pointer = Pointer(slop=12, max_hold_s=0.6)
    pointer.attach(Scene(hits=(BOX,)))
    return pointer


def _press(x=20, y=110):
    return Cmd("UI_PRESS", {"x": x, "y": y})


def _move(x, y):
    return Cmd("UI_MOVE", {"x": x, "y": y})


def _release(x, y):
    return Cmd("UI_RELEASE", {"x": x, "y": y})


def test_a_press_on_a_hit_box_reports_its_id():
    event = _pointer().apply(_press(), now=0.0)

    assert (event.kind, event.id) == ("press", "button:EXIT")


def test_a_press_on_nothing_reports_an_empty_id():
    event = _pointer().apply(_press(400, 300), now=0.0)

    assert (event.kind, event.id) == ("press", "")


def test_a_short_press_and_release_is_a_tap():
    pointer = _pointer()
    pointer.apply(_press(), now=0.0)

    event = pointer.apply(_release(20, 110), now=0.1)

    assert event.kind == "tap"
    assert event.id == "button:EXIT"


def test_a_small_drift_still_counts_as_a_tap():
    pointer = _pointer()
    pointer.apply(_press(), now=0.0)
    pointer.apply(_move(25, 112), now=0.05)

    event = pointer.apply(_release(25, 112), now=0.1)

    assert event.kind == "tap"


def test_a_drift_beyond_the_slop_is_a_cancel():
    pointer = _pointer()
    pointer.apply(_press(), now=0.0)
    drag = pointer.apply(_move(200, 300), now=0.05)

    assert drag.kind == "drag"
    assert pointer.apply(_release(200, 300), now=0.1).kind == "cancel"


def test_a_release_that_slides_off_the_box_is_a_cancel():
    """Pressed inside, let go outside: not a tap on that box."""
    pointer = _pointer()
    pointer.apply(_press(), now=0.0)
    pointer.apply(_move(15, 105), now=0.05)

    event = pointer.apply(_release(15, 400), now=0.1)

    assert event.kind == "cancel"


def test_a_long_press_is_not_a_tap():
    pointer = _pointer()
    pointer.apply(_press(), now=0.0)

    event = pointer.apply(_release(20, 110), now=1.5)

    assert event.kind == "cancel"


def test_a_release_with_nothing_held_is_ignored():
    assert _pointer().apply(_release(20, 110), now=0.0) is None


def test_a_move_with_nothing_held_is_ignored():
    assert _pointer().apply(_move(20, 110), now=0.0) is None


def test_without_a_scene_a_press_lands_on_nothing():
    pointer = Pointer()
    event = pointer.apply(_press(), now=0.0)

    assert event.kind == "press" and event.id == ""


def test_the_topmost_box_wins_when_boxes_overlap():
    """Later hit boxes are drawn on top, so they are tested first."""
    upper = HitBox("det:1", 0, 0, 100, 100)
    lower = HitBox("button:EXIT", 10, 10, 40, 40)
    pointer = Pointer()
    pointer.attach(Scene(hits=(upper, lower)))

    assert pointer.apply(_press(20, 20), now=0.0).id == "button:EXIT"


def test_attach_replaces_the_hit_boxes():
    pointer = _pointer()
    pointer.attach(Scene(hits=(HitBox("row:2", 0, 0, 50, 50),)))

    assert pointer.apply(_press(20, 20), now=0.0).id == "row:2"
