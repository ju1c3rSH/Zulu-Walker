# -*- coding: utf-8 -*-
"""Contract for the panel widgets and the cv2 window sink.

Guarantees:

1. A slider's value is a pure function of the pointer x within its track - no
   accumulated deltas, so the same code behaves identically for a finger and a
   mouse, and a dropped move event cannot desynchronise it.
2. A press outside the track never changes the value (labels are not hits).
3. The sink presents only when the frame object changed and the interval has
   elapsed; a skipped present must not consume the interval.
4. Fill and stroke differ in edge treatment but agree on colour order.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from framework.hal.platforms.linux.canvas import CvCanvas
from framework.hal.platforms.linux.window import CvWindowSink
from framework.ui.panel import Rect
from framework.ui.widgets import Button, Slider, find_button, find_slider


def _slider(**kw):
    s = Slider(key="h_lo", label="H MIN", rect=Rect(0, 0, 640, 56),
               lo=0, hi=180, value=90, track_inset=100, readout_w=56, **kw)
    return s


def test_slider_maps_track_ends_to_range():
    s = _slider()
    t = s.track
    s.begin(t.x, t.y)
    assert s.value == 0
    s.drag(t.right)
    assert s.value == 180


def test_slider_press_on_label_is_ignored():
    s = _slider()
    assert not s.begin(20, 28)
    assert s.value == 90
    assert not s.dragging


def test_slider_latches_beyond_the_track():
    s = _slider()
    t = s.track
    assert s.begin(t.x + t.w // 2, t.y)
    s.drag(t.right + 200)          # dragged past the end
    assert s.value == 180
    s.end()
    assert not s.dragging


def test_slider_knob_position_round_trips():
    s = _slider()
    s.value = 90
    t = s.track
    assert abs(s.knob_x() - (t.x + t.w / 2)) <= 2


def test_find_helpers_respect_geometry():
    btn = Button("EXIT", Rect(540, 434, 88, 38))
    assert find_button([btn], 560, 450) is btn
    assert find_button([btn], 10, 10) is None
    s = _slider()
    assert find_slider([s], s.track.x + 5, s.track.y) is s


class _StubWindow:
    def __init__(self):
        self.presented = 0

    def present(self, frame):
        self.presented += 1
        return True

    def close(self):
        pass


def test_sink_skips_the_same_frame_object():
    win = _StubWindow()
    sink = CvWindowSink(win)
    frame = np.zeros((480, 640, 3), np.uint8)
    sink.push(frame)
    sink.flush()
    sink.flush()
    assert win.presented == 1


def test_canvas_fill_and_stroke_agree_on_colour_order():
    buf = np.zeros((40, 40, 3), np.uint8)
    c = CvCanvas(buf)
    c.rect(2, 2, 13, 13, (255, 0, 0), fill=True)
    c.rect(20, 2, 13, 13, (0, 255, 0), width=1)
    assert tuple(buf[8, 8]) == (0, 0, 255)          # BGR of red
    b, g, r = (int(v) for v in buf[2, 26])          # stroke: anti-aliased edge
    assert (b, r) == (0, 0) and g > 200


def test_canvas_clips_instead_of_dropping():
    buf = np.zeros((40, 40, 3), np.uint8)
    c = CvCanvas(buf)
    c.rect(-10, -10, 30, 30, (255, 255, 255), fill=True)
    assert tuple(buf[0, 0]) == (255, 255, 255)
    assert tuple(buf[25, 25]) == (0, 0, 0)


def test_canvas_text_size_covers_descenders():
    c = CvCanvas(np.zeros((80, 160, 3), np.uint8))
    w_plain, h_plain = c.text_size("BALL", 14)
    w_desc, h_desc = c.text_size("jpgy", 14)
    assert w_plain > 0 and w_desc > 0
    assert h_desc >= h_plain
