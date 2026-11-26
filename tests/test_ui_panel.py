# -*- coding: utf-8 -*-
"""Contract for panel-space UI geometry and the desktop canvas backend.

Guarantees:

1. ``Panel.fit`` letterboxes with the same rule as ``Display.show(FIT_CONTAIN)``
   (scale by the tight axis, centre) - this is what makes one layout valid on
   a 640x480 panel, a 552x368 panel and a resized desktop window.
2. Panel <-> frame mapping is exact for interior points and refuses points in
   the letterbox bars, so a tap outside the image cannot select a detection.
3. ``TapGate`` reports a tap only for a short, in-place press; any drift or a
   long hold is a drag. Touchscreens have no hover, so no UI path may rely on
   the mouse being merely over a control.
4. ``CvCanvas`` converts (r, g, b) to OpenCV's (b, g, r) exactly once; a wrong
   order is invisible in review and obvious on screen.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.ui.panel import Panel, Rect, TapGate, TouchEvent


def test_fit_letterboxes_a_wide_frame():
    panel = Panel(640, 480)
    r = panel.fit(1280, 352)
    assert r == Rect(0, 152, 640, 176)


def test_fit_letterboxes_a_tall_frame():
    panel = Panel(640, 480)
    r = panel.fit(240, 480)
    assert r.w == 240 and r.h == 480
    assert r.x == 200 and r.y == 0


def test_fit_rejects_empty_source():
    panel = Panel(640, 480)
    assert panel.fit(0, 0) == Rect(0, 0, 640, 480)


def test_to_panel_divides_by_window_scale():
    panel = Panel(640, 480)
    assert panel.to_panel(700, 300, scale=2.0, offset=(20, 0)) == (340, 150)


def test_to_frame_round_trips_inside_the_image():
    panel = Panel(640, 480)
    fit = panel.fit(1280, 352)
    point = panel.to_frame(fit.x + fit.w // 2, fit.y + fit.h // 2, 1280, 352)
    assert point == (640, 176)


def test_to_frame_rejects_the_letterbox_bar():
    panel = Panel(640, 480)
    fit = panel.fit(1280, 352)
    assert panel.to_frame(320, fit.y - 10, 1280, 352) is None


def test_tap_gate_accepts_a_short_still_press():
    gate = TapGate(slop=12, max_hold_s=0.6)
    assert gate.feed(TouchEvent("press", 100, 100), now=0.0) is None
    assert gate.feed(TouchEvent("move", 104, 103), now=0.1) is None
    gesture = gate.feed(TouchEvent("release", 104, 103), now=0.2)
    assert gesture is not None and gesture.kind == "tap"


def test_tap_gate_rejects_drift():
    gate = TapGate(slop=12, max_hold_s=0.6)
    gate.feed(TouchEvent("press", 100, 100), now=0.0)
    gate.feed(TouchEvent("move", 140, 100), now=0.1)
    gesture = gate.feed(TouchEvent("release", 140, 100), now=0.2)
    assert gesture is not None and gesture.kind == "drag"


def test_tap_gate_rejects_a_long_hold():
    gate = TapGate(slop=12, max_hold_s=0.6)
    gate.feed(TouchEvent("press", 100, 100), now=0.0)
    gesture = gate.feed(TouchEvent("release", 100, 100), now=1.5)
    assert gesture is not None and gesture.kind == "drag"


def test_cv_canvas_maps_rgb_to_bgr_once():
    try:
        import numpy as np

        from framework.hal.platforms.linux.canvas import CvCanvas
    except ImportError:
        return  # numpy/cv2 absent: this suite also runs on the device tree
    buf = np.zeros((32, 32, 3), np.uint8)
    canvas = CvCanvas(buf)
    canvas.rect(4, 4, 8, 8, (255, 0, 0), fill=True)
    assert tuple(buf[6, 6]) == (0, 0, 255)
    assert tuple(buf[0, 0]) == (0, 0, 0)


def test_cv_canvas_text_reports_a_size():
    try:
        import numpy as np

        from framework.hal.platforms.linux.canvas import CvCanvas
    except ImportError:
        return
    canvas = CvCanvas(np.zeros((64, 128, 3), np.uint8))
    w, h = canvas.text_size("BALL 42.1 cm", 14)
    assert w > 20 and 0 < h < 40
