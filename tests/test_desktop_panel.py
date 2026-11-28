"""Unit tests for the desktop panel and the content the app feeds it.

Run: python -m pytest tests/test_desktop_panel.py -q

The panel is exercised headless: the composed buffer is the observable, so
neither a window nor a GUI session is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.cmd_queue import Cmd
from framework.hal.interface import Detection
from framework.hal.platforms.linux.panel import DesktopPanel, default_buttons
from framework.ui import hud

pytest.importorskip("cv2")

PANEL_W, PANEL_H = 640, 480


def _panel() -> DesktopPanel:
    """A headless panel: the buffer is what the test looks at."""
    return DesktopPanel(width=PANEL_W, height=PANEL_H, headless=True)


def _frame(width: int, height: int, color=(10, 20, 30)) -> np.ndarray:
    return np.full((height, width, 3), color, np.uint8)


def test_the_panel_has_no_gui_when_it_is_headless():
    panel = _panel()
    assert panel.has_gui is False
    assert panel.size == (PANEL_W, PANEL_H)


def test_the_frame_is_letterboxed_into_the_panel():
    """A 16:9 frame lands in the middle of the panel, bars above and below."""
    panel = _panel()
    buf = panel.compose(_frame(640, 360, color=(10, 20, 30)), [], hud.HudState())

    # The bar above the picture is empty (black).
    assert buf[0, 0].tolist() == [0, 0, 0]
    # The picture starts at the letterbox offset and carries the frame colour.
    assert buf[60, 320].tolist() == [10, 20, 30]
    # The bar below the picture is empty as well.
    assert buf[PANEL_H - 1, 320].tolist() == [0, 0, 0]


def test_the_header_bar_is_drawn_over_the_picture():
    """The device HUD bar covers the top of the panel."""
    panel = _panel()
    buf = panel.compose(_frame(640, 360, color=(10, 20, 30)), [], hud.HudState())

    # The bar is black, whatever the frame underneath was.
    assert buf[5, 320].tolist() == [0, 0, 0]


def test_the_detection_boxes_are_mapped_into_panel_space():
    """A frame-space box lands where the letterboxed picture put it."""
    panel = _panel()
    frame = _frame(640, 360)
    hit = hud.Hit(x=0, y=0, w=320, h=180, text="ball:0.9")
    state = hud.HudState(hits=[hit])
    buf = panel.compose(frame, [hit], state)

    # The frame is scaled by 1 and moved down by 60, so the box starts at
    # (0, 60) in panel pixels and is 320x180 large.  The HUD strokes the
    # border, so the edge pixels carry the box colour.
    left_edge = buf[70, 0]
    top_edge = buf[60, 160]
    assert int(left_edge[1]) > 150, "the box edge must be green"
    assert int(top_edge[1]) > 150, "the box edge must be green"
    # The HUD green is (60, 205, 110) in RGB, so the blue channel stays low
    # in BGR order - unlike the frame colour underneath it.
    assert int(left_edge[2]) < 100


def test_the_exit_button_asks_to_stop():
    panel = _panel()
    # Hit boxes come from the Scene the layout produced, so the panel must
    # have composed once before a click can land on anything.
    panel.compose(_frame(PANEL_W, PANEL_H), [], hud.HudState())
    assert panel.exit_requested is False
    button = default_buttons(PANEL_W, PANEL_H)[0]
    x, y = button.rect.x + 4, button.rect.y + 4

    panel.handle_command(Cmd("UI_PRESS", {"x": x, "y": y}))
    assert panel.exit_requested is False, "a press alone is not a tap"

    panel.handle_command(Cmd("UI_RELEASE", {"x": x, "y": y}))
    assert panel.exit_requested is True


def test_a_press_that_drags_off_the_button_does_not_stop_the_panel():
    """A resting or sliding finger must not trigger the button."""
    panel = _panel()
    panel.compose(_frame(PANEL_W, PANEL_H), [], hud.HudState())
    button = default_buttons(PANEL_W, PANEL_H)[0]
    x, y = button.rect.x + 4, button.rect.y + 4

    panel.handle_command(Cmd("UI_PRESS", {"x": x, "y": y}))
    panel.handle_command(Cmd("UI_MOVE", {"x": x + 200, "y": y + 100}))
    panel.handle_command(Cmd("UI_RELEASE", {"x": x + 200, "y": y + 100}))

    assert panel.exit_requested is False


def test_the_window_close_asks_to_stop():
    panel = _panel()
    panel.handle_command(Cmd("EXIT"))
    assert panel.exit_requested is True


def test_a_press_outside_the_buttons_does_not_stop_the_panel():
    panel = _panel()
    panel.compose(_frame(PANEL_W, PANEL_H), [], hud.HudState())
    panel.handle_command(Cmd("UI_PRESS", {"x": 300, "y": 100}))
    panel.handle_command(Cmd("UI_RELEASE", {"x": 300, "y": 100}))
    assert panel.exit_requested is False


def test_the_app_turns_detections_into_hits():
    """The app hands the panel frame-space boxes with a readable label."""
    from app.main import _hits_of

    detections = [
        Detection(x=10, y=20, w=30, h=40, class_id=2, score=0.75, label="car"),
        Detection(x=0, y=0, w=10, h=10, class_id=1, score=0.5),
    ]
    hits = _hits_of(detections, ["ball", "person"])

    assert len(hits) == 2
    assert hits[0].text == "car:0.75"
    assert hits[0].x == 10 and hits[0].h == 40
    # Without a label on the detection the class name comes from the list.
    assert hits[1].text == "person:0.50"


def test_the_app_reports_the_ai_fault_in_the_alert_bar():
    """A failed inference must be visible, not look like an empty scene."""
    from app.main import _panel_content

    class _Result:
        success = False
        error_code = "E_MODEL_LOAD_FAILED"
        result_data = {}

    class _Vm:
        def get_all_results(self):
            return {"main": {"ai_inference": _Result()}}

        def get_pipeline_fps(self, pipeline_id):
            return 12.5

    class _Ai:
        loaded = False
        labels = ["ball"]
        _engine_id = "onnxruntime"
        active_model = "yolo11n"

    class _Machine:
        ai = _Ai()
        camera_hub = None

    frame, hits, state = _panel_content(_Machine(), _Vm(), {})
    assert frame is None
    assert hits == []
    assert state.alert == "E_MODEL_LOAD_FAILED"
    assert state.link_ok is False
    assert state.fps == 12.5
    assert "onnxruntime" in state.line2


def test_the_exit_check_follows_the_panel():
    """The main loop stops when the panel says so."""
    from app.main import _exit_check

    class _Panel:
        exit_requested = False

    panel = _Panel()
    check = _exit_check(object(), panel)
    assert check() is False
    panel.exit_requested = True
    assert check() is True


def test_the_platform_offers_the_panel_only_when_a_window_opened():
    """A headless host gets no panel instead of an invisible one."""
    from framework.hal.platforms.linux import create_panel

    assert create_panel(headless=True) is None


def test_the_device_platform_has_no_panel():
    """The device keeps its LCD path: no panel hook, no window."""
    from app.main import _panel_of

    class _Machine:
        platform = "maixcam2"

    assert _panel_of(_Machine(), {}) is None


def test_the_frame_is_taken_from_the_camera_of_the_pipeline():
    """A pipeline id is not a camera id, and the panel must still get a frame.

    The vision module names them independently ("cam_main" versus "main"),
    so asking the hub for the pipeline id finds nothing and the panel would
    show black while the boxes are still drawn.
    """
    from app.main import _panel_content

    frame = np.full((480, 640, 3), (7, 8, 9), np.uint8)

    class _Camera:
        last_frame = frame

    class _Pipeline:
        camera = _Camera()

    class _Vm:
        def get_all_results(self):
            return {"cam_main": {}}

        def get_pipeline(self, pipeline_id):
            assert pipeline_id == "cam_main"
            return _Pipeline()

        def get_pipeline_fps(self, pipeline_id):
            return 20.0

    class _Machine:
        ai = None
        camera_hub = None

    got, hits, state = _panel_content(_Machine(), _Vm(), {})
    assert got is frame
    assert state.fps == 20.0


def test_the_panel_draws_the_frame_it_was_given():
    """The composed buffer really carries the picture, not only the HUD."""
    panel = _panel()
    frame = np.full((480, 640, 3), (7, 8, 9), np.uint8)
    buf = panel.compose(frame, [], hud.HudState())

    # Below the header bar the picture is visible.
    assert buf[300, 320].tolist() == [7, 8, 9]
