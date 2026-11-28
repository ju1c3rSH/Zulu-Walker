"""The device HUD, as data.

Run: python -m pytest tests/test_device_layout.py -q

These assertions used to live in ``test_vision_overlay`` and called
``VisionManager._header_primitives`` and friends.  The layout moved out of the
vision module, so they moved with it: the same expectations, now expressed
against ``ViewState`` and a ``Scene``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.ui.overlay import Circle, Primitive, Rect, Text
from framework.ui.space import Constraints
from framework.ui.view_state import ViewState
from framework.ui.layouts.device import DeviceLayout

PANEL = Constraints.of((640, 480))


class FixedMeasurer:
    def text_size(self, text: str, size: int):
        return int(len(text) * size * 0.55), size + 5

    def fit(self, text: str, max_width: int, size: int, min_size: int = 10) -> int:
        while size > min_size and self.text_size(text, size)[0] > max_width:
            size -= 1
        return size


MEASURE = FixedMeasurer()


def _det(x=10, y=20, w=30, h=40, class_id=0, score=0.5, mask=None):
    return SimpleNamespace(x=x, y=y, w=w, h=h, class_id=class_id,
                           score=score, mask_stats=mask)


def _scene(**overrides) -> tuple:
    state = replace(
        ViewState(pipeline_id="cam_main", fps=30.0,
                  labels=("ball", "obstacle"), ai_ok=True, clock="12:00:00"),
        **overrides,
    )
    return DeviceLayout().scene(state, PANEL, MEASURE).primitives


def test_the_header_is_a_bar_an_fps_field_and_a_right_anchored_clock():
    items = _scene()

    bar = items[0]
    assert isinstance(bar, Rect) and bar.fill
    assert (bar.y, bar.h, bar.w) == (0, 30, 640)

    fps = items[1]
    assert isinstance(fps, Text)
    assert fps.text.startswith("cam_main")
    assert "FPS:30.0" in fps.text
    assert fps.x == 6

    clock = items[2]
    assert isinstance(clock, Text)
    # Anchored right: a centred clock collides with the fps field once the
    # glyphs are wider than the device font.
    assert clock.x > 320


def test_a_detection_becomes_a_box_and_an_index_inside_it():
    items = _scene(detections=(_det(),))

    box, index = items[3], items[4]
    assert isinstance(box, Rect)
    assert (box.x, box.y, box.w, box.h) == (10, 20, 30, 40)
    assert isinstance(index, Text) and index.text == "1"
    assert box.x <= index.x <= box.x + box.w
    assert box.y <= index.y <= box.y + box.h


def test_the_index_is_clamped_into_a_box_smaller_than_the_label():
    items = _scene(detections=(_det(x=100, y=100, w=8, h=8),))

    box, index = items[3], items[4]
    assert index.x >= box.x and index.y >= box.y


def test_a_mask_adds_a_centre_dot_and_an_area_label():
    mask = SimpleNamespace(area_px=120, center_x=25, center_y=40)
    items = _scene(detections=(_det(mask=mask),))

    assert any(isinstance(p, Circle) for p in items)
    labels = [p.text for p in items if isinstance(p, Text)]
    assert "area:120px" in labels


def test_an_empty_mask_adds_nothing_extra():
    mask = SimpleNamespace(area_px=0, center_x=0, center_y=0)
    items = _scene(detections=(_det(mask=mask),))

    assert not any(isinstance(p, Circle) for p in items)


def test_a_camera_fault_banner_sits_below_the_header():
    items = _scene(reconnecting=True)

    banner = [p for p in items if isinstance(p, Rect) and p.y > 30]
    assert banner and banner[-1].y >= 38
    assert any("CAMERA FAULT" in p.text for p in items if isinstance(p, Text))


def test_an_ai_error_carries_its_code():
    items = _scene(error_code="E_INFER_FAILED")

    texts = [p.text for p in items if isinstance(p, Text)]
    assert "AI FAULT - E_INFER_FAILED" in texts


def test_a_missing_model_reads_as_a_fault():
    items = _scene(ai_ok=False)

    texts = [p.text for p in items if isinstance(p, Text)]
    assert any("MODEL NOT LOADED" in t for t in texts)


def test_the_detection_list_stops_at_the_line_limit():
    items = _scene(detections=tuple(_det() for _ in range(40)))

    texts = [p.text for p in items if isinstance(p, Text)]
    rows = [t for t in texts if t[:1].isdigit() and ":" in t]
    assert len(rows) < 40
    assert any("more" in t for t in texts)


def test_an_empty_detection_list_draws_no_rows():
    items = _scene(detections=())

    assert not any(isinstance(p, Text) and p.text.startswith("1:") for p in items)


def test_the_list_can_be_disabled():
    items = _scene(detections=(_det(),), list_enabled=False)

    assert not any(isinstance(p, Text) and p.text.startswith("1:") for p in items)


def test_a_class_id_without_a_label_falls_back_to_a_stable_name():
    state = ViewState(pipeline_id="p", labels=("ball",), clock="")
    assert state.label_of(0) == "ball"
    assert state.label_of(7) == "cls7"
