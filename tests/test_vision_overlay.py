"""Unit tests for the vision HUD as data.

Run: python -m pytest tests/test_vision_overlay.py -q

Before the overlay refactor the only way to see whether the device HUD was
correct was to flash a board: the drawing code needed ``maix.image`` and the
desktop path never reached it.  These tests assert the primitive list
instead, which is the part that carries the layout decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.ui.overlay import Circle, Rect, Text
from modules.zw_opencv_module.vision_manager import VisionManager

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeFrame:
    """A stand-in for maix.Image: size, copy, nothing else."""

    def __init__(self, width: int = 320, height: int = 240) -> None:
        self._w, self._h = width, height

    def width(self):
        return self._w

    def height(self):
        return self._h

    def copy(self):
        return FakeFrame(self._w, self._h)


class RecordingCanvas:
    def __init__(self) -> None:
        self.calls = []

    def size(self):
        return 320, 240

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


class FakeCamera:
    def __init__(self, raw=None, serial=None, reconnecting=False) -> None:
        self.last_raw = raw
        self.frame_serial = serial
        self.is_reconnecting = reconnecting


class FakePipe:
    def __init__(self, camera, results=None) -> None:
        self.camera = camera
        self.last_results = results or {}


def _manager() -> VisionManager:
    """A manager that owns no camera and runs no thread."""
    return VisionManager(camera_hub=object())


def _det(**overrides):
    base = dict(x=10, y=20, w=30, h=40, class_id=0, score=0.5, mask_stats=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_module_imports_no_drawing_backend():
    """The whole point: no maix import on the module's dependency path."""
    source = (
        REPO_ROOT / "modules/zw_opencv_module/vision_manager.py"
    ).read_text(encoding="utf-8")
    assert "import maix" not in source
    assert "maix.image" not in source


def test_the_header_is_a_bar_an_fps_field_and_a_clock():
    primitives = _manager()._header_primitives("cam_main", 30.0, 640)

    bar, fps_field, clock = primitives
    assert isinstance(bar, Rect)
    assert (bar.y, bar.h, bar.w) == (0, 32, 640)
    assert bar.fill

    assert isinstance(fps_field, Text)
    assert fps_field.text.startswith("cam_main")
    assert "FPS:30.0" in fps_field.text
    assert fps_field.x == 6

    # Right-aligned so the two fields cannot collide whatever the glyph widths.
    assert isinstance(clock, Text)
    assert clock.align == "right"
    assert clock.x == 632


def test_a_detection_becomes_a_box_and_an_index_inside_it():
    primitives = _manager()._detection_primitives([_det()])

    box, index = primitives
    assert isinstance(box, Rect)
    assert (box.x, box.y, box.w, box.h) == (10, 20, 30, 40)
    assert isinstance(index, Text)
    assert index.text == "1"
    assert box.x <= index.x <= box.x + box.w
    assert box.y <= index.y <= box.y + box.h


def test_the_index_is_clamped_into_a_box_smaller_than_the_label():
    """A one-digit label in an 8 px box used to be drawn outside it."""
    box, index = _manager()._detection_primitives([_det(x=100, y=100, w=8, h=8)])

    assert index.x >= box.x
    assert index.y >= box.y


def test_a_mask_adds_a_centre_dot_and_an_area_label():
    stats = SimpleNamespace(area_px=120, center_x=25, center_y=40)
    primitives = _manager()._detection_primitives([_det(mask_stats=stats)])

    assert any(isinstance(p, Circle) for p in primitives)
    labels = [p.text for p in primitives if isinstance(p, Text)]
    assert "area:120px" in labels


def test_an_empty_mask_contributes_no_primitives():
    stats = SimpleNamespace(area_px=0, center_x=0, center_y=0)
    primitives = _manager()._detection_primitives([_det(mask_stats=stats)])

    assert [p for p in primitives if isinstance(p, Circle)] == []


def test_the_camera_fault_banner_sits_below_the_header():
    bar, text = _manager()._fault_primitives("camera", 640)

    assert bar.y == 40
    assert "CAMERA FAULT" in text.text


def test_an_ai_error_carries_its_code():
    _bar, text = _manager()._fault_primitives("ai_error", 640, "E_INFER_FAILED")

    assert text.text == "AI FAULT - E_INFER_FAILED"


def test_a_missing_model_reads_as_a_fault():
    bar, text = _manager()._fault_primitives("ai_fault", 640)

    assert bar.y == 0
    assert "MODEL NOT LOADED" in text.text


def test_the_detection_list_stops_at_the_line_limit():
    manager = _manager()
    manager._ai = SimpleNamespace(labels=["ball"])
    primitives = manager._detection_list_primitives([_det() for _ in range(12)])

    assert len(primitives) == manager._LIST_MAX_LINES + 1
    assert primitives[-1].text.endswith("more")


def test_an_empty_detection_list_draws_nothing():
    manager = _manager()
    manager._ai = SimpleNamespace(labels=["ball"])
    assert manager._detection_list_primitives([]) == []


def test_a_frame_width_comes_from_either_frame_kind():
    assert VisionManager._frame_width(SimpleNamespace(width=lambda: 320)) == 320
    assert VisionManager._frame_width(SimpleNamespace(shape=(480, 640, 3))) == 640
    assert VisionManager._frame_width(object()) == 0


def test_the_display_frame_is_kept_even_without_a_renderer():
    """No Canvas factory means no HUD, not no frame."""
    manager = _manager()
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=FakeFrame(), serial=1))

    manager._update_display_frame()

    assert manager._display_frame is not None


def test_no_native_frame_means_no_render_and_no_copy():
    manager = _manager()
    rendered = []
    manager._canvas_factory = lambda frame: rendered.append(frame) or RecordingCanvas()
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=None, serial=1))

    manager._update_display_frame()

    assert rendered == []
    assert manager._display_frame is None


def test_a_fresh_frame_is_rendered_and_stored():
    manager = _manager()
    canvas = RecordingCanvas()
    manager._canvas_factory = lambda frame: canvas
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=FakeFrame(), serial=1))

    manager._update_display_frame()

    # No AI loaded, so the HUD is the fault banner: a background and a label.
    assert canvas.calls == ["rect", "text"]
    assert manager._display_frame is not None


def test_the_same_sensor_frame_is_not_rebuilt():
    manager = _manager()
    built = []

    def factory(frame):
        built.append(frame)
        return RecordingCanvas()

    manager._canvas_factory = factory
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=FakeFrame(), serial=7))

    manager._update_display_frame()
    manager._update_display_frame()

    assert len(built) == 1


def test_rendering_can_be_disabled_at_runtime():
    manager = _manager()
    built = []
    manager._canvas_factory = lambda frame: built.append(frame) or RecordingCanvas()
    manager.set_render_enabled(False)
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=FakeFrame(), serial=1))

    manager._update_display_frame()

    assert built == []
    assert manager._display_frame is None


def test_a_broken_renderer_does_not_escape_the_display_step():
    """The vision loop guard would otherwise drop the frame rate to 1 Hz."""
    manager = _manager()

    def exploding(frame):
        raise RuntimeError("canvas backend lost its surface")

    manager._canvas_factory = exploding
    manager._pipelines["p"] = FakePipe(FakeCamera(raw=FakeFrame(), serial=1))

    manager._update_display_frame()

    assert manager._display_frame is not None
    assert manager._hud_error_logged is True
