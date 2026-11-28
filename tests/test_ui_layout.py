"""Layout arithmetic: slicing, scaling, and fitting.

Run: python -m pytest tests/test_ui_layout.py -q

These test the primitives a layout is built from, not the pixels a backend
produces.  A wrong ``take_right`` or a metric that does not reproduce the
reference surface shifts every label at once, and that is cheaper to catch
here than on a board.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.ui.hud import HudState
from framework.ui.layout import get
from framework.ui.metrics import Metrics
from framework.ui.space import Constraints


class FixedMeasurer:
    """Deterministic widths, so layout assertions do not depend on a font."""

    def text_size(self, text: str, size: int):
        return int(len(text) * size * 0.55), size + 5

    def fit(self, text: str, max_width: int, size: int, min_size: int = 10) -> int:
        while size > min_size and self.text_size(text, size)[0] > max_width:
            size -= 1
        return size


def test_take_right_reserves_a_slice_at_the_edge():
    area = Constraints(0, 0, 640, 30)

    taken, rest = area.take_right(100, gap=10)

    assert (taken.x, taken.max_width) == (540, 100)
    assert rest.x == 0
    assert rest.max_width == 530


def test_take_left_reserves_a_slice_at_the_origin():
    area = Constraints(0, 0, 640, 30)

    taken, rest = area.take_left(120, gap=8)

    assert (taken.x, taken.max_width) == (0, 120)
    assert (rest.x, rest.max_width) == (128, 512)


def test_a_slice_never_extends_past_the_area():
    area = Constraints(0, 0, 50, 30)

    taken, rest = area.take_right(200)

    assert taken.max_width == 50
    assert rest.max_width == 0


def test_metrics_reproduce_the_reference_surface():
    """640x480 must give back the numbers the device layout was tuned at."""
    m = Metrics.of((640, 480))

    assert (m.bar_h, m.line2_h, m.alert_h) == (30, 24, 20)
    assert (m.text_big, m.text_mid, m.text_small) == (22, 18, 15)
    assert (m.pad, m.gap) == (6, 12)


def test_metrics_scale_down_without_collapsing():
    m = Metrics.of((552, 368))

    assert m.bar_h < 30
    assert m.text_big >= 14 and m.text_small >= 10


def test_the_desktop_layout_stays_inside_a_small_panel():
    measure = FixedMeasurer()
    panel = Constraints.of((552, 368))
    state = HudState(mode="TRACK", found=True, fps=28.4,
                     line2="LOCK 3 | order A-B-C",
                     alert="E_INFER_FAILED")

    scene = get("desktop").scene(state, panel, measure)

    scene.validate(measure)  # raises LayoutOverflow if a label overruns


def test_buttons_become_hit_boxes_named_by_their_label():
    from framework.ui.layouts.desktop import DesktopLayout
    from framework.ui.panel import Rect as PanelRect

    class Button:
        label = "EXIT"

        def __init__(self):
            self.rect = PanelRect(10, 400, 90, 40)

    scene = DesktopLayout().scene(
        HudState(), Constraints.of((640, 480)), FixedMeasurer(), [Button()]
    )

    box = scene.hit_test(50, 420)
    assert box is not None and box.id == "button:EXIT"
    assert scene.hit_test(400, 100) is None
