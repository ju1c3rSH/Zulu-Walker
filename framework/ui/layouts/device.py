"""The device HUD: pipeline + fps + clock, boxes, a list, fault banners.

This is the layout that used to live inside ``vision_manager`` as five
``_xxx_primitives`` helpers.  Moving it here changed two things and only two:

* the sizes come from ``Metrics`` instead of literals (``18`` -> ``m.text_mid``),
  so the same code reads correctly on a 552x368 panel;
* the geometry is expressed against a ``Constraints`` rather than
  ``frame_size(frame)[0]``, so nothing here knows what a frame is.

Everything else - which elements appear in which state, the colours, the
device's habit of putting the index inside the box - is unchanged.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from framework.ui import layout
from framework.ui import metrics as metrics_mod
from framework.ui.measurer import Measurer
from framework.ui.overlay import Circle, Primitive, Rect, Text
from framework.ui.scene import Scene
from framework.ui.space import Constraints
from framework.ui.view_state import ViewState

BLACK = (0, 0, 0)
WHITE = (238, 238, 238)
GREEN = (60, 205, 110)
YELLOW = (240, 190, 60)
RED = (225, 70, 60)
BLUE = (70, 140, 235)


class DeviceLayout:
    """The on-device screen: dense, no titles, one line per fact."""

    name = "device"

    def scene(self, state: Any, constraints: Constraints,
              measure: Measurer) -> Scene:
        m = metrics_mod.Metrics.of(size_of(constraints))
        items: List[Primitive] = []

        if state.ai_ok:
            items.append(Rect(constraints.x, constraints.y,
                              constraints.max_width, m.bar_h, BLACK, fill=True))
            items.extend(self._header(state, constraints, m, measure))
            items.extend(self._detections(state, m, measure))
            if state.list_enabled:
                items.extend(self._list(state, m, measure))

        if state.error_code:
            items.extend(self._banner(f"AI FAULT - {state.error_code}", m,
                                      RED, WHITE, constraints.max_width))
        elif not state.ai_ok:
            items.extend(self._banner("AI FAULT - MODEL NOT LOADED", m,
                                      RED, WHITE, constraints.max_width))
        if state.reconnecting:
            items.extend(self._banner("CAMERA FAULT - RECONNECTING", m,
                                      YELLOW, BLACK, constraints.max_width,
                                      top=m.bar_h + 8))

        return Scene(tuple(items), (), constraints)

    # ------------------------------------------------------------------ #

    def _header(self, state: ViewState, area: Constraints,
                m: metrics_mod.Metrics, measure: Measurer) -> List[Primitive]:
        """Pipeline and fps anchored left, the clock anchored right.

        Both are anchored rather than centred: a centred clock collides with
        the fps field as soon as the glyphs are wider than the device font
        (they are, on a desktop backend).
        """
        row = area.inset(left=m.pad, right=m.pad)
        fps_text = f"{state.pipeline_id}  FPS:{state.fps:.1f}"
        y = max(2, (m.bar_h - measure.text_size(fps_text, m.text_mid)[1]) // 2)

        items = [Text(row.x, y, fps_text, m.text_mid, WHITE)]
        if state.clock:
            clock_w = int(measure.text_size(state.clock, m.text_mid)[0])
            items.append(Text(row.right - clock_w, y, state.clock,
                              m.text_mid, WHITE))
        return items

    def _detections(self, state: ViewState, m: metrics_mod.Metrics,
                    measure: Measurer) -> List[Primitive]:
        items: List[Primitive] = []
        for index, det in enumerate(state.detections, start=1):
            items.append(Rect(det.x, det.y, det.w, det.h, GREEN, width=m.stroke))

            # The index sits inside the box's bottom-right corner.  A label in
            # a box smaller than the label is clamped back to the origin, which
            # is what the previous implementation guarded against with max().
            label = str(index)
            label_w = int(len(label) * m.text_small * 0.55)
            label_h = int(m.text_small * 1.45)
            nx = max(det.x, det.x + det.w - label_w - 2)
            ny = max(det.y, det.y + det.h - label_h)
            items.append(Text(nx, ny, label, m.text_small, BLUE))

            items.extend(self._mask(det, m, measure))
        return items

    def _mask(self, det: Any, m: metrics_mod.Metrics,
              measure: Measurer) -> List[Primitive]:
        stats = getattr(det, "mask_stats", None)
        if stats is None or not getattr(stats, "area_px", 0):
            return []
        return [
            Circle(int(stats.center_x), int(stats.center_y), 3, RED, fill=True),
            Text(det.x, det.y + det.h + 4, f"area:{stats.area_px}px",
                 m.text_big, YELLOW),
        ]

    def _list(self, state: ViewState, m: metrics_mod.Metrics,
              measure: Measurer) -> List[Primitive]:
        """The left-hand list: ``1: ball 0.91``, one line per detection."""
        if not state.detections:
            return []

        items: List[Primitive] = []
        top = m.bar_h + 4
        line_spacing = m.text_big + 8
        limit = max(1, (m.height - top) // line_spacing)

        for index, det in enumerate(state.detections[:limit]):
            name = state.label_of(det.class_id)
            text = f"{index + 1}: {name} {det.score:.2f}"
            items.append(Text(m.pad, top + index * line_spacing, text,
                              m.text_big, BLUE))

        hidden = len(state.detections) - limit
        if hidden > 0:
            items.append(Text(m.pad, top + limit * line_spacing,
                              f"+{hidden} more", m.text_small, BLUE))
        return items

    def _banner(self, text: str, m: metrics_mod.Metrics, background: Any,
                ink: Any, width: int, top: int = 0) -> List[Primitive]:
        return [
            Rect(0, top, width, m.bar_h, background, fill=True),
            Text(m.pad, top + 2, text, m.text_mid, ink),
        ]


def size_of(constraints: Constraints) -> Tuple[int, int]:
    return constraints.max_width, constraints.max_height


layout.register(DeviceLayout())
