"""The desktop panel's HUD: MODE / LINK / T: / FPS over the camera image.

Geometry is ported from the device screen on purpose: a person who knows the
board should recognise the laptop window.  What changed is that no constant
is a position any more - the right cluster is carved off with ``take_right``
and the left side gets what remains, so a wider or narrower surface reflows
instead of drifting.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from framework.ui import layout
from framework.ui import metrics as metrics_mod
from framework.ui.measurer import Measurer
from framework.ui.overlay import Primitive, Rect, Text
from framework.ui.scene import HitBox, Scene
from framework.ui.space import Constraints
from framework.ui.widgets import Button

BLACK = (0, 0, 0)
WHITE = (238, 238, 238)
GRAY = (150, 152, 158)
GREEN = (60, 205, 110)
YELLOW = (240, 190, 60)
RED = (225, 70, 60)
INK = (16, 8, 8)
BUTTON_FILL = (20, 20, 22)

ALERT_INK = INK


class DesktopLayout:
    """Two text lines, an optional alert bar, detection boxes and buttons."""

    name = "desktop"

    def scene(self, state: Any, constraints: Constraints, measure: Measurer,
              buttons: Sequence[Button] = (), pressed: str = "") -> Scene:
        m = metrics_mod.Metrics.of(size_of(constraints))
        items: List[Primitive] = []

        items.append(Rect(constraints.x, constraints.y, constraints.max_width,
                          m.header_h, BLACK, fill=True))
        items.extend(self._line1(state, constraints, m, measure))
        items.extend(self._line2(state, constraints, m))

        top = constraints.y + m.header_h
        if state.alert:
            items.append(Rect(constraints.x, top, constraints.max_width,
                              m.alert_h, RED, fill=True))
            items.append(Text(m.pad, top + 2, state.alert, m.text_small, ALERT_INK))

        items.extend(self._hits(state, m, measure))
        hits = self._buttons(buttons, pressed, m, measure, items)

        return Scene(tuple(items), tuple(hits), constraints)

    # ------------------------------------------------------------------ #

    def _line1(self, state: Any, area: Constraints,
               m: metrics_mod.Metrics, measure: Measurer) -> List[Primitive]:
        """``MODE:`` on the left, then ``LINK:``, with ``T:`` and ``FPS:`` right.

        The right cluster is reserved first: a long mode name then shrinks to
        whatever is left instead of running underneath it.
        """
        row = area.inset(left=m.pad, right=m.pad)

        fps_text = f"FPS:{state.fps:4.1f}"
        fps_slot, row = row.take_right(_width(measure, fps_text, m.text_mid), m.gap)

        found_text = "FOUND" if state.found else "-"
        target = f"T:{found_text}"
        target_slot, row = row.take_right(
            _width(measure, target, m.text_mid), m.gap
        )

        link_text = "LINK:OK" if state.link_ok else "LINK:--"
        link_w = _width(measure, link_text, m.text_big)
        mode_text = f"MODE:{state.mode}"
        mode_size = measure.fit(
            mode_text, max(0, row.max_width - link_w - m.gap), m.text_big, 14
        )
        mode_w = _width(measure, mode_text, mode_size)

        y = max(2, (m.bar_h - measure.text_size(mode_text, mode_size)[1]) // 2)
        return [
            Text(row.x, y, mode_text, mode_size, WHITE),
            Text(row.x + mode_w + m.gap, y, link_text, m.text_big,
                 GREEN if state.link_ok else GRAY),
            Text(target_slot.x, y, target, m.text_mid,
                 GREEN if state.found else GRAY),
            Text(fps_slot.x, y, fps_text, m.text_mid, WHITE),
        ]

    def _line2(self, state: Any, area: Constraints,
               m: metrics_mod.Metrics) -> List[Primitive]:
        idle = state.mode == "IDLE" and not state.line2
        text = "IDLE (wait CMD_REQUEST)" if idle else state.line2
        if not text:
            return []
        color = GRAY if idle else (GREEN if state.found else YELLOW)
        return [Text(area.x + m.pad, area.y + m.bar_h + 2, text, m.text_mid, color)]

    def _hits(self, state: Any, m: metrics_mod.Metrics,
              measure: Measurer) -> List[Primitive]:
        """A detection box with a dark halo and its label above it.

        The halo is not decoration: a green box on a green object disappears
        without it, and the tracker's boxes land on exactly that kind of scene.
        """
        items: List[Primitive] = []
        for hit in state.hits:
            items.append(Rect(hit.x, hit.y, hit.w, hit.h, BLACK, width=m.halo))
            items.append(Rect(hit.x, hit.y, hit.w, hit.h, hit.color, width=m.stroke))
            if not hit.text:
                continue
            text_h = measure.text_size(hit.text, m.text_small)[1]
            items.append(Text(hit.x, max(2, hit.y - text_h - 2), hit.text,
                              m.text_small, hit.color))
        return items

    def _buttons(self, buttons: Sequence[Button], pressed: str,
                 m: metrics_mod.Metrics, measure: Measurer,
                 items: List[Primitive]) -> List[HitBox]:
        hits: List[HitBox] = []
        for btn in buttons:
            x, y, w, h = btn.rect.x, btn.rect.y, btn.rect.w, btn.rect.h
            active = btn.label == pressed
            ink = GREEN if active else WHITE
            items.append(Rect(x, y, w, h, BUTTON_FILL, fill=True))
            items.append(Rect(x, y, w, h, ink, width=m.stroke))

            size = measure.fit(btn.label, w - 2 * m.pad, m.text_small)
            text_w, text_h = measure.text_size(btn.label, size)
            items.append(Text(
                x + max(m.pad, (w - text_w) // 2),
                y + max(0, (h - text_h) // 2),
                btn.label, size, ink,
            ))
            hits.append(HitBox(f"button:{btn.label}", x, y, w, h))
        return hits


def _width(measure: Measurer, text: str, size: int) -> int:
    return int(measure.text_size(text, size)[0])


def size_of(constraints: Constraints) -> Tuple[int, int]:
    return constraints.max_width, constraints.max_height


layout.register(DesktopLayout())
