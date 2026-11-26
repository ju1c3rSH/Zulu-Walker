"""Threshold page, ported from the device's ``GksThresholdTuner``.

Source of truth: ``feat/maixcam1-port``
``modules/zw_opencv_module/debug/threshold_tuner.py``. The device page is a
group bar (30 px) + action bar (30 px) + one info line + a 2x3 slider grid of
50 px rows over the live preview. All of that geometry is copied here, so the
desktop page and the LCD page differ only in what feeds them.

Replicated: positions, colours (green = selected, yellow = reset, magenta =
blob), ``NAME:value`` labels, and the latch-on-press behaviour.
Not replicated: the six-group LAB session persisted to ``/my-threshold`` - that
is competition business state, and this page takes plain sliders instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from framework.ui.canvas import Canvas, Rgb, draw_fitted, fit_size
from framework.ui.hud import GREEN, GRAY_OFF, MAGENTA, WHITE, YELLOW
from framework.ui.panel import Rect
from framework.ui.widgets import Slider

GROUP_H = 30
ACTION_H = 30
SLIDER_H = 50
T_LABEL = 18
T_SMALL = 15
T_VALUE = 12


@dataclass(frozen=True, slots=True)
class Action:
    label: str
    x: int
    y: int
    w: int
    h: int
    color: Rgb = WHITE


def group_actions(width: int) -> Tuple[Action, Action, Action]:
    """Exit / SAVE / RESET at the device's coordinates."""
    y = GROUP_H
    return (Action("< Exit", 2, y, 90, ACTION_H - 4, WHITE),
            Action("SAVE", width - 200, y, 90, ACTION_H - 4, GREEN),
            Action("RESET", width - 104, y, 100, ACTION_H - 4, YELLOW))


def slider_rects(width: int, height: int) -> list:
    """2 columns x 3 rows anchored to the bottom edge, as on the device."""
    col_w = width // 2
    top = height - 3 * SLIDER_H
    out = []
    for i in range(6):
        col, row = i % 2, i // 2
        x0 = col * col_w + 10
        y0 = top + row * SLIDER_H + 6
        out.append((x0, y0, col_w - 60))
    return out


def slider_for(key: str, geom, lo: float, hi: float, value: float) -> Slider:
    """A slider whose track matches :func:`slider_rects` exactly."""
    x0, y0, w = geom
    return Slider(key=key, label=key, rect=Rect(x0, y0 - 6, w + 7, 36),
                  lo=lo, hi=hi, value=value, track_inset=0, readout_w=0)


def draw(canvas: Canvas, groups: Sequence[Tuple[str, bool]],
         sliders: Sequence[Slider], actions: Sequence[Action],
         info: str = "", boxes: Sequence[Tuple[int, int, int, int]] = ()) -> None:
    w, _h = canvas.size()

    bw = w // max(1, len(groups))
    for i, (name, active) in enumerate(groups):
        rect = (i * bw + 2, 2, bw - 4, GROUP_H - 6)
        color = GREEN if active else GRAY_OFF
        canvas.rect(rect[0], rect[1], rect[2], rect[3], color, width=2)
        draw_fitted(canvas, Rect(*rect), name, T_LABEL, color, pad=6)

    for a in actions:
        canvas.rect(a.x, a.y, a.w, a.h, a.color, width=2)
        draw_fitted(canvas, Rect(a.x, a.y, a.w, a.h), a.label, T_LABEL,
                    a.color, pad=6)

    if info:
        canvas.text(8, GROUP_H + ACTION_H + 2, info, T_SMALL, WHITE)

    for (x, y, bw_, bh) in boxes:
        canvas.rect(x, y, bw_, bh, MAGENTA, width=2)

    for s in sliders:
        _slider(canvas, s)


def _slider(canvas: Canvas, s: Slider) -> None:
    """Label on top, track 20 px lower: the device stacks them too, and the
    two must not touch or the value becomes unreadable at 14 px."""
    t = s.track
    track_y = s.rect.y + 24
    canvas.rect(t.x, track_y, t.w, 8, (58, 58, 62), fill=True)
    kx = s.knob_x()
    canvas.rect(t.x, track_y, max(1, kx - t.x), 8, GREEN, fill=True)
    canvas.rect(kx - 4, track_y - 4, 8, 16, GREEN, fill=True)
    label = f"{s.key}:{s.readout()}"
    canvas.text(t.x, s.rect.y + 2, label,
                fit_size(canvas, label, max(t.w, 60), T_VALUE), WHITE)
