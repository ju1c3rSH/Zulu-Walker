"""Main-page HUD, ported from the device's ``DisplayHUD``.

Source of truth: ``feat/maixcam1-port`` ``modules/zw_opencv_module/display/hud.py``.
Geometry is copied rather than reinvented, because the point of the port is
that a person who knows the device screen recognises the desktop one:

* a black bar of ``BAR_H + LINE2_H`` (30 + 24) at the top;
* line 1: ``MODE:...`` at x=6, then ``LINK:OK``, ``T:<found>`` and ``FPS:``
  right-aligned;
* line 2: the status string (codes / order / lock) in one colour;
* hits: a 2 px box plus the label 16 px above it, same colour, no plate;
* an optional one-line alert bar under the header.

Text sizes follow the device's ``draw_string(scale=...)``: 1.5 -> 22 px,
1.2 -> 18 px, 1.0 -> 15 px in panel pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from framework.ui.canvas import Canvas, Rgb, draw_fitted, fit_size
from framework.ui.widgets import Button

BLACK = (0, 0, 0)
WHITE = (238, 238, 238)
GRAY = (150, 152, 158)
GREEN = (60, 205, 110)
YELLOW = (240, 190, 60)
RED = (225, 70, 60)
MAGENTA = (235, 90, 235)
CYAN = (80, 215, 235)
GRAY_OFF = (105, 105, 110)

BAR_H = 30
LINE2_H = 24
ALERT_H = 20

T_BIG = 22
T_MID = 18
T_SMALL = 15


@dataclass(frozen=True, slots=True)
class Hit:
    """One detection box; ``text`` is drawn above it, device-style."""

    x: int
    y: int
    w: int
    h: int
    text: str = ""
    color: Rgb = GREEN


@dataclass
class HudState:
    """Main-page content. Names mirror the device's status dict."""

    mode: str = "IDLE"
    link_ok: bool = True
    found: bool = False
    fps: float = 0.0
    line2: str = ""
    hits: List[Hit] = field(default_factory=list)
    alert: str = ""


def draw(canvas: Canvas, state: HudState, buttons: Sequence[Button] = (),
         pressed: str = "") -> None:
    w, h = canvas.size()
    canvas.rect(0, 0, w, BAR_H + LINE2_H, BLACK, fill=True)
    _line1(canvas, state, w)
    _line2(canvas, state)
    if state.alert:
        canvas.rect(0, BAR_H + LINE2_H, w, ALERT_H, RED, fill=True)
        canvas.text(6, BAR_H + LINE2_H + 2, state.alert, T_SMALL, (16, 8, 8))
    for hit in state.hits:
        canvas.rect(hit.x, hit.y, hit.w, hit.h, hit.color, width=2)
        if hit.text:
            _th = canvas.text_size(hit.text, T_SMALL)[1]
            canvas.text(hit.x, max(2, hit.y - _th - 2), hit.text, T_SMALL,
                        hit.color)
    if buttons:
        _buttons(canvas, buttons, h, pressed)


def _line1(canvas: Canvas, state: HudState, w: int) -> None:
    mode_text = f"MODE:{state.mode}"
    # Reserve the right cluster (T:/FPS) so a long mode name cannot run into it.
    mode_size = fit_size(canvas, mode_text, w - 380, T_BIG, min_size=14)
    canvas.text(6, 4, mode_text, mode_size, WHITE)
    mw, _ = canvas.text_size(mode_text, mode_size)

    link = "LINK:OK" if state.link_ok else "LINK:--"
    canvas.text(6 + mw + 12, 4, link, T_BIG, GREEN if state.link_ok else GRAY)

    found = "FOUND" if state.found else "-"
    canvas.text(w - 240, 4, f"T:{found}", T_MID, GREEN if state.found else GRAY)
    canvas.text(w - 130, 4, f"FPS:{state.fps:4.1f}", T_MID, WHITE)


def _line2(canvas: Canvas, state: HudState) -> None:
    y = BAR_H + 2
    if state.mode == "IDLE" and not state.line2:
        canvas.text(6, y, "IDLE (wait CMD_REQUEST)", T_MID, GRAY)
        return
    canvas.text(6, y, state.line2, T_MID, GREEN if state.found else YELLOW)


def _buttons(canvas: Canvas, buttons: Sequence[Button], h: int,
             pressed: str) -> None:
    for btn in buttons:
        active = btn.label == pressed
        canvas.rect(btn.rect.x, btn.rect.y, btn.rect.w, btn.rect.h,
                    (20, 20, 22), fill=True)
        canvas.rect(btn.rect.x, btn.rect.y, btn.rect.w, btn.rect.h,
                    GREEN if active else WHITE, width=2)
        draw_fitted(canvas, btn.rect, btn.label, T_SMALL,
                    GREEN if active else WHITE, pad=6)
