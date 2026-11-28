"""Main-page HUD: the state types, the palette, and the panel entry point.

The layout itself lives in ``framework.ui.layouts.desktop``.  This module
keeps what other code depends on:

* ``HudState`` / ``Hit`` - the content the panel and the demo scripts build;
* the palette - ``tuner.py`` imports its colours from here;
* ``draw()`` - the one call ``LinuxPanel.compose`` makes, kept so the panel
  does not need to know the layout registry exists.

Sizes are no longer constants: they come from ``framework.ui.metrics``, which
derives them from the surface height, so the same layout is legible at
552x368 and 640x480.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from framework.ui.canvas import Canvas, Rgb
from framework.ui.layout import get as get_layout
from framework.ui.measurer import CanvasMeasurer
from framework.ui.space import Constraints
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

DEFAULT_LAYOUT = "desktop"


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
         pressed: str = "", layout_name: str = DEFAULT_LAYOUT) -> None:
    """Draw *state* onto *canvas* using a registered layout."""
    scene = get_layout(layout_name).scene(
        state,
        Constraints.of(canvas.size()),
        CanvasMeasurer(canvas),
        buttons,
        pressed,
    )
    scene.render(canvas)
