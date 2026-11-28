"""A custom HUD layout, start to finish.

This file is the whole cost of a second-party layout: one class with a
``name`` and a ``scene()`` method, plus one ``register`` call.  Nothing about
the vision module, the camera or the platform appears here - a layout takes a
state and a constraint and returns primitives.

Run it:

    python examples/custom_layout.py

It renders the layout onto a stand-in frame, simulates a tap on the bar it
declares clickable, and writes ``logs/layout_custom.png``.

To use it in an application, either import this module before the panel is
built, or point the config at it:

    display:
      layout: "examples.custom_layout:MinimalLayout"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.ui import layout
from framework.ui.measurer import Measurer
from framework.ui.metrics import Metrics
from framework.ui.overlay import Primitive, Rect, Text
from framework.ui.scene import HitBox, Scene
from framework.ui.space import Constraints
from framework.ui.view_state import ViewState

CARD = (18, 18, 20)
INK = (238, 238, 238)
MUTED = (150, 152, 158)
OK = (80, 215, 120)
FAULT = (225, 70, 60)
ACCENT = (70, 140, 235)


class MinimalLayout:
    """A deliberately plain layout, to show that none of this is magic."""

    name = "minimal"

    def scene(self, state: ViewState, constraints: Constraints,
              measure: Measurer) -> Scene:
        m = Metrics.of((constraints.max_width, constraints.max_height))
        items: list[Primitive] = []

        # A two-line header band: status on the left, fps on the right.  No
        # plate behind the text - the layout draws over the image directly.
        band = m.bar_h * 2
        status = f"{state.pipeline_id or 'CAM'}  {'OK' if state.ai_ok else 'FAULT'}"
        items.append(Text(m.pad, m.pad, status, m.text_big, OK if state.ai_ok else FAULT))

        fps = f"{state.fps:.0f} fps"
        fps_w = measure.text_size(fps, m.text_mid)[0]
        items.append(Text(constraints.max_width - m.pad - fps_w, m.pad, fps, m.text_mid, INK))

        # One summary line under the band.
        items.append(Text(m.pad, band + m.pad, f"targets {len(state.detections)}",
                          m.text_mid, MUTED))

        # A footer bar this layout declares clickable.  The scene's hit box is
        # what makes that true: Pointer tests the boxes, not a button list.
        # Outline only, for the same reason as the header: no fill.
        footer = Rect(m.pad, constraints.max_height - m.bar_h * 2,
                      constraints.max_width - 2 * m.pad, m.bar_h, CARD, fill=False)
        items.append(footer)
        items.append(Rect(footer.x, footer.y, footer.w, footer.h, ACCENT,
                          width=m.stroke))
        # Centre the label vertically: m.gap is a horizontal spacing and is
        # too large for a 30 px bar.
        label = "tap this bar"
        label_h = measure.text_size(label, m.text_small)[1]
        items.append(Text(footer.x + m.gap,
                          footer.y + max(2, (footer.h - label_h) // 2),
                          label, m.text_small, INK))

        hits = (HitBox("custom:footer", footer.x, footer.y, footer.w, footer.h),)
        return Scene(tuple(items), hits, constraints)


layout.register(MinimalLayout())


def main() -> int:
    import cv2
    import numpy as np

    from framework.cmd_queue import Cmd
    from framework.hal.platforms.linux.canvas import CvCanvas
    from framework.ui.measurer import CanvasMeasurer
    from framework.ui.pointer import Pointer

    width, height = 640, 480
    frame = np.full((height, width, 3), 46, np.uint8)
    cv2.circle(frame, (230, 300), 40, (70, 80, 205), -1)

    state = ViewState(pipeline_id="cam_main", fps=28.4, ai_ok=True,
                      detections=(object(), object()))

    canvas = CvCanvas(frame)
    custom = layout.get("minimal")
    scene = custom.scene(state, Constraints.of((width, height)),
                         CanvasMeasurer(canvas))
    scene.render(canvas)

    # The same three commands a mouse or a touchscreen produces.
    pointer = Pointer()
    pointer.attach(scene)
    pointer.apply(Cmd("UI_PRESS", {"x": width // 2, "y": height - 40}), now=0.0)
    event = pointer.apply(Cmd("UI_RELEASE", {"x": width // 2, "y": height - 40}), now=0.1)

    cv2.putText(frame, f"tap landed on: {event.id}", (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out = ROOT / "logs" / "layout_custom.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}  ({width}x{height})")
    print(f"hit boxes: {[box.id for box in scene.hits]}")
    print(f"tap -> {event.kind}: {event.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
