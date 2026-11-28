"""The desktop panel: the camera frame under the device HUD, in a cv2 window.

The panel is the desktop end of the runtime UI.  The semantic layer
(framework.ui) decides what is on it, this module draws it and owns the
window, so the same HUD runs on the device LCD and on the desktop.

Three facts shape the implementation:

* the panel is the only surface the user ever sees, so the frame is
  letterboxed into the panel and the detection boxes are mapped along with
  it - the boxes are therefore drawn by the HUD in panel pixels, not by the
  frame itself, and the glyphs are rendered once at the size they are shown.
* the window owns the keyboard, the mouse and the close request, so one
  object owns all three (CvWindow) and the exit signal is a flag, never the
  return value of a show() call.
* the frame the panel shows comes from the camera's newest capture, not from
  the inference queue: a display that steals frames would starve the
  detector.

The module lives under hal/platforms because it is the only place allowed to
speak to OpenCV; everything it draws is described by the platform-neutral
semantic layer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from framework.cmd_queue import CmdQueue
from framework.ui import hud
from framework.ui.panel import Panel, Rect
from framework.ui.pointer import Pointer
from framework.ui.widgets import Button

from .canvas import CvCanvas
from .window import CvMouseSource, CvWindow, CvWindowSink

__all__ = ["DesktopPanel", "EXIT_BUTTON", "default_buttons"]

EXIT_BUTTON = "EXIT"


def _button_label(hit_id: str) -> str:
    """``"button:EXIT"`` -> ``"EXIT"``; the layout compares against btn.label."""
    prefix = "button:"
    return hit_id[len(prefix):] if hit_id.startswith(prefix) else ""


def default_buttons(width: int, height: int) -> list[Button]:
    """The panel buttons, at the device's coordinates."""
    return [Button(EXIT_BUTTON, Rect(8, height - 46, 88, 38))]


class DesktopPanel:
    """Camera frame, device HUD and mouse input in one cv2 window.

    The panel owns no pixels beyond its own buffer: the caller composes a
    frame, hands it over with the HUD content, and the panel presents it.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        scale: int = 2,
        title: str = "Zulu-Walker",
        headless: bool = False,
        interval_ms: Optional[float] = None,
        buttons: Optional[Sequence[Button]] = None,
        layout_name: str = "desktop",
    ) -> None:
        self._panel = Panel(int(width), int(height))
        self._window = CvWindow(
            self._panel.width, self._panel.height, scale=scale, title=title,
            headless=headless,
        )
        self._sink = CvWindowSink(self._window, interval_ms=interval_ms)
        self._input = CvMouseSource(self._window)
        self._cmds = CmdQueue()
        self._buffer = np.zeros((self._panel.height, self._panel.width, 3), np.uint8)
        self._tmp: Optional[np.ndarray] = None
        self._tmp_size = (0, 0)
        self._buttons = list(buttons) if buttons is not None else default_buttons(
            self._panel.width, self._panel.height
        )
        self._pressed = ""
        self._exit_requested = False
        self._layout_name = layout_name
        self._pointer = Pointer()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #

    @property
    def has_gui(self) -> bool:
        """False when there is no window to draw into (headless host)."""
        return self._window.has_gui

    @property
    def exit_requested(self) -> bool:
        """True once the user asked to quit (q, Esc or the EXIT button)."""
        return self._exit_requested

    @property
    def size(self) -> tuple[int, int]:
        """The panel size in panel pixels."""
        return self._panel.size()

    @property
    def buttons(self) -> list[Button]:
        """The buttons the panel draws and hit-tests."""
        return list(self._buttons)

    # ------------------------------------------------------------------ #
    # composition
    # ------------------------------------------------------------------ #

    def compose(self, frame: Optional[np.ndarray], hits: Sequence[Any],
                state: Any) -> np.ndarray:
        """Letterbox the frame into the panel and draw the HUD on top.

        Args:
            frame: The camera frame (BGR) or None when there is none yet.
            hits: The detection boxes in FRAME pixels.
            state: The HUD content (a hud.HudState).

        Returns:
            The panel buffer.  It is the same object every call, so the
            caller presents it before composing the next one.
        """
        self._buffer[:] = (0, 0, 0)
        rect = None
        if frame is not None:
            rect = self._panel.fit(frame.shape[1], frame.shape[0])
            self._blit(frame, rect)
        scene = hud.draw(
            CvCanvas(self._buffer),
            self._state_in_panel_space(state, hits, rect, frame),
            self._buttons,
            self._pressed,
            layout_name=self._layout_name,
        )
        self._pointer.attach(scene)
        return self._buffer

    def _state_in_panel_space(self, state: Any, hits: Sequence[Any],
                              rect: Optional[Rect],
                              frame: Optional[np.ndarray]) -> Any:
        """Map the frame-space boxes into panel space."""
        if not hits or rect is None or frame is None:
            return state
        scale_x = rect.w / float(frame.shape[1])
        scale_y = rect.h / float(frame.shape[0])
        mapped = [
            hud.Hit(
                x=rect.x + int(hit.x * scale_x),
                y=rect.y + int(hit.y * scale_y),
                w=max(1, int(hit.w * scale_x)),
                h=max(1, int(hit.h * scale_y)),
                text=hit.text,
                color=hit.color,
            )
            for hit in hits
        ]
        return replace(state, hits=mapped)

    def _blit(self, frame: np.ndarray, rect: Rect) -> None:
        """Put the frame into the panel, resized when it does not fit."""
        frame_h, frame_w = frame.shape[:2]
        if (rect.w, rect.h) == (frame_w, frame_h):
            self._buffer[rect.y:rect.bottom, rect.x:rect.right] = frame
            return
        if self._tmp_size != (rect.w, rect.h):
            self._tmp = np.empty((rect.h, rect.w, 3), np.uint8)
            self._tmp_size = (rect.w, rect.h)
        cv2.resize(frame, (rect.w, rect.h), dst=self._tmp,
                   interpolation=cv2.INTER_AREA)
        self._buffer[rect.y:rect.bottom, rect.x:rect.right] = self._tmp

    # ------------------------------------------------------------------ #
    # presentation and input
    # ------------------------------------------------------------------ #

    def present(self) -> None:
        """Show the composed panel."""
        self._sink.push(self._buffer)
        self._sink.flush()

    def pump(self) -> None:
        """Drain the GUI events into the panel state.

        The window turns q, Esc and the close button into an EXIT command,
        and the mouse into press/release commands; the buttons are hit
        tested here, so the caller only has to look at exit_requested.
        """
        self._input.poll(self._cmds)
        for cmd in self._cmds.drain():
            self.handle_command(cmd)

    def handle_command(self, cmd: Any) -> None:
        """Apply one input command to the panel state.

        Split out from pump() so the button and exit handling can be driven
        without a window.  Hit-testing belongs to ``Pointer``, which tests the
        Scene the layout just produced: anything the layout drew as clickable
        is clickable, not only the fixed button list.
        """
        if cmd.kind == "EXIT":
            self._exit_requested = True
            return

        event = self._pointer.apply(cmd)
        if event is None:
            return

        if event.kind == "press":
            self._pressed = _button_label(event.id)
        else:
            # drag / cancel / tap all end the press visual.
            self._pressed = ""

        if event.kind == "tap" and event.id == f"button:{EXIT_BUTTON}":
            self._exit_requested = True

    def close(self) -> None:
        """Release the window."""
        self._sink.close()
