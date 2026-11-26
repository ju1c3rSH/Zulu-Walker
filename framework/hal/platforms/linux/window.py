"""A cv2 window used as the device panel, plus the sink and input adapters.

One object owns the window: cv2 routes keyboard, mouse and the close request
through the same ``pollKey`` call, so splitting them across a sink and a
source would mean two owners and half the events staying unread.

Two measurements shape the design (640x480, OpenCV 5.1, WIN32UI):

* ``cv2.waitKey(1)`` costs ~15.5 ms per call - it sleeps until the next event.
  ``pollKey()`` returns at once, so the loop pumps with ``pollKey`` every tick
  and never stalls on the GUI.
* ``imshow`` into a ``WINDOW_NORMAL`` window rescaled every frame costs
  ~7.4 ms; pre-scaling into a buffer and showing it 1:1 into a
  ``WINDOW_AUTOSIZE`` window costs ~1.1 ms. The window is therefore fixed at
  ``panel x scale`` and never resized by the user.

Presentation and input are deliberately separate objects over the same window:
``CvWindowSink`` satisfies the FrameSink contract, ``CvMouseSource`` satisfies
InputSource. Exit is a ``Cmd("EXIT")`` - never a ``show()`` return value, which
``MaixLcdSink`` discards (framework/hal/sink.py:139).
"""

from __future__ import annotations

import time
import sys
from typing import Optional

import cv2
import numpy as np

from framework.cmd_queue import Cmd, CmdQueue
from framework.ui.panel import TouchEvent

_QUIT_KEYS = (27, ord("q"))


class CvWindow:
    """Fixed-size window that shows panel frames 1:1 after integer scaling."""

    def __init__(self, width: int, height: int, scale: int = 2,
                 title: str = "Zulu-Walker", headless: bool = False,
                 topmost: bool = False) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scale = max(1, int(scale))
        self.title = title
        self._closed = False
        self._events = []
        self._frame = np.zeros((self.height * self.scale, self.width * self.scale, 3),
                               np.uint8)
        self._checked = 0
        self.has_gui = False if headless else self._open()
        if self.has_gui:
            if topmost:
                cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback(self.title, self._on_mouse)

    def _open(self) -> bool:
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.title, self.width * self.scale,
                             self.height * self.scale)
            self._lock_geometry()
            return True
        except cv2.error:
            return False

    def _lock_geometry(self) -> None:
        """Refuse resize/maximize so the panel is never letterboxed by the WM.

        A maximized cv2 window shows the image at its own size and paints the
        rest of the client area gray - that border is the symptom, the resizable
        frame is the cause. Removing ``WS_THICKFRAME`` and ``WS_MAXIMIZEBOX``
        on Windows makes the window a fixed-size panel, like the device.
        Other platforms rely on ``WINDOW_AUTOSIZE``, which the X11/Qt backends
        honour for sizing but not always for user resizes.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, self.title)
            if not hwnd:
                return
            gwl_style = -16
            ws_thickframe, ws_maximizebox = 0x00040000, 0x00010000
            style = user32.GetWindowLongW(hwnd, gwl_style)
            user32.SetWindowLongW(hwnd, gwl_style,
                                  style & ~(ws_thickframe | ws_maximizebox))
            swp = 0x0001 | 0x0002 | 0x0004 | 0x0020  # NOSIZE|NOMOVE|NOZORDER|FRAMECHANGED
            user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, swp)
        except Exception:
            pass

    def _enforce_size(self) -> None:
        """Snap back if anything resized the window (WM, DPI change, user)."""
        want = (self.width * self.scale, self.height * self.scale)
        try:
            _x, _y, w, h = cv2.getWindowImageRect(self.title)
        except cv2.error:
            return
        if (w, h) != want:
            try:
                cv2.resizeWindow(self.title, *want)
                self._lock_geometry()
            except cv2.error:
                pass

    # ---------------- presentation ----------------

    def present(self, panel_frame: np.ndarray) -> bool:
        """Scale *panel_frame* into the window buffer and show it."""
        if not self.has_gui or self._closed:
            return False
        if self.scale == 1:
            self._frame[:] = panel_frame
        else:
            cv2.resize(panel_frame, (self.width * self.scale, self.height * self.scale),
                       dst=self._frame, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.title, self._frame)
        self._checked += 1
        if self._checked == 1:
            # The window is only mapped now: FindWindow needs it to exist.
            self._lock_geometry()
        elif self._checked % 30 == 0:
            self._enforce_size()
        return True

    # ---------------- input ----------------

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            kind = "press"
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            kind = "move"
        elif event == cv2.EVENT_LBUTTONUP:
            kind = "release"
        else:
            return
        self._events.append(TouchEvent(kind, x // self.scale, y // self.scale))

    def pump(self, cmds: CmdQueue) -> None:
        """Drain GUI events into *cmds*. Call exactly once per tick."""
        if not self.has_gui or self._closed:
            return
        key = cv2.pollKey()
        if key in _QUIT_KEYS:
            cmds.put(Cmd("EXIT"))
        elif cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            cmds.put(Cmd("EXIT"))
        for ev in self._events:
            cmds.put(Cmd(f"UI_{ev.kind.upper()}", {"x": ev.x, "y": ev.y}),
                     replace=(ev.kind == "move"))
        self._events.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.has_gui:
            try:
                cv2.destroyWindow(self.title)
                cv2.waitKey(1)
            except cv2.error:
                pass


class CvWindowSink:
    """FrameSink over :class:`CvWindow`; throttling mirrors MaixLcdSink."""

    def __init__(self, window: CvWindow, interval_ms: Optional[float] = None) -> None:
        self._win = window
        self._min_interval = max(0.0, float(interval_ms or 0.0)) / 1000.0
        self._latest = None
        self._pending = False
        self._last_presented = 0.0

    def push(self, frame) -> None:
        """Mark content dirty: a push means a freshly composed frame.

        Identity comparison would be wrong here - the composer reuses one
        buffer, so every frame after the first is the same object.
        """
        if frame is not None:
            self._latest = frame
            self._pending = True

    def reset(self) -> None:
        self._pending = True

    def flush(self) -> None:
        frame = self._latest
        if frame is None or not self._pending:
            return
        now = time.monotonic()
        if now - self._last_presented < self._min_interval:
            return
        if self._win.present(frame):
            self._pending = False
            self._last_presented = now

    def close(self) -> None:
        self._win.close()


class CvMouseSource:
    """InputSource over :class:`CvWindow`; owns the once-per-tick GUI pump."""

    name = "cv-mouse"

    def __init__(self, window: CvWindow) -> None:
        self._win = window

    def poll(self, cmds: CmdQueue) -> None:
        self._win.pump(cmds)
