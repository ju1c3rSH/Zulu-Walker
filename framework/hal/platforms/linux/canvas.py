"""CvCanvas: the desktop backend of :mod:`framework.ui.canvas`.

Draws straight into a BGR uint8 ``ndarray`` - the container the Linux/Windows
camera already delivers - so nothing is converted before presentation.

Two measurements drive the implementation (640x480, OpenCV 5.1, WIN32UI):

* ``cv2.rectangle(..., thickness=-1, LINE_8)`` costs ~0.03 ms, while
  ``buf[y:y+h, x:x+w] = color`` costs ~1.2 ms because numpy broadcasts a
  3-byte value across the slice element by element. Fills therefore go through
  cv2, not numpy. cv2 also clips negative/oversized rectangles; a numpy slice
  silently drops the whole block instead.
* ``cv2.putText`` costs ~7 us per call, so a text cache buys nothing. Text is
  drawn directly, with the baseline offset computed per string.

Fills use ``LINE_8`` (hard edge, matching a numpy fill); strokes use
``LINE_AA``. Mixing the two inside one primitive would make edges depend on
``fill``, which the protocol does not promise.
"""

from __future__ import annotations

import cv2
import numpy as np

from framework.ui.canvas import Rgb, scale_for

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HERSHEY_UNIT_PX = 22.0


def _bgr(color: Rgb) -> tuple[int, int, int]:
    """(r, g, b) -> OpenCV's (b, g, r). The single conversion point."""
    r, g, b = color
    return int(b), int(g), int(r)


def _thickness(size: int) -> int:
    return max(1, int(round(size / 16.0)))


class CvCanvas:
    """Draws into a pre-allocated BGR buffer; never reallocates."""

    def __init__(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR buffer, got {frame.shape}")
        if not frame.flags["C_CONTIGUOUS"]:
            raise ValueError("buffer must be contiguous: a copy per draw would hide the cost")
        self._buf = frame
        self._h, self._w = frame.shape[:2]

    def size(self) -> tuple[int, int]:
        return self._w, self._h

    def buffer(self) -> np.ndarray:
        """The backing array, for the presenter. Not part of the protocol."""
        return self._buf

    def rect(self, x: int, y: int, w: int, h: int, color: Rgb,
             fill: bool = False, width: int = 1) -> None:
        if w <= 0 or h <= 0:
            return
        cv2.rectangle(self._buf, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)),
                      _bgr(color), -1 if fill else max(1, width),
                      cv2.LINE_8 if fill else cv2.LINE_AA)

    def line(self, x1: int, y1: int, x2: int, y2: int, color: Rgb,
             width: int = 1) -> None:
        cv2.line(self._buf, (int(x1), int(y1)), (int(x2), int(y2)),
                 _bgr(color), max(1, width), cv2.LINE_AA)

    def circle(self, x: int, y: int, r: int, color: Rgb,
               fill: bool = False, width: int = 1) -> None:
        if r <= 0:
            return
        cv2.circle(self._buf, (int(x), int(y)), int(r), _bgr(color),
                   -1 if fill else max(1, width),
                   cv2.LINE_8 if fill else cv2.LINE_AA)

    def text(self, x: int, y: int, text: str, size: int, color: Rgb) -> None:
        if not text:
            return
        scale = scale_for(size, _HERSHEY_UNIT_PX)
        thick = _thickness(size)
        (_tw, th), _base = cv2.getTextSize(text, _FONT, scale, thick)
        # OpenCV anchors on the baseline; the protocol anchors on the top-left.
        cv2.putText(self._buf, text, (int(x), int(y) + th), _FONT, scale,
                    _bgr(color), thick, cv2.LINE_AA)

    def text_size(self, text: str, size: int) -> tuple[int, int]:
        """Ink box, descenders included: ``getTextSize`` reports them apart."""
        if not text:
            return 0, 0
        scale = scale_for(size, _HERSHEY_UNIT_PX)
        (tw, th), base = cv2.getTextSize(text, _FONT, scale, _thickness(size))
        return int(tw), int(th + base)
