from __future__ import annotations

import maix.display
import maix.err

from framework.display.types import DisplayCaps, PixelFormat


class MaixCam2Display:
    def __init__(self) -> None:
        self._disp: maix.display.Display = maix.display.Display(
            open=True,
        )

    def show(self, frame) -> bool:
        if self._disp is None:
            return False
        try:
            err = self._disp.show(frame)
            return err == maix.err.Err.ERR_NONE
        except Exception:
            return False

    def caps(self) -> DisplayCaps:
        """平台上屏能力; 尺寸读真实 Display, 读不到退回 640x480。"""
        try:
            w, h = self._disp.width(), self._disp.height()
        except Exception:
            w = h = 0
        return DisplayCaps(
            width=w or 640, height=h or 480,
            native_format=PixelFormat.RGB888,
            accepted=frozenset((PixelFormat.NATIVE, PixelFormat.RGB888)),
            encode_in_show=True,
        )

    def width(self) -> int:
        return self._disp.width() if self._disp else 0

    def height(self) -> int:
        return self._disp.height() if self._disp else 0

    def close(self) -> None:
        if self._disp is not None:
            try:
                self._disp.close()
            except Exception:
                pass
            self._disp = None
