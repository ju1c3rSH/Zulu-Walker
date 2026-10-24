from __future__ import annotations

import maix.display
import maix.err

from framework.display.types import DisplayCaps, PixelFormat

# 面板物理尺寸; 显示对象打不开时 width()/height() 与 caps() 共用它。
_FALLBACK_W, _FALLBACK_H = 640, 480


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
        """平台上屏能力; 尺寸直接取 width()/height(), 保证两者永不矛盾。"""
        return DisplayCaps(
            width=self.width(), height=self.height(),
            native_format=PixelFormat.RGB888,
            accepted=frozenset((PixelFormat.NATIVE, PixelFormat.RGB888)),
            encode_in_show=True,
        )

    def width(self) -> int:
        """⚠ 打不开时退回面板物理尺寸而非 0: 0 会让调试页 resize(0,0) 直接失败。"""
        try:
            return self._disp.width() or _FALLBACK_W
        except Exception:
            return _FALLBACK_W

    def height(self) -> int:
        try:
            return self._disp.height() or _FALLBACK_H
        except Exception:
            return _FALLBACK_H

    def close(self) -> None:
        if self._disp is not None:
            try:
                self._disp.close()
            except Exception:
                pass
            self._disp = None
