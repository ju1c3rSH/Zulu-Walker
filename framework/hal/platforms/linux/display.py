# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np

from framework.display.types import DisplayCaps, PixelFormat

# 无窗口/未上过帧时的画布兜底（与相机默认分辨率一致）。
_DEFAULT_W, _DEFAULT_H = 640, 480


class LinuxDisplay:
    """Linux 桌面窗口显示; 只吃 BGR888 numpy, 不做格式转换。"""

    _window_name = "Zulu-Walker"

    def __init__(self) -> None:
        self._w, self._h = _DEFAULT_W, _DEFAULT_H

    def show(self, frame: np.ndarray) -> bool:
        self._remember(frame)
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return True

    def caps(self) -> DisplayCaps:
        """能力声明: 仅 BGR888; 尺寸取最近一帧, 没有则用默认值。"""
        return DisplayCaps(
            width=self._w, height=self._h,
            native_format=PixelFormat.BGR888,
            accepted=frozenset((PixelFormat.BGR888,)),
            encode_in_show=False,
        )

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h

    def _remember(self, frame) -> None:
        """记住帧尺寸供 caps()/width() 用; 形状异常就保持旧值, 绝不影响上屏。"""
        try:
            h, w = frame.shape[0], frame.shape[1]
        except Exception:
            return
        if w > 0 and h > 0:
            self._w, self._h = int(w), int(h)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass
