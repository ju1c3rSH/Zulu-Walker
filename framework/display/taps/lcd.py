# -*- coding: utf-8 -*-
"""LCD tap: 唯一允许调 display.show() 的地方。"""
from __future__ import annotations

from typing import Any

from framework.display.taps.base import BaseTap


class LcdTap(BaseTap):
    name = "lcd"
    is_lcd = True

    def __init__(self, display: Any) -> None:
        super().__init__()
        self.display = display

    def attached(self) -> bool:
        return self._attached and self.display is not None

    def offer(self, frame: Any, seq: int) -> None:
        try:
            self.display.show(frame)
        except Exception:
            pass        # ⚠ 上屏异常不得冒泡: 会打断调试页 step 并被踢回 MAIN
