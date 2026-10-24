# -*- coding: utf-8 -*-
"""空 tap: 推流关闭时的占位, 让日志能说清"这里本该有流"。"""
from __future__ import annotations

from typing import Any

from framework.display.taps.base import BaseTap


class NullTap(BaseTap):
    name = "null"
    is_lcd = False

    def attached(self) -> bool:
        return False

    def offer(self, frame: Any, seq: int) -> None:
        return None
