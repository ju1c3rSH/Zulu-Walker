# -*- coding: utf-8 -*-
"""JPEG tap: 只把帧塞进现有 JpegStreamer（fps 节流/编码/降采样全在它那边）。"""
from __future__ import annotations

from typing import Any

from framework.display.taps.base import BaseTap


class JpegTap(BaseTap):
    name = "jpeg"
    is_lcd = False

    def __init__(self, streamer: Any) -> None:
        super().__init__()
        self._streamer = streamer

    def attached(self) -> bool:
        """无客户端连接时不编码 —— 这是推流唯一的成本闸门。"""
        st = self._streamer
        if st is None or not self._attached:
            return False
        try:
            return bool(getattr(st, "is_running", False))
        except Exception:
            return False

    def offer(self, frame: Any, seq: int) -> None:
        try:
            self._streamer.push_frame(frame)     # 内部只做入队, 不阻塞
        except Exception:
            pass
