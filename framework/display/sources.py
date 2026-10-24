# -*- coding: utf-8 -*-
"""显示帧发布槽: 视觉线程 publish(), 显示线程 take_new()。

用 seq 代替 `frame is not _last_seen_frame` 身份比较; Frame 只在需要时创建。
"""
from __future__ import annotations

from typing import Any, Optional

from framework.display.types import Frame, PixelFormat


class DisplaySource:
    __slots__ = ("_handle", "_seq", "_taken")

    def __init__(self) -> None:
        self._handle: Any = None
        self._seq: int = -1        # ⚠ -1 = 还没发布过; 0 会被当成"已取过"
        self._taken: int = -1

    def publish(self, handle: Any) -> int:
        """每帧调用（视觉线程）; 零分配, 只递增 seq 并存引用。"""
        if handle is None:
            return self._seq
        self._handle = handle
        self._seq += 1
        return self._seq

    def seq(self) -> int:
        return self._seq

    def latest(self) -> Any:
        return self._handle

    def take_new(self) -> Any:
        """有新帧返回 handle 并记账, 否则 None（同一帧不会重复上屏）。"""
        if self._seq == self._taken or self._handle is None:
            return None
        self._taken = self._seq
        return self._handle

    def snapshot(self) -> Optional[Frame]:
        """按需创建 Frame 句柄（诊断/tap 用, 不在热路径上）。"""
        img = self._handle
        if img is None:
            return None
        try:
            w, h = int(img.width()), int(img.height())
        except Exception:
            shape = getattr(img, "shape", None)
            if shape is None:
                return None
            h, w = int(shape[0]), int(shape[1])
        return Frame(handle=img, fmt=PixelFormat.of(img), w=w, h=h,
                     seq=self._seq, origin="camera")
