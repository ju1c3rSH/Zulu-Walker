# -*- coding: utf-8 -*-
"""Tap 协议: `offer()` 不得阻塞（编码/写盘一律在 tap 自己的线程里）。"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tap(Protocol):
    name: str

    def attached(self) -> bool:
        ...

    def offer(self, frame: Any, seq: int) -> None:
        ...


class BaseTap:
    """带 attach/detach 的默认实现; `is_lcd` 决定它是否吃显示节流。"""

    name = "tap"
    is_lcd = False

    def __init__(self) -> None:
        self._attached = True

    def attach(self) -> None:
        self._attached = True

    def detach(self) -> None:
        self._attached = False

    def attached(self) -> bool:
        return self._attached

    def offer(self, frame: Any, seq: int) -> None:   # pragma: no cover - 抽象
        raise NotImplementedError
