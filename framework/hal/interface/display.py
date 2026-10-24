from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from framework.display.types import DisplayCaps


@runtime_checkable
class Display(Protocol):
    """上屏 HAL: show() 收 maix Image 或 BGR numpy; caps() 声明平台白名单。"""

    def show(self, frame: Any) -> bool:
        ...

    def caps(self) -> DisplayCaps:
        ...

    def width(self) -> int:
        ...

    def height(self) -> int:
        ...

    def close(self) -> None:
        ...
