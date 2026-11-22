from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BatteryReading:
    """一次采样。percent=-1 = 无电池 / 不支持 / 读取失败。

    frozen: 快照跨线程读，写者不能就地改字段。

    `reason`: percent=-1 时说明"为什么"。HAL 只报事实、不打日志 —— 日志只有一个
    出口（主循环慢节拍），否则同一个 -1 会在 HAL 和模块里各说一遍，而且控制台
    只看得到其中一句。
    """

    percent: int = -1
    millivolts: int = -1
    charging: bool | None = None     # None = 平台无法判定
    reason: str = ""                 # 仅 percent<0 时有意义


@runtime_checkable
class Battery(Protocol):
    def read(self) -> BatteryReading:
        """阻塞读一次，不抛异常。只许慢节拍调用（主循环 / 显示循环 / 独立线程）。"""
        ...


class NullBattery:
    """无电量计的默认实现。"""

    def read(self) -> BatteryReading:
        return BatteryReading(reason="平台无电量计")
