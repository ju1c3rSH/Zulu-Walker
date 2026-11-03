from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BatteryReading:
    """一次采样。HAL 只报数，不判断够不够用。

    frozen: 快照要跨线程传递，读者拿到的对象不能被改写。
    percent=-1 与官方 `pmu.get_bat_percent()` 的语义对齐: 无电池 / 不支持 / 读取失败。
    """

    percent: int = -1
    millivolts: int = -1


@runtime_checkable
class Battery(Protocol):
    def read(self) -> BatteryReading:
        """阻塞读一次；失败或无电池返回 `percent=-1`，不抛异常。

        ⚠ 调用契约: 只许"慢节拍"调用（主循环计数器 / 显示循环 / 独立线程，位置待定，
        见 docs/architecture/battery_hal_hud.md §2）。
        实测（2026-11-03, MaixCAM-Pro）: 电量 + 电压一次共 1.02ms（max 1.19ms），
        拆分是每次 I2C 事务约 0.26ms —— 其中约 0.18ms 是 i2c-dev 的固定开销，
        所以耗时由**事务次数**决定，不是总线时钟。总线卡住时不返回（i2c-4 与
        RTC/IMU 共用），所以视觉线程每帧跑 find_blobs 的地方、以及持锁区
        （presenter._lock）都不能调它。
        """
        ...


class NullBattery:
    """无电量计的默认实现（linux / mock / 没有 PMU 的板子）。"""

    def read(self) -> BatteryReading:
        return BatteryReading()
