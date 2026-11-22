# -*- coding: utf-8 -*-
"""MaixCAM2 (CW2015) 电量: 读内核 power_supply 的 sysfs 节点。

官方 pmu 内部读的是同一个 `capacity`（百分比一致），但它构造时会把 B29 复用成
GPIO、且 `get_bat_vol()` 恒 0，所以这里直读 sysfs。capacity 只标定到 90，按官方
数值放大到 100。
"""
from __future__ import annotations

from framework.hal.interface.battery import BatteryReading

_DIR = "/sys/class/power_supply/cw2015-battery"


def _read_text(name: str) -> str:
    with open(f"{_DIR}/{name}") as f:
        return f.read().strip()


class MaixCam2Battery:
    min_poll_s = 8.0        # 内核驱动 8s 才轮询一次电量计，读更快纯属白读

    def read(self) -> BatteryReading:
        """节点不存在 = 没装电池 / 驱动没起，按无电池处理。"""
        try:
            capacity = int(_read_text("capacity"))      # 0~90
        except (OSError, ValueError):
            return BatteryReading(reason="sysfs 无 capacity（没装电池/驱动没起）")

        millivolts = -1
        try:
            millivolts = int(_read_text("voltage_now")) // 1000     # 内核给 µV
        except (OSError, ValueError):
            pass

        charging = None
        try:
            charging = _read_text("status") in ("Charging", "Full")
        except OSError:
            pass

        return BatteryReading(percent=min(100, capacity * 100 // 90),
                              millivolts=millivolts, charging=charging)
