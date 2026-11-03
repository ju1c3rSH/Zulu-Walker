# -*- coding: utf-8 -*-
"""MaixCAM2 (CW2015 电量计) 电量: 读内核 power_supply 的 sysfs 节点。

官方 `pmu` 的 maixcam2 分支内部就是读同一个 `capacity` 再 `×100/90`，百分比完全一致。
差异只有两处（调研 §2.4）:

- 它构造时会把 **B29** 复用成 GPIO（`is_charging()` 要读它）。当前工程没占用 B29，
  所以不是冲突，但属于无谓的引脚副作用: 哪天有人用 B29 接按键，症状会是
  "电量读得挺好，那个功能突然不工作"。
- `get_bat_vol()` 直接 `return 0`，拿不到电压；sysfs 的 `voltage_now` 是实值。

所以这里读 sysfs。`capacity` 原始值只标定到 90，放大口径与官方保持一致。
"""
from __future__ import annotations

from framework.hal.interface.battery import BatteryReading

_DIR = "/sys/class/power_supply/cw2015-battery"


def _read_text(name: str) -> str:
    with open(f"{_DIR}/{name}") as f:
        return f.read().strip()


class MaixCam2Battery:
    """节点不存在 = 没装电池或内核没起驱动，按无电池处理，不抛。"""

    min_poll_s = 8.0        # 内核驱动 8s 才轮询一次电量计，读更快纯属白读

    def read(self) -> BatteryReading:
        try:
            capacity = int(_read_text("capacity"))      # 0~90
        except (OSError, ValueError):
            return BatteryReading()

        millivolts = -1
        try:
            millivolts = int(_read_text("voltage_now")) // 1000     # 内核给 µV
        except (OSError, ValueError):
            pass

        return BatteryReading(percent=min(100, capacity * 100 // 90),
                              millivolts=millivolts)
