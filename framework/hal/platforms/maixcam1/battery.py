# -*- coding: utf-8 -*-
"""MaixCAM-Pro (AXP2101) 电量: 用户态 I2C-4，从地址 0x34，走官方 pmu。

三个已知语义:
- I2C 读失败时 `get_bat_percent()` 返回 0 而不是 -1，与"真的没电"不可区分 ——
  去抖在 BatterySampler，这里只如实上报。
- `get_bat_vol()` 可能恒 0（ADC 未使能），0 按不可用上报，避免下游当"电芯已死"。
- 构造要跑一次芯片 ID 校验（~2.4ms），所以句柄常驻、全工程只在这里构造
  （C 层是进程内单例，但并发读没有锁）。

⚠ 这里**不打日志**：失败原因写进 `BatteryReading.reason`，由主循环那一个出口统一
打印。否则同一个 `percent=-1` 会在 HAL 和模块里各说一遍，而且 `app/main.py` 的
`basicConfig` 只挂 FileHandler —— logger 打的字在设备控制台本来就看不见。
"""
from __future__ import annotations

from framework.hal.interface.battery import BatteryReading


class MaixCam1Battery:
    min_poll_s = 0.0        # AXP2101 的电量计是连续的，没有内核轮询下限

    def __init__(self) -> None:
        self._reason = "pmu 不可用"
        try:
            from maix.ext_dev import pmu      # 老固件没有 ext_dev → 退化为无电池
            self._pmu = pmu.PMU()
            self._reason = ""
        except Exception as e:
            self._pmu = None
            self._reason = f"pmu 不可用: {type(e).__name__}"

    def read(self) -> BatteryReading:
        if self._pmu is None:
            return BatteryReading(reason=self._reason)

        try:
            percent = int(self._pmu.get_bat_percent())
            millivolts = int(self._pmu.get_bat_vol() or 0)
        except Exception as e:
            # 总线坏了会每 10s 抛一次。打印节流在上层（变化 + 心跳），这里只报原因，
            # 不自己判断"要不要说" —— 那是日志出口的事。
            return BatteryReading(reason=f"读失败: {type(e).__name__}")

        try:
            charging = bool(self._pmu.is_charging())
        except Exception:
            charging = None

        if percent < 0:
            return BatteryReading(reason="没装电池")
        return BatteryReading(percent=min(100, percent),
                              millivolts=millivolts if millivolts > 0 else -1,
                              charging=charging)
