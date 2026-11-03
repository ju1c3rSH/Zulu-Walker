# -*- coding: utf-8 -*-
"""MaixCAM-Pro (AXP2101) 电量: 用户态 I2C-4，从地址 0x34。

官方入口 `maix.ext_dev.pmu.PMU()` 在本板上走 axp2101 分支，`get_bat_percent()` 读
寄存器 0xA4，`-1` 表示没插电池。三条已知语义（调研 §3.5）:

- I2C 读失败的分支写的是 `return false` → Python 收到 **0**，与"电量耗尽"不可区分；
  去抖在 `BatterySampler` 里做，这里只如实上报。
- `AXP2101::init()` 没使能 ADC，`get_bat_vol()` 可能恒 0 → 报 -1 而不是 0
  （0mV 会被下游当"电芯已死"）。
- `pmu.PMU()` 构造要跑一次芯片 ID 校验（I2C 事务），实测 2.375ms（2026-11-03）。
  所以句柄常驻、只构造一次；C 层是带引用计数的进程内单例，但**并发读没有锁** ——
  全工程只许这里构造。

耗时（同一次实测, 20 次）: `get_bat_percent()` 中位 0.500ms、`get_bat_vol()` 0.523ms，
合计一次完整读 1.02ms。按事务拆: `get_bat_percent()` 内含两次事务（STATUS1 判在位 +
读 0xA4），单次事务约 0.26ms，其中约 0.18ms 是 i2c-dev 的固定开销。
"""
from __future__ import annotations

import logging

from framework.hal.interface.battery import BatteryReading

logger = logging.getLogger(__name__)


class MaixCam1Battery:
    min_poll_s = 0.0        # AXP2101 的电量计是连续的，没有内核轮询下限

    def __init__(self) -> None:
        self._pmu = None
        self._dead = False

    def read(self) -> BatteryReading:
        if self._dead:
            return BatteryReading()

        if self._pmu is None:
            try:
                from maix.ext_dev import pmu      # 老固件没有 ext_dev → 退化为无电池
                self._pmu = pmu.PMU()
            except Exception as e:
                # 一次失败即放弃: 板子上没有这颗 PMIC，重试的收益是零，代价是日志。
                self._dead = True
                logger.warning("[BAT] pmu 不可用，按无电池处理: %s", e)
                return BatteryReading()

        try:
            percent = int(self._pmu.get_bat_percent())
            millivolts = int(self._pmu.get_bat_vol() or 0)
        except Exception as e:
            logger.warning("[BAT] 读取失败: %s", e)
            return BatteryReading()

        if percent < 0:
            return BatteryReading()
        return BatteryReading(percent=min(100, percent),
                              millivolts=millivolts if millivolts > 0 else -1)
