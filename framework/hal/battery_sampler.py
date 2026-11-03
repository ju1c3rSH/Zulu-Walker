# -*- coding: utf-8 -*-
"""电量采样器: 把 HAL 的一次读变成可安全跨线程取用的快照。
没有线程、没有定时器 —— 节拍由调用方给（主循环计数器、显示循环、独立线程都行），
这里只负责"读一次、判一次、发布一次"，因此测试不需要线程也不需要时钟。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from framework.hal.interface.battery import Battery, BatteryReading
from utils.log_util import log_print

# 什么算"不合理的下降": 一次掉 20 个点以上。
# 依据（调研 §3.5）: Pro 的 `get_bat_percent()` I2C 失败分支 `return false`，Python
# 收到 **0**，与"真的耗尽"不可区分。不拦的话，一次总线抖动就会让屏幕上出现 0%
# 并把颜色刷红，10s 后才恢复 —— 比不显示更糟。
_JUMP_POINTS = 20


@dataclass(frozen=True, slots=True)
class BatterySnapshot:
    """渲染路径读的对象。level 在写线程算好，HUD 只查颜色表。"""

    percent: int = -1
    millivolts: int = -1
    level: str = "unknown"      # unknown | ok | low | critical
    ts: float = 0.0             # time.monotonic()，HUD 据此判陈旧


class BatterySampler:
    def __init__(self, battery: Battery, *, low: int = 20, critical: int = 10) -> None:
        self._battery = battery
        self._low = int(low)
        self._critical = int(critical)
        self._snap = BatterySnapshot()
        self._suspect = -1          # 上次被拒绝的可疑值（连续两次相同才采纳）
        self._err_logged = False
        self.reads = 0
        self.last_ms = 0.0          # 最近一次读的墙上耗时，调用方用它判"总线是不是坏了"

    def snapshot(self) -> BatterySnapshot:
        """无锁: 单写者 + 一次引用赋值，读者要么拿到旧对象要么拿到新对象，不会撕裂。"""
        return self._snap

    def poll(self) -> None:
        t0 = time.perf_counter()
        try:
            reading = self._battery.read()
        except Exception as e:                  # 平台实现不该抛；抛了也不能影响主循环
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            if not self._err_logged:
                self._err_logged = True
                log_print(f"[BAT] read failed: {type(e).__name__}: {e}（同错不再打印）")
            return
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.reads += 1
        snap = self._accept(reading)
        if snap is not None:
            self._snap = snap

    def _accept(self, reading: BatteryReading) -> Optional[BatterySnapshot]:
        now = time.monotonic()
        if reading.percent < 0:
            self._suspect = -1
            return BatterySnapshot(percent=-1, millivolts=reading.millivolts, ts=now)

        prev = self._snap.percent
        if self._implausible(prev, reading.percent) and reading.percent != self._suspect:
            self._suspect = reading.percent
            log_print(f"[BAT] 拒绝跳变 {prev}% -> {reading.percent}%（连续两次才采纳）")
            return None

        self._suspect = -1
        return BatterySnapshot(percent=reading.percent,
                               millivolts=reading.millivolts,
                               level=self._level(reading.percent),
                               ts=now)

    @staticmethod
    def _implausible(prev: int, new: int) -> bool:
        """10s 掉 20 个点意味着电芯短路 —— 那时设备早就断电了。"""
        return prev > 0 and new <= prev - _JUMP_POINTS

    def _level(self, percent: int) -> str:
        if percent <= self._critical:
            return "critical"
        if percent <= self._low:
            return "low"
        return "ok"
