# -*- coding: utf-8 -*-
"""把 HAL 的一次读变成可跨线程读的快照。无线程、无定时器，节拍由调用方给。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from framework.hal.interface.battery import Battery, BatteryReading
from utils.log_util import log_print

# Pro 的 I2C 失败会返回 0（与真耗尽不可区分），所以大跳变要连续两次才采纳。
_JUMP_POINTS = 20


@dataclass(frozen=True, slots=True)
class BatterySnapshot:
    """渲染/日志路径读的对象。level 在写线程算好，HUD 只查颜色表。"""

    percent: int = -1
    millivolts: int = -1
    level: str = "unknown"      # charging | ok | low | unknown
    ts: float = 0.0             # time.monotonic()，HUD 据此判陈旧
    reason: str = ""            # percent<0 时"为什么"，透传自 BatteryReading


class BatterySampler:
    def __init__(self, battery: Battery, *, low: int = 20) -> None:
        self._battery = battery
        self._low = int(low)
        self._snap = BatterySnapshot()
        self._suspect = -1          # 上次被拒绝的可疑值（连续两次相同才采纳）
        self._err_logged = False
        self.reads = 0
        self.last_ms = 0.0          # 最近一次读的墙上耗时，调用方用它判"总线是不是坏了"

    def snapshot(self) -> BatterySnapshot:
        """无锁: 只有一个线程写 + 一次引用赋值，不会撕裂。"""
        return self._snap

    def poll(self) -> Optional[BatterySnapshot]:
        """读一次并发布。返回新发布的快照；没发布（读异常 / 跳变被拒）返回 None。

        返回值只有一个用途: 调用方据此决定要不要打日志。所以判断的是"有没有新快照"，
        而不是"值变没变" —— 打电话的人不用自己记上一次是多少，也不会在读失败时
        把旧值当新值重打一遍。
        """
        t0 = time.perf_counter()
        try:
            reading = self._battery.read()
        except Exception as e:                  # 平台实现不该抛；抛了也不能影响主循环
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            if not self._err_logged:
                self._err_logged = True
                log_print(f"[BAT] read failed: {type(e).__name__}: {e}（同错不再打印）")
            return None
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.reads += 1
        snap = self._accept(reading)
        if snap is None:
            return None
        self._snap = snap
        return snap

    def _accept(self, reading: BatteryReading) -> Optional[BatterySnapshot]:
        now = time.monotonic()
        if reading.percent < 0:
            self._suspect = -1
            return BatterySnapshot(percent=-1, millivolts=reading.millivolts,
                                   reason=reading.reason, ts=now)

        prev = self._snap.percent
        if self._implausible(prev, reading.percent) and reading.percent != self._suspect:
            self._suspect = reading.percent
            log_print(f"[BAT] 拒绝跳变 {prev}% -> {reading.percent}%（连续两次才采纳）")
            return None

        self._suspect = -1
        return BatterySnapshot(percent=reading.percent,
                               millivolts=reading.millivolts,
                               level=self._level(reading.percent, reading.charging),
                               ts=now)

    @staticmethod
    def _implausible(prev: int, new: int) -> bool:
        """一个周期掉 20 个点意味着电芯短路，那时设备早就断电了。"""
        return prev > 0 and new <= prev - _JUMP_POINTS

    def _level(self, percent: int, charging: bool | None) -> str:
        if charging:
            return "charging"
        return "low" if percent < self._low else "ok"
