# -*- coding: utf-8 -*-
"""Turn one HAL battery read into a snapshot safe to share across threads.

No threads or timers of its own; the caller supplies the polling cadence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from framework.hal.interface.battery import Battery, BatteryReading
from utils.log_util import log_print

# On Pro boards a failed I2C read returns 0 (indistinguishable from a truly
# empty battery), so a large drop is accepted only after two consecutive reads.
_JUMP_POINTS = 20


@dataclass(frozen=True, slots=True)
class BatterySnapshot:
    """Snapshot read by the render and log paths; level is precomputed by the writer."""

    percent: int = -1
    millivolts: int = -1
    level: str = "unknown"      # charging | ok | low | unknown
    ts: float = 0.0             # time.monotonic(); HUD uses it to detect staleness
    reason: str = ""            # why percent < 0; forwarded from BatteryReading


class BatterySampler:
    def __init__(self, battery: Battery, *, low: int = 20) -> None:
        self._battery = battery
        self._low = int(low)
        self._snap = BatterySnapshot()
        self._suspect = -1          # last rejected suspect; needs two matching reads
        self._err_logged = False
        self.reads = 0
        self.last_ms = 0.0          # wall time of last read; caller detects a dead bus

    def snapshot(self) -> BatterySnapshot:
        """Lock-free: one writer and a single reference assignment cannot tear."""
        return self._snap

    def poll(self) -> Optional[BatterySnapshot]:
        """Read once, publish, and return the new snapshot or None if not published.

        None means a read error or a rejected jump. The return value tells the
        caller whether to log: it reports a fresh snapshot, not a changed value,
        so the caller need not remember the previous reading or re-log a stale
        one after a failure.
        """
        t0 = time.perf_counter()
        try:
            reading = self._battery.read()
        except Exception as e:    # platform reads should not raise; never break the main loop
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
        """A 20-point drop in one cycle means a shorted cell; the device would already be off."""
        return prev > 0 and new <= prev - _JUMP_POINTS

    def _level(self, percent: int, charging: bool | None) -> str:
        if charging:
            return "charging"
        return "low" if percent < self._low else "ok"
