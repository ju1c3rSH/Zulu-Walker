# -*- coding: utf-8 -*-
"""Battery readings for the MaixCAM-Pro AXP2101 via userspace I2C-4 (0x34).

Use the official ``pmu`` interface, with three known behaviors:

- ``get_bat_percent()`` returns 0, not -1, on an I2C read failure, which is
  indistinguishable from a truly empty battery. Debouncing belongs in
  BatterySampler; this module only reports what the chip returns.
- ``get_bat_vol()`` may be permanently 0 when the ADC is disabled. Report a
  zero voltage as unavailable so downstream code does not read it as a dead
  cell.
- Construction performs a chip ID check (~2.4 ms), so keep the handle for the
  process lifetime and build it only here. The C layer is a process-wide
  singleton but does not lock concurrent reads.

Do not log here: store failure reasons in ``BatteryReading.reason`` and print
them at the single main-loop exit.
"""
from __future__ import annotations

from framework.hal.interface.battery import BatteryReading


class MaixCam1Battery:
    min_poll_s = 0.0        # The AXP2101 gauge is continuous; no kernel poll floor.

    def __init__(self) -> None:
        self._reason = "pmu 不可用"
        try:
            from maix.ext_dev import pmu      # Older firmware lacks ext_dev -> no battery.
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
            # A broken bus throws about every 10 s. Throttling is handled
            # upstream (on change plus heartbeat); report only the reason here
            # and leave the decision to log to the single logging exit.
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
