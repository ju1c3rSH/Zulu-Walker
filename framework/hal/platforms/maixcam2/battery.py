# -*- coding: utf-8 -*-
"""Read MaixCAM2 (CW2015) battery state from kernel power_supply sysfs nodes.

The official PMU reads the same ``capacity`` value internally, but its
constructor reconfigures B29 as a GPIO and ``get_bat_vol()`` always returns 0,
so read sysfs directly. The gauge reports capacity on a 0..90 scale; scale it
to 0..100.
"""
from __future__ import annotations

from framework.hal.interface.battery import BatteryReading

_DIR = "/sys/class/power_supply/cw2015-battery"


def _read_text(name: str) -> str:
    with open(f"{_DIR}/{name}") as f:
        return f.read().strip()


class MaixCam2Battery:
    min_poll_s = 8.0        # Kernel driver polls the gauge only every 8s; faster reads are stale.

    def read(self) -> BatteryReading:
        """Return an empty reading when the node is absent (no battery or driver not running)."""
        try:
            capacity = int(_read_text("capacity"))      # Calibrated 0..90.
        except (OSError, ValueError):
            return BatteryReading(reason="sysfs 无 capacity（没装电池/驱动没起）")

        millivolts = -1
        try:
            millivolts = int(_read_text("voltage_now")) // 1000     # Kernel reports microvolts.
        except (OSError, ValueError):
            pass

        charging = None
        try:
            charging = _read_text("status") in ("Charging", "Full")
        except OSError:
            pass

        return BatteryReading(percent=min(100, capacity * 100 // 90),
                              millivolts=millivolts, charging=charging)
