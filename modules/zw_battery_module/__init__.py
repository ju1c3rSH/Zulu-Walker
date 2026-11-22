# -*- coding: utf-8 -*-
"""Sample battery state on the main loop tick, without a worker thread.

The main loop is used because the display thread only runs on the MAIN page;
sampling there would tie the data source to page state. Timing uses
``time.monotonic()`` because loop cadence varies with load (measured 89 Hz,
about 11.2 ms per tick), so a tick counter is not a time unit. A sample takes
1.02 ms (about 1.30 ms including ``is_charging``), which delays streaming by
at most 1 ms and drops no frames.

Each published snapshot is logged once: ``poll()`` returns a snapshot or
``None``, and this module only checks for ``None`` before calling ``log_print``.
No previous value is retained, no change detection or rate limiting is done,
and failed reads are never replayed as stale snapshots. ``log_print`` writes
to the console and ``logs/debug.log`` in one call, so both stay consistent.

A read over 50 ms is treated as a bus fault and doubles the interval, capped
at 60 s; one good read halves it again so a single spike cannot pin the
interval at the cap.
"""
from __future__ import annotations

import time
from typing import Optional

from framework.hal.battery_sampler import BatterySampler, BatterySnapshot
from framework.hal.interface.battery import NullBattery
from utils.log_util import log_print

_SLOW_READ_MS = 50.0            # A read above this is treated as a bus fault and doubles the interval.
_MAX_INTERVAL_S = 60.0          # Upper bound for the backoff interval in seconds.

_sampler: Optional[BatterySampler] = None
_base_interval = 10.0           # Configured interval; backoff never drops below it.
_interval = 0.0
_next_t = 0.0                   # Zero samples on the first tick instead of waiting one full interval.
_driver = "-"


def init(machine, **kwargs) -> None:
    global _sampler, _base_interval, _interval, _next_t, _driver
    _sampler, _interval, _next_t = None, 0.0, 0.0

    cfg = (kwargs.get("config") or {}).get("battery") or {}
    if not cfg.get("enabled", True):
        log_print("[BAT] 未启用（battery.enabled=false）")
        return
    battery = getattr(machine, "battery", None)
    if battery is None:
        log_print("[BAT] 平台未提供电量 HAL，不显示")
        return
    if isinstance(battery, NullBattery):
        # Platforms without a fuel gauge (desktop/mock) stay quiet instead
        # of logging "no reading" every 10 s.
        log_print("[BAT] 平台无电量计，不采集")
        return

    _base_interval = max(float(cfg.get("poll_interval_ms", 10000)) / 1000.0,
                         float(getattr(battery, "min_poll_s", 0.0) or 0.0))
    _interval = _base_interval
    _driver = type(battery).__name__
    _sampler = BatterySampler(battery, low=cfg.get("low_percent", 20))
    log_print(f"[BAT] poll={_interval:.0f}s 每拍一行（首拍即采）driver={_driver}")


def loop() -> None:
    if _sampler is None:
        return
    _poll_if_due(time.monotonic())


def _poll_if_due(now: float) -> None:
    """Poll at a caller-supplied time so tests can inject timestamps without sleeping."""
    global _next_t, _interval
    if _sampler is None or now < _next_t:
        return

    snap = _sampler.poll()
    if _sampler.last_ms > _SLOW_READ_MS:
        _interval = min(_interval * 2, _MAX_INTERVAL_S)
        log_print(f"[BAT] 单次读 {_sampler.last_ms:.0f}ms，周期退避到 {_interval:.0f}s")
    elif _interval > _base_interval:
        # One good read halves the interval; otherwise a single 50 ms spike keeps the 60 s cap forever.
        _interval = max(_base_interval, _interval / 2)

    if snap is not None:                       # A new snapshot produces one line; no snapshot produces none.
        log_print(_format(snap, _sampler.last_ms, _driver))

    # Schedule the next read after the read and backoff, so a new interval takes effect immediately.
    _next_t = now + _interval


def _format(snap: BatterySnapshot, read_ms: float, driver: str) -> str:
    """Format one snapshot as a single line; failures carry ``snap.reason`` from the HAL."""
    if snap.percent < 0:
        return (f"[BAT] 无读数（{snap.reason or '原因未知'}）"
                f" read={read_ms:.2f}ms driver={driver}")
    return (f"[BAT] {snap.percent}% {snap.millivolts}mV {snap.level} "
            f"read={read_ms:.2f}ms")


def get_sampler() -> Optional[BatterySampler]:
    """Return the sampler for HUD wiring, or None when disabled or unsupported."""
    return _sampler
