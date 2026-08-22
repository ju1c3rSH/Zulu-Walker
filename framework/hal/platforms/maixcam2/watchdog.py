"""MaixCAM2 hardware watchdog adapter (ARCH-02 capability implementation).

Deliberately dumb: construction starts the 10s countdown, ``feed`` raises on
hardware errors, and rate limiting / failure accounting belong to the caller
(see app.main._make_wdt_feed).
"""

from __future__ import annotations


class MaixWatchdog:
    """Adapter over ``maix.peripheral.wdt.WDT`` implementing Watchdog."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        from maix.peripheral import wdt as _mwdt

        self._wdt = _mwdt.WDT(0, int(timeout_s * 1000))

    def feed(self) -> None:
        self._wdt.feed()

    def disable(self) -> None:
        # Firmware revisions expose different shutdown verbs; probe them.
        for name in ("disable", "stop", "close"):
            fn = getattr(self._wdt, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue
