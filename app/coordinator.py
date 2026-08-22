"""Per-project orchestration slot — the "dirty code" area of app/.

Every project branch fills THIS file with its mission logic. The framework
never imports anything else from app/, so replacing this file is the whole
integration cost of starting a new competition project.

Contract exercised by app.main.main():
    AppCoordinator(event_bus)
    .set_wdt_feed(fn)     # framework feeds the dog through you if you need it
    .set_sys_info(info)   # late-injected AFTER Machine.create (rule 2)
    .loop()               # once per main tick; exceptions are rate-limited

A complete worked example lives on branch ``release/2026H``
(Ti2026Coordinator: UART slave bridge, alpha-beta ball filter, two-phase
rail calibration, record signaling).
"""

from __future__ import annotations

from typing import Optional

from framework.event_bus import EventBus


class AppCoordinator:
    """TODO(project): replace the no-op bodies with real orchestration."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._wdt_feed = lambda: None
        self._sys_info = None

    def set_wdt_feed(self, feed_fn) -> None:
        """Receive the boot watchdog closure (disable() available on it)."""
        self._wdt_feed = feed_fn

    def set_sys_info(self, sys_info) -> None:
        """Late-injected platform memory stats (None where unsupported)."""
        self._sys_info = sys_info

    def loop(self) -> None:
        """Once per main tick. Drain CmdQueues / Slots here; keep it fast."""
        pass
