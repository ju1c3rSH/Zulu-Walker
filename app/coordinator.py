"""Hold per-project orchestration logic in a single replaceable module.

The framework imports nothing else from ``app``, so replacing this file is the
entire integration cost of a new competition project.

Contract used by ``app.main.main``:
    AppCoordinator(event_bus)
    .set_wdt_feed(fn)    # watchdog feed hook, if needed
    .set_sys_info(info)  # late-injected after Machine.create
    .loop()              # once per main tick; exceptions are rate-limited

A worked example (Ti2026Coordinator) lives on branch ``release/2026H``.
"""

from __future__ import annotations

from typing import Optional

from framework.event_bus import EventBus


class AppCoordinator:
    """Project orchestration slot whose hooks are no-ops until filled in."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._wdt_feed = lambda: None
        self._sys_info = None

    def set_wdt_feed(self, feed_fn) -> None:
        """Store the watchdog feed closure, which also exposes ``disable()``."""
        self._wdt_feed = feed_fn

    def set_sys_info(self, sys_info) -> None:
        """Store late-injected platform memory stats (None where unsupported)."""
        self._sys_info = sys_info

    def loop(self) -> None:
        """Run one main-tick step; drain CmdQueues and Slots here and stay fast."""
        pass
