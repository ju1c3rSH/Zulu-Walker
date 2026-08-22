"""Watchdog contract: keep-alive feed plus a way to stand down on exit."""

from __future__ import annotations

from typing import Protocol


class Watchdog(Protocol):
    def feed(self) -> None:
        """Reset the countdown. Must tolerate being called frequently."""
        ...

    def disable(self) -> None:
        """Stop the watchdog before a controlled exit. Idempotent."""
        ...
