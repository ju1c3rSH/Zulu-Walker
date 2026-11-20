"""Readable timestamps for startup stages.

Uses ``log_print`` rather than ``logging`` so marks are visible on the bench
stdout. Zero dependencies, never raises: a timing failure degrades silently.

Usage::

    with boot_stage("camera"):
        cam = factory(...)
    # or:
    timer = BootTimer("boot")
    timer.mark("camera")
    timer.summary()
"""
from __future__ import annotations

import time
from typing import Optional

try:
    from utils.log_util import log_print
except Exception:  # pragma: no cover - log_util unavailable at import time
    def log_print(msg: str = "") -> None:  # type: ignore
        pass

from utils.thread_cpu import thread_cpu  # re-export for existing import sites

_STAGE_SLOW_MS = 200.0


class BootTimer:
    """Cumulative stage timer; each mark reports since-previous and since-start."""

    def __init__(self, name: str = "boot", enabled: bool = True) -> None:
        self._name = name
        self._enabled = enabled
        self._t0 = time.monotonic()
        self._prev = self._t0
        self._rows = []

    def mark(self, stage: str, since_start: bool = False) -> float:
        """Record one stage and return its elapsed milliseconds.

        Args:
            stage: Label for this stage.
            since_start: Report since-start instead of since-previous (totals).

        Return:
            Elapsed milliseconds, or 0.0 when disabled or on clock anomaly.

        Context:
            Non-monotonic readings clamped to zero; never raises.
        """
        if not self._enabled:
            return 0.0
        try:
            now = time.monotonic()
            delta_ms = max(0.0, (now - self._prev) * 1000.0)
            start_ms = max(0.0, (now - self._t0) * 1000.0)
            self._prev = now
            self._rows.append((stage, delta_ms))
            shown = start_ms if since_start else delta_ms
            slow = "  <-- SLOW" if shown >= _STAGE_SLOW_MS else ""
            log_print(f"[BOOT] {self._name} {stage}: {shown:.0f}ms{slow}")
            return shown
        except Exception:
            return 0.0

    def summary(self, stage: str = "TOTAL", since_start: bool = True) -> float:
        """Close out the boot path with a total; call once at the end."""
        return self.mark(stage, since_start=since_start)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this timer was created."""
        try:
            return max(0.0, (time.monotonic() - self._t0) * 1000.0)
        except Exception:
            return 0.0

    @property
    def rows(self):
        """Copy of recorded ``(stage, delta_ms)`` pairs."""
        return list(self._rows)


class _StageContext:
    """Context manager for ``with boot_stage(...): ...``."""

    __slots__ = ("_label", "_t0", "_timer")

    def __init__(self, label: str, timer: Optional[BootTimer] = None) -> None:
        self._label = label
        self._timer = timer
        self._t0 = time.monotonic()

    def __enter__(self) -> "_StageContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            ms = max(0.0, (time.monotonic() - self._t0) * 1000.0)
            if exc_type is not None:
                log_print(f"[BOOT] {self._label}: {ms:.0f}ms  <-- FAILED "
                          f"({exc_type.__name__}: {exc})")
            elif self._timer is not None:
                self._timer.mark(self._label)
            else:
                slow = "  <-- SLOW" if ms >= _STAGE_SLOW_MS else ""
                log_print(f"[BOOT] {self._label}: {ms:.0f}ms{slow}")
        except Exception:
            pass
        return False  # never swallow the business exception


def boot_stage(label: str, timer: Optional[BootTimer] = None) -> _StageContext:
    """Wrap a startup step; marks on both success and exception paths."""
    return _StageContext(label, timer)
