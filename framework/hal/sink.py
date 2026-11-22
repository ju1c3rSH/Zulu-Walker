"""FrameSink implementations shared by every Maix platform.

Threadless by contract: the main loop pushes the latest composed frame and
pumps flush() once per tick. The sink owns presentation policy - fps limit,
unchanged-frame skip, CMM pressure gate - and nothing else.
"""

from __future__ import annotations

import time
from typing import Optional


class MaixLcdSink:
    """Presents composed frames via the platform display.

    ``pressure_check`` optionally gates presentation on system memory
    pressure (SysInfo snapshot on MaixCAM2); a True result skips one flush.
    """

    def __init__(self, display, interval_ms: Optional[float] = None,
                 pressure_check=None, pressure_interval_s: float = 1.0,
                 fps_limit: Optional[float] = None) -> None:
        """Configure a threadless LCD sink.

        Args:
            display: Platform display exposing ``show()``.
            interval_ms: Minimum milliseconds between presents. ``None`` or
                ``0`` disables time gating entirely — presentation is then
                bounded only by the main-loop cadence and the identity skip.
            pressure_check: Optional callable returning True to skip a present.
            pressure_interval_s: Min seconds between pressure_check calls.
            fps_limit: Legacy rate cap kept for older call sites; when given
                and ``interval_ms`` is None it is converted to milliseconds.

        Context:
            ``display_show_interval_ms`` in project_config maps straight to
            ``interval_ms``: 0/absent means "no throttle", which keeps the
            1-frame-per-tick behaviour available for boards whose LCD can
            take it, while slow panels can pin e.g. 100 (10 Hz).

            ``pressure_check`` is downsampled to once per interval: on MaixCAM
            the underlying query is a syscall, and calling it every flush
            showed up in the frame budget.
        """
        self._disp = display
        self._min_interval = self._resolve_interval(interval_ms, fps_limit)
        self._pressure_check = pressure_check
        self._pressure_interval = max(0.0, float(pressure_interval_s))
        self._last_pressure_check = 0.0
        self._pressure = False
        self._latest = None
        self._shown = None
        self._last_presented = 0.0
        self._force_next = False

    @staticmethod
    def _resolve_interval(interval_ms: Optional[float],
                          fps_limit: Optional[float]) -> float:
        """Milliseconds between presents -> seconds; 0 means "ungated".

        Args:
            interval_ms: Explicit interval in milliseconds (wins when set).
            fps_limit: Legacy frames-per-second cap, used only when
                ``interval_ms`` is None.

        Returns:
            Minimum seconds between presents; ``0.0`` disables time gating.

        Context:
            Malformed config must never throttle silently: bad values fall
            back to "ungated" instead of raising or pinning a bogus rate.
        """
        if interval_ms is not None:
            try:
                return max(0.0, float(interval_ms)) / 1000.0
            except (TypeError, ValueError):
                return 0.0
        if fps_limit:
            try:
                rate = float(fps_limit)
            except (TypeError, ValueError):
                return 0.0
            return (1.0 / rate) if rate > 0 else 0.0
        return 0.0

    def set_interval_ms(self, interval_ms: Optional[float],
                        fps_limit: Optional[float] = None) -> None:
        """Retune the present cadence at runtime (0/None = unthrottled)."""
        self._min_interval = self._resolve_interval(interval_ms, fps_limit)

    @property
    def interval_ms(self) -> float:
        """Current present interval in milliseconds; 0 = unthrottled."""
        return self._min_interval * 1000.0

    def push(self, frame) -> None:
        """Store the newest frame reference. Cheap; overwrites unsent."""
        if frame is not None:
            self._latest = frame

    def reset(self) -> None:
        """Force the next non-None frame to present, bypassing identity skip.

        Context:
            Call when the content source changes (page switch, camera
            reconnect) and the composer may reuse the same buffer object
            for genuinely new content.
        """
        self._force_next = True

    def _under_pressure(self, now: float) -> bool:
        """Return cached pressure state, refreshing at most once per interval."""
        if self._pressure_check is None:
            return False
        if now - self._last_pressure_check >= self._pressure_interval:
            self._last_pressure_check = now
            try:
                self._pressure = bool(self._pressure_check())
            except Exception:
                self._pressure = False
        return self._pressure

    def flush(self) -> None:
        """Present the stored frame if due and content changed."""
        frame = self._latest
        if frame is None:
            return
        if frame is self._shown and not self._force_next:
            # Identity skip: the composer reuses buffers between rebuilds,
            # so an unchanged object means no new content (DISP-05).
            return
        now = time.monotonic()
        if not self._force_next and now - self._last_presented < self._min_interval:
            return
        if self._under_pressure(now):
            return
        try:
            self._disp.show(frame)
            self._shown = frame
            self._last_presented = now
            self._force_next = False
        except Exception:
            pass

    def close(self) -> None:
        # Display lifetime is owned by Machine.close(); nothing to release.
        pass
