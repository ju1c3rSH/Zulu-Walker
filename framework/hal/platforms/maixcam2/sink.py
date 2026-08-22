"""FrameSink implementations for MaixCAM2 (D1/D4).

Threadless by contract: the main loop pushes the latest composed frame and
pumps flush() once per tick. The sink owns presentation policy - fps limit,
unchanged-frame skip, CMM pressure gate - and nothing else.
"""

from __future__ import annotations

import time


class MaixLcdSink:
    """Presents composed frames via the platform display.

    ``pressure_check`` optionally gates presentation on system memory
    pressure (SysInfo snapshot on MaixCAM2); a True result skips one flush.
    """

    def __init__(self, display, fps_limit: float = 30.0, pressure_check=None) -> None:
        self._disp = display
        self._min_interval = (1.0 / fps_limit) if fps_limit else 0.0
        self._pressure_check = pressure_check
        self._latest = None
        self._shown = None
        self._last_presented = 0.0

    def push(self, frame) -> None:
        """Store the newest frame reference. Cheap; overwrites unsent."""
        if frame is not None:
            self._latest = frame

    def flush(self) -> None:
        frame = self._latest
        if frame is None or frame is self._shown:
            # Identity skip: the composer reuses buffers between rebuilds,
            # so an unchanged object means no new content (DISP-05).
            return
        now = time.monotonic()
        if now - self._last_presented < self._min_interval:
            return
        if self._pressure_check is not None:
            try:
                if self._pressure_check():
                    return
            except Exception:
                pass
        try:
            self._disp.show(frame)
            self._shown = frame
            self._last_presented = now
        except Exception:
            pass

    def close(self) -> None:
        # Display lifetime is owned by Machine.close(); nothing to release.
        pass
