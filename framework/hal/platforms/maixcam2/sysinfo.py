"""MaixCAM2 memory stats adapter (ARCH-07 capability implementation)."""

from __future__ import annotations


class MaixSysInfo:
    """Reports heap + CMM media-pool pressure from ``maix.sys``."""

    def memory_snapshot(self):
        from maix import sys as _sys

        return _sys.memory_info()
