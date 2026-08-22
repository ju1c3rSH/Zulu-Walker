"""Desktop memory stats from /proc/meminfo (ARCH-07 capability)."""

from __future__ import annotations


class LinuxSysInfo:
    """General-heap snapshot only; no CMM concept on this platform."""

    def memory_snapshot(self):
        try:
            fields = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    fields[key.strip()] = int(rest.split()[0]) * 1024  # kB -> B
            total = fields.get("MemTotal")
            avail = fields.get("MemAvailable")
            if avail is None:
                avail = fields.get("MemFree")
            if not total or avail is None:
                return None
            return {"used": max(total - avail, 0), "total": total}
        except (OSError, ValueError):
            return None
