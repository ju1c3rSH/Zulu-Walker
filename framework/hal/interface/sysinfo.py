"""Memory statistics contract for pressure gating and diagnostics."""

from __future__ import annotations

from typing import Dict, Optional, Protocol


class SysInfo(Protocol):
    def memory_snapshot(self) -> Optional[Dict[str, int]]:
        """Platform memory stats in bytes; None when the concept is absent.

        Recognized keys: ``used`` / ``total`` (general heap) and
        ``cmm_used`` / ``cmm_total`` (media buffer pool on Axera SoCs).
        Consumers derive their own thresholds and formatting - the platform
        only reports raw numbers.
        """
        ...
