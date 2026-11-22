from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BatteryReading:
    """A single sample; ``percent=-1`` means absent, unsupported, or unreadable.

    Frozen so a snapshot can be shared across threads without in-place mutation.
    ``reason`` explains a ``-1`` value. The HAL reports facts without logging:
    the slow main-loop tick is the single log outlet, so one failure is not
    reported twice and then lost in a crowded console.
    """

    percent: int = -1
    millivolts: int = -1
    charging: bool | None = None     # None = platform cannot determine charging state
    reason: str = ""                 # meaningful only when percent < 0


@runtime_checkable
class Battery(Protocol):
    def read(self) -> BatteryReading:
        """Blocking read that never raises; call only on the slow tick."""
        ...


class NullBattery:
    """Default battery for platforms without a fuel gauge."""

    def read(self) -> BatteryReading:
        return BatteryReading(reason="平台无电量计")
