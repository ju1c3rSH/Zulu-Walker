# -*- coding: utf-8 -*-
"""Battery sampler: snapshot semantics and outlier handling.

The contract these tests pin down (all "wrong but invisible" cases):

1. Never sampled = unknown, not 0%, since 0% reads as "almost empty".
2. One failed read must neither zero the snapshot nor raise to the caller.
3. A single jump to 0 is rejected (Pro returns 0 on I2C failure); accept only
   after two consecutive reads.

Polling cadence (the main loop's slow tick) is provided by the sampler module
and is out of scope here: that layer depends on the competition branch's
main-loop architecture, not yet merged into main, so its tests stay there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.hal.battery_sampler import BatterySampler
from framework.hal.interface.battery import BatteryReading


class _ScriptedBattery:
    """Emit readings from a script; repeat the last value once it is exhausted."""

    def __init__(self, *values):
        self._values = list(values) or [BatteryReading()]
        self.reads = 0

    def read(self) -> BatteryReading:
        self.reads += 1
        v = self._values.pop(0) if len(self._values) > 1 else self._values[0]
        if isinstance(v, Exception):
            raise v
        return v


def _reading(pct, mv=3900, charging=None):
    return BatteryReading(percent=pct, millivolts=mv, charging=charging)


# ---------------- snapshot semantics ----------------

def test_untouched_snapshot_is_unknown_not_zero():
    s = BatterySampler(_ScriptedBattery(_reading(80)))
    snap = s.snapshot()
    assert snap.percent == -1 and snap.level == "unknown" and snap.ts == 0.0


def test_first_poll_publishes_value_and_level():
    s = BatterySampler(_ScriptedBattery(_reading(80)))
    s.poll()
    snap = s.snapshot()
    assert (snap.percent, snap.millivolts, snap.level) == (80, 3900, "ok")
    assert snap.ts > 0


def test_level_thresholds():
    """Only a value below 20 turns red (strictly less than); charging wins."""
    for pct, level in ((100, "ok"), (21, "ok"), (20, "ok"), (19, "low"), (0, "low")):
        s = BatterySampler(_ScriptedBattery(_reading(pct)), low=20)
        s.poll()
        assert s.snapshot().level == level, pct

    s = BatterySampler(_ScriptedBattery(_reading(5, charging=True)), low=20)
    s.poll()
    assert s.snapshot().level == "charging", "充电时即使是 5% 也走绿"


def test_no_battery_reports_minus_one():
    s = BatterySampler(_ScriptedBattery(_reading(87), BatteryReading()))
    s.poll()
    assert s.snapshot().percent == 87
    s.poll()
    assert s.snapshot().percent == -1
    assert s.snapshot().level == "unknown"


# ---------------- publication contract (the only "print a line" criterion) ----------------

def test_poll_returns_the_published_snapshot():
    s = BatterySampler(_ScriptedBattery(_reading(80)))
    snap = s.poll()
    assert snap is not None and snap.percent == 80
    assert snap is s.snapshot(), "返回的就是发布出去的那个对象"


def test_poll_returns_none_when_the_jump_is_rejected():
    """Rejected = not published = returns None, so callers do not print the stale value again."""
    s = BatterySampler(_ScriptedBattery(_reading(80), _reading(20)))
    assert s.poll() is not None
    assert s.poll() is None
    assert s.snapshot().percent == 80, "快照还是旧的"


def test_poll_returns_none_when_the_read_raises():
    s = BatterySampler(_ScriptedBattery(OSError("i2c nack")))
    assert s.poll() is None


# ---------------- outlier values ----------------

def test_single_jump_to_zero_is_rejected_then_accepted():
    """Pro returns 0 on I2C failure, indistinguishable from a true empty cell:
    reject the first reading, accept the second.
    """
    s = BatterySampler(_ScriptedBattery(_reading(87), _reading(0), _reading(0)))
    s.poll()
    s.poll()
    assert s.snapshot().percent == 87, "单次 0 不能刷屏"
    s.poll()
    assert s.snapshot().percent == 0 and s.snapshot().level == "low"


def test_small_drop_is_accepted_immediately():
    """Drops of fewer than 20 points are accepted at once; only large jumps need confirmation."""
    s = BatterySampler(_ScriptedBattery(_reading(87), _reading(72), _reading(60)))
    for _ in range(3):
        s.poll()
    assert s.snapshot().percent == 60


def test_drop_of_exactly_twenty_points_needs_confirmation():
    """At the threshold, 72 -> 52 is exactly 20 points: treated as implausible,
    so it is accepted only on the second read.
    """
    s = BatterySampler(_ScriptedBattery(_reading(72), _reading(52), _reading(52)))
    s.poll()
    s.poll()
    assert s.snapshot().percent == 72
    s.poll()
    assert s.snapshot().percent == 52


def test_rise_is_always_accepted():
    s = BatterySampler(_ScriptedBattery(_reading(30), _reading(95)))
    s.poll()
    s.poll()
    assert s.snapshot().percent == 95


def test_read_exception_keeps_previous_snapshot():
    s = BatterySampler(_ScriptedBattery(_reading(80), RuntimeError("i2c boom")))
    s.poll()
    before = s.snapshot()
    s.poll()
    assert s.snapshot() is before, "读失败不能把画面清掉，也不能变成 0%"
    assert s.reads == 1


def test_reason_survives_the_snapshot():
    """The snapshot is the only source for rendering and logging; keep the HAL's reason."""
    s = BatterySampler(_ScriptedBattery(BatteryReading(reason="没装电池")))
    s.poll()
    assert s.snapshot().percent == -1 and s.snapshot().reason == "没装电池"


def test_reason_is_cleared_by_a_good_reading():
    s = BatterySampler(_ScriptedBattery(BatteryReading(reason="读失败: OSError"),
                                        _reading(80)))
    s.poll()
    assert s.snapshot().reason == "读失败: OSError"
    s.poll()
    assert s.snapshot().percent == 80 and s.snapshot().reason == ""
