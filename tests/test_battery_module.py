# -*- coding: utf-8 -*-
"""Tests for the battery module's main-loop sampling cadence.

These tests cover when ``modules.zw_battery_module`` samples, when it logs a
line, and how it backs off on bus faults. Sampler behaviour itself is covered
in ``test_battery_sampler.py``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.hal.interface.battery import BatteryReading


class _ScriptedBattery:
    """Return scripted readings, repeating the last one once the script ends."""

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


class _StubMachine:
    def __init__(self, battery):
        self.battery = battery


def _init_module(poll_ms=10000):
    from modules import zw_battery_module as mod
    battery = _ScriptedBattery(_reading(76))
    mod.init(machine=_StubMachine(battery),
             config={"battery": {"enabled": True, "poll_interval_ms": poll_ms}})
    return mod, battery


def test_first_call_polls_immediately():
    """Sample on the first tick so boot does not wait a full interval for a reading."""
    mod, battery = _init_module(poll_ms=10000)
    mod._poll_if_due(0.0)
    assert battery.reads == 1


def test_polls_once_per_interval():
    mod, battery = _init_module(poll_ms=10000)
    mod._poll_if_due(0.0)
    mod._poll_if_due(9.9)
    assert battery.reads == 1
    mod._poll_if_due(10.0)
    assert battery.reads == 2
    mod._poll_if_due(19.9)
    assert battery.reads == 2
    mod._poll_if_due(20.0)
    assert battery.reads == 3


def test_kernel_poll_floor_wins_over_config():
    """The platform's ``min_poll_s = 8`` floor overrides a shorter configured interval."""
    from modules import zw_battery_module as mod

    class _SlowBattery(_ScriptedBattery):
        min_poll_s = 8.0

    battery = _SlowBattery(_reading(76))
    mod.init(machine=_StubMachine(battery),
             config={"battery": {"enabled": True, "poll_interval_ms": 1000}})
    mod._poll_if_due(0.0)
    mod._poll_if_due(7.9)
    assert battery.reads == 1
    mod._poll_if_due(8.0)
    assert battery.reads == 2


def test_slow_read_doubles_interval():
    """Treat a read over 50 ms as a bus fault and double the polling interval."""
    import time

    from modules import zw_battery_module as mod

    class _SlowBattery(_ScriptedBattery):
        def read(self):
            r = super().read()
            time.sleep(0.06)
            return r

    mod.init(machine=_StubMachine(_SlowBattery(_reading(76))),
             config={"battery": {"enabled": True, "poll_interval_ms": 1000}})
    mod._poll_if_due(0.0)
    assert mod._interval == 2.0, "读超时后周期应翻倍"
    mod._poll_if_due(1.9)
    assert mod._sampler.reads == 1
    mod._poll_if_due(2.0)
    assert mod._sampler.reads == 2


def test_module_is_idempotent_across_init():
    """Repeated ``init`` calls reset the next due time for tests and restarts."""
    mod, _ = _init_module(poll_ms=10000)
    mod._poll_if_due(500.0)
    mod, battery2 = _init_module(poll_ms=10000)
    mod._poll_if_due(0.0)
    assert battery2.reads == 1


def test_disabled_module_never_reads():
    from modules import zw_battery_module as mod
    battery = _ScriptedBattery(_reading(76))
    mod.init(machine=_StubMachine(battery),
             config={"battery": {"enabled": False}})
    assert mod.get_sampler() is None
    for _ in range(1000):
        mod.loop()
    assert battery.reads == 0


def test_every_sample_is_logged(monkeypatch):
    """Log one line per published snapshot, not only when the value changes.

    Logging only changes would leave gaps in ``debug.log``: twenty stable
    minutes would produce a single line, making it impossible to tell whether
    sampling ran. ``log_print`` also writes the same line to the console.
    """
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    mod, _ = _init_module(poll_ms=10000)       # Fixed reading of 76 percent.
    for t in (0.0, 10.0, 20.0, 30.0):
        mod._poll_if_due(t)
    assert sum("76%" in x for x in lines) == 4, lines


def test_absent_is_logged_every_sample_too(monkeypatch):
    """Log missing readings with the same policy, without a separate logging state.

    The cost is one line every 10 s on boards with no battery (six per minute).
    Set ``battery.enabled: false`` to silence it; adding rate limiting to the
    print path would reintroduce the state this design avoids.
    """
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    mod.init(machine=_StubMachine(_ScriptedBattery(BatteryReading(reason="没装电池"))),
             config={"battery": {"enabled": True, "poll_interval_ms": 10000}})
    for t in range(0, 601, 10):                # 61 ticks.
        mod._poll_if_due(float(t))

    assert len([x for x in lines if "无读数" in x]) == 61


def test_failed_read_does_not_reprint_the_stale_value(monkeypatch):
    """A failed read publishes no snapshot, so no line is logged: a stale value is not new."""
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    class _Boom(_ScriptedBattery):
        def read(self):
            self.reads += 1
            raise OSError("i2c nack")

    mod.init(machine=_StubMachine(_Boom(_reading(76))),
             config={"battery": {"enabled": True, "poll_interval_ms": 10000}})
    for t in (0.0, 10.0, 20.0):
        mod._poll_if_due(t)

    assert not [x for x in lines if "76%" in x], lines


def test_absent_reading_is_named_not_just_minus_one(monkeypatch):
    """A bare ``-1`` says nothing in the console; ``snap.reason`` names the actual cause."""
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    battery = _ScriptedBattery(BatteryReading(reason="pmu 不可用: ImportError"))
    mod.init(machine=_StubMachine(battery),
             config={"battery": {"enabled": True, "poll_interval_ms": 10000}})
    mod._poll_if_due(0.0)

    line = [x for x in lines if "无读数" in x]
    assert line and "pmu 不可用" in line[0], line


def test_backoff_recovers_after_a_good_read():
    """A single 50 ms spike must not pin the interval at 60 s; a good read halves it again."""
    import time

    from modules import zw_battery_module as mod

    class _Toggle(_ScriptedBattery):
        slow = True

        def read(self):
            r = super().read()
            if self.slow:
                time.sleep(0.06)
            return r

    battery = _Toggle(_reading(76))
    mod.init(machine=_StubMachine(battery),
             config={"battery": {"enabled": True, "poll_interval_ms": 1000}})
    mod._poll_if_due(0.0)
    assert mod._interval == 2.0, "慢读后周期翻倍"

    battery.slow = False
    mod._poll_if_due(2.0)
    assert mod._interval == 1.0, "恢复正常后要还回配置值"


def test_missing_hal_is_not_fatal():
    from modules import zw_battery_module as mod
    mod.init(machine=_StubMachine(None), config={"battery": {"enabled": True}})
    assert mod.get_sampler() is None
    mod.loop()                                     # Passing means no exception was raised.


def test_platform_without_gauge_stays_quiet(monkeypatch):
    """``NullBattery`` on desktop/mock platforms must not log "no reading" every 10 s."""
    from framework.hal.interface.battery import NullBattery
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    mod.init(machine=_StubMachine(NullBattery()),
             config={"battery": {"enabled": True, "poll_interval_ms": 10000}})
    assert mod.get_sampler() is None
    for t in (0.0, 10.0, 20.0):
        mod._poll_if_due(float(t))
    assert not [x for x in lines if "无读数" in x], lines
