# -*- coding: utf-8 -*-
"""电量采集模块: 主循环慢节拍。

被测对象是 "modules/zw_battery_module" —— 它决定何时采样、何时打一行、以及总线
异常时怎么退避。采样器本身的行为在 test_battery_sampler.py。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.hal.interface.battery import BatteryReading


class _ScriptedBattery:
    """按脚本吐读数；脚本用完就重复最后一项。"""

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
    """首拍即采: 否则开机后要等满一个周期才有第一个读数。"""
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
    """CAM2 的 min_poll_s=8 要压住更小的配置值。"""
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
    """单次读超过 50ms = 总线不正常, 周期翻倍退避（这是唯一的自动止损点）。"""
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
    """init 可以被调多次（测试/重启），下次命中时间必须归零。"""
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
    """快照发布一次就一行 —— 这是存档的来源，不是"值变了才打"。

    只打变化时，`debug.log` 里的电量是断的：稳定 20 分钟就只有一行，事后无法回答
    "那段时间到底采没采"。同一行也进控制台（`log_print` 一次调用两处落地）。
    """
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    mod, _ = _init_module(poll_ms=10000)       # 读数固定 76%
    for t in (0.0, 10.0, 20.0, 30.0):
        mod._poll_if_due(t)
    assert sum("76%" in x for x in lines) == 4, lines


def test_absent_is_logged_every_sample_too(monkeypatch):
    """无读数也一样: 没有第二套策略, 也就不需要"上一次是什么时候打的"这种状态。

    代价是没装电池的板子每 10s 一行（6 行/分）。要安静就 `battery.enabled: false`,
    不要在打印路径上加限流 —— 那会重新引入一份要维护的日志状态。
    """
    from modules import zw_battery_module as mod
    lines = []
    monkeypatch.setattr(mod, "log_print", lines.append)

    mod.init(machine=_StubMachine(_ScriptedBattery(BatteryReading(reason="没装电池"))),
             config={"battery": {"enabled": True, "poll_interval_ms": 10000}})
    for t in range(0, 601, 10):                # 61 拍
        mod._poll_if_due(float(t))

    assert len([x for x in lines if "无读数" in x]) == 61


def test_failed_read_does_not_reprint_the_stale_value(monkeypatch):
    """读失败时采样器不发布快照, 所以这里一行都不该出 —— 旧值不是新值。"""
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
    """`-1` 在控制台等于没说: 原因跟着快照走, 一行说清是没电池还是 pmu 打不开。"""
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
    """一次 50ms 抖动不该把周期永久打进 60s（原实现只翻倍、没有回落）。"""
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
    mod.loop()                                     # 不抛即通过


def test_platform_without_gauge_stays_quiet(monkeypatch):
    """桌面/mock 平台的 NullBattery 不该每 10s 打一行"无读数"。"""
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
