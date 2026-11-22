# -*- coding: utf-8 -*-
"""电量采样器: 快照的用法 + 异常值处理。

这些签名要保证的行为（都是"错了但看不出来"那一类）:
  1. 没采过 = unknown，不是 0% —— 0% 会被读成"快没电了"；
  2. 一次读失败不能让快照变成 0，也不能抛给调用方；
  3. 单次跳到 0 不采纳（Pro 的 I2C 失败就返回 0），连续两次才采纳。

采集节拍（主循环慢节拍）由采集模块提供，不在本文件覆盖范围内: 那一层
依赖比赛分支的主循环架构，尚未并入主干，它的用例留在分支上。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.hal.battery_sampler import BatterySampler
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


# ---------------- 快照的用法 ----------------

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
    """低于 20 才转红（严格小于）; 充电优先。"""
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


# ---------------- 发布约定（"要不要打一行"的唯一依据） ----------------

def test_poll_returns_the_published_snapshot():
    s = BatterySampler(_ScriptedBattery(_reading(80)))
    snap = s.poll()
    assert snap is not None and snap.percent == 80
    assert snap is s.snapshot(), "返回的就是发布出去的那个对象"


def test_poll_returns_none_when_the_jump_is_rejected():
    """拒收 = 没发布 = 返回 None。调用方因此不会把旧值当新值再打一遍。"""
    s = BatterySampler(_ScriptedBattery(_reading(80), _reading(20)))
    assert s.poll() is not None
    assert s.poll() is None
    assert s.snapshot().percent == 80, "快照还是旧的"


def test_poll_returns_none_when_the_read_raises():
    s = BatterySampler(_ScriptedBattery(OSError("i2c nack")))
    assert s.poll() is None


# ---------------- 异常值 ----------------

def test_single_jump_to_zero_is_rejected_then_accepted():
    """Pro 的 I2C 失败返回 0，与真耗尽不可区分: 一次不信，两次信。"""
    s = BatterySampler(_ScriptedBattery(_reading(87), _reading(0), _reading(0)))
    s.poll()
    s.poll()
    assert s.snapshot().percent == 87, "单次 0 不能刷屏"
    s.poll()
    assert s.snapshot().percent == 0 and s.snapshot().level == "low"


def test_small_drop_is_accepted_immediately():
    """每次掉 <20 个点都立即采纳（只有大跳变才要两次确认）。"""
    s = BatterySampler(_ScriptedBattery(_reading(87), _reading(72), _reading(60)))
    for _ in range(3):
        s.poll()
    assert s.snapshot().percent == 60


def test_drop_of_exactly_twenty_points_needs_confirmation():
    """阈值边界: 72 -> 52 正好 20 个点，属于"不合理"，要第二次才采纳。"""
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
    """快照是渲染/日志的唯一来源: HAL 说"为什么", 这里不能丢。"""
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
