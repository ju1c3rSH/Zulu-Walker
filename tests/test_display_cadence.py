# -*- coding: utf-8 -*-
"""LCD 呈现节拍（display_show_interval_ms）行为契约。

保证:
  1. 缺省 / 0 = 不节流 —— 每拍有新帧就上屏;
  2. >0 = 按时间限流, 与帧率无关（这正是加这个开关的原因）;
  3. 内容未变（同一对象）仍然跳过, 节流开关不得绕过它;
  4. 非法配置值退化为"不节流", 不能静默把屏钉在一个离谱的帧率上;
  5. VisionManager 的 HUD 重建窗口与呈现节拍对齐, 且 0 时回到内置 33ms。
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from framework.hal.platforms.maixcam2 import sink as sink_mod
    from framework.hal.platforms.maixcam2.sink import MaixLcdSink

    _HAS_MAIX = True
except Exception:            # no MaixPy runtime (desktop / CI)
    sink_mod = None
    MaixLcdSink = None
    _HAS_MAIX = False

# The LCD sink lives in the maixcam2 platform, which imports the MaixPy
# runtime. Skip the module everywhere else instead of failing collection.
pytestmark = pytest.mark.skipif(
    not _HAS_MAIX, reason="requires the MaixPy runtime (maixcam2 platform)"
)


class _SpyDisplay:
    def __init__(self):
        self.shown = []

    def show(self, frame):
        self.shown.append(frame)
        return True


@pytest.fixture()
def clock(monkeypatch):
    """Controllable monotonic clock injected into the sink module only."""
    now = {"t": 1000.0}
    fake_time = types.SimpleNamespace(monotonic=lambda: now["t"])
    monkeypatch.setattr(sink_mod, "time", fake_time)
    return now


def _frame(i):
    """Distinct objects so the identity skip does not interfere."""
    return ("frame", i)


def test_absent_interval_does_not_throttle(clock):
    """缺省（None）: 不节流 —— 连续 flush 每拍都上屏。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp)
    assert s.interval_ms == 0.0
    for i in range(5):
        s.push(_frame(i))
        s.flush()                     # 同一时刻连续调用, 也不该被挡
    assert len(disp.shown) == 5


def test_explicit_zero_does_not_throttle(clock):
    """显式 0: 与缺省等价, 不节流。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=0)
    for i in range(3):
        s.push(_frame(i))
        s.flush()
    assert len(disp.shown) == 3


def test_positive_interval_throttles_by_time(clock):
    """100ms: 同一时刻只上屏一帧, 时间推进后才放行下一帧。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=100)
    s.push(_frame(0))
    s.flush()
    for i in range(1, 5):
        s.push(_frame(i))
        s.flush()                     # 时钟未动 -> 全被挡
    assert len(disp.shown) == 1

    clock["t"] += 0.1                 # 恰好一个周期
    s.push(_frame(9))
    s.flush()
    assert len(disp.shown) == 2


def test_interval_shorter_than_tick_still_gates(clock):
    """10ms 间隔 + 时钟只走 5ms: 仍被挡（限流不是"每拍一次"）。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=10)
    s.push(_frame(0))
    s.flush()
    clock["t"] += 0.005
    s.push(_frame(1))
    s.flush()
    assert len(disp.shown) == 1


def test_unchanged_frame_is_skipped_even_when_ungated(clock):
    """身份跳帧优先于节流开关: 同一对象重复 push 不重复上屏。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=0)
    same = _frame(0)
    for _ in range(4):
        s.push(same)
        s.flush()
    assert len(disp.shown) == 1

    s.reset()                          # 内容源切换 -> 强制重放
    s.push(same)
    s.flush()
    assert len(disp.shown) == 2


def test_set_interval_ms_switches_both_ways(clock):
    """运行时可从"限流"切回"不节流", 且时钟不被重置。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=1000)
    s.push(_frame(0))
    s.flush()
    s.push(_frame(1))
    s.flush()
    assert len(disp.shown) == 1

    s.set_interval_ms(0)
    assert s.interval_ms == 0.0
    s.push(_frame(2))
    s.flush()
    assert len(disp.shown) == 2


def test_invalid_interval_falls_back_to_ungated(clock):
    """非法值（字符串/负值）不得把屏钉住 —— 退化为不节流。"""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms="not-a-number")
    assert s.interval_ms == 0.0
    s.push(_frame(0))
    s.flush()
    s.push(_frame(1))
    s.flush()
    assert len(disp.shown) == 2

    s2 = MaixLcdSink(_SpyDisplay(), interval_ms=-50)
    assert s2.interval_ms == 0.0


def test_legacy_fps_limit_still_honoured(clock):
    """旧调用点（只传 fps_limit）行为不变: 30fps -> 33.3ms。"""
    s = MaixLcdSink(_SpyDisplay(), fps_limit=30)
    assert s.interval_ms == pytest.approx(1000.0 / 30.0, rel=1e-3)


def test_explicit_interval_wins_over_fps_limit(clock):
    """同时给出时以 interval_ms 为准（新配置优先）。"""
    s = MaixLcdSink(_SpyDisplay(), interval_ms=100, fps_limit=30)
    assert s.interval_ms == pytest.approx(100.0)


# ---------------- VisionManager 侧: HUD 重建窗口对齐 ----------------

def _bare_vm():
    """Construct a VisionManager without touching hardware.

    Only the display-cadence fields are read, so bypass __init__ entirely.
    """
    from modules.zw_opencv_module.vision_manager import VisionManager

    vm = object.__new__(VisionManager)
    vm._DISPLAY_REBUILD_INTERVAL_S = 0.033
    vm._DISPLAY_REBUILD_MIN_S = 0.033
    vm._render_enabled = True
    return vm


def test_cadence_none_keeps_builtin_window():
    """缺键（None）不改变内置 ~30fps 窗口。"""
    vm = _bare_vm()
    vm.set_display_cadence_ms(None)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_cadence_zero_restores_builtin_window():
    """0 = 不节流呈现, 但 HUD 仍按内置 33ms 重建（每帧重建太贵）。"""
    vm = _bare_vm()
    vm.set_display_cadence_ms(100)
    vm.set_display_cadence_ms(0)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_cadence_matches_present_interval():
    """100ms 呈现 -> HUD 也是 100ms, 做到"画一帧 show 一帧"。"""
    vm = _bare_vm()
    vm.set_display_cadence_ms(100)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.1)


def test_cadence_is_floored_at_builtin_window():
    """16ms（60fps）不得让视觉线程退回每帧 copy: 窗口夹到 33ms。"""
    vm = _bare_vm()
    vm.set_display_cadence_ms(16)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_render_disabled_skips_composition():
    """display_enabled=false: 不再合成显示帧（无消费者, 省 copy+redraw）。"""
    vm = _bare_vm()
    vm.set_render_enabled(False)
    vm._update_display_frame()          # 直接返回, 不访问 _pipelines
    assert vm._render_enabled is False

    vm.set_render_enabled(True)
    assert vm._render_enabled is True


def test_render_none_keeps_current_state():
    """缺键（None）不得关掉渲染。"""
    vm = _bare_vm()
    vm.set_render_enabled(None)
    assert vm._render_enabled is True
