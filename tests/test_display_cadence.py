# -*- coding: utf-8 -*-
"""Behavioral contract for the LCD presentation cadence (display_show_interval_ms).

Guarantees:

1. Default/0 = no throttle: every tick with a new frame goes to the screen.
2. >0 = time-based throttling, independent of frame rate (the reason this
   switch exists).
3. Unchanged content (same object) is still skipped; the throttle switch must
   not bypass that.
4. Invalid config values fall back to "no throttle"; they must not silently
   pin the screen at an absurd frame rate.
5. VisionManager's HUD rebuild window aligns with the presentation cadence,
   reverting to the built-in 33ms when it is 0.
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
    """Default (None): no throttle, so consecutive flushes all reach the screen."""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp)
    assert s.interval_ms == 0.0
    for i in range(5):
        s.push(_frame(i))
        s.flush()                     # consecutive calls at the same time still pass
    assert len(disp.shown) == 5


def test_explicit_zero_does_not_throttle(clock):
    """Explicit 0: equivalent to the default, no throttle."""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=0)
    for i in range(3):
        s.push(_frame(i))
        s.flush()
    assert len(disp.shown) == 3


def test_positive_interval_throttles_by_time(clock):
    """100ms: only one frame shows at a time; the next is released once time advances."""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=100)
    s.push(_frame(0))
    s.flush()
    for i in range(1, 5):
        s.push(_frame(i))
        s.flush()                     # clock unchanged -> all blocked
    assert len(disp.shown) == 1

    clock["t"] += 0.1                 # exactly one period
    s.push(_frame(9))
    s.flush()
    assert len(disp.shown) == 2


def test_interval_shorter_than_tick_still_gates(clock):
    """A 10ms interval with the clock advanced only 5ms still blocks: the
    throttle is not "once per tick".
    """
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=10)
    s.push(_frame(0))
    s.flush()
    clock["t"] += 0.005
    s.push(_frame(1))
    s.flush()
    assert len(disp.shown) == 1


def test_unchanged_frame_is_skipped_even_when_ungated(clock):
    """Identity skip takes priority over the throttle: a repeated object is not shown twice."""
    disp = _SpyDisplay()
    s = MaixLcdSink(disp, interval_ms=0)
    same = _frame(0)
    for _ in range(4):
        s.push(same)
        s.flush()
    assert len(disp.shown) == 1

    s.reset()                          # content source changed -> force replay
    s.push(same)
    s.flush()
    assert len(disp.shown) == 2


def test_set_interval_ms_switches_both_ways(clock):
    """At runtime one can switch from throttled back to unthrottled without resetting the clock."""
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
    """Invalid values (string/negative) must not pin the screen; fall back to no throttle."""
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
    """Legacy call sites (fps_limit only) keep their behavior: 30fps -> 33.3ms."""
    s = MaixLcdSink(_SpyDisplay(), fps_limit=30)
    assert s.interval_ms == pytest.approx(1000.0 / 30.0, rel=1e-3)


def test_explicit_interval_wins_over_fps_limit(clock):
    """When both are given, interval_ms wins (the newer config takes priority)."""
    s = MaixLcdSink(_SpyDisplay(), interval_ms=100, fps_limit=30)
    assert s.interval_ms == pytest.approx(100.0)


# ---------------- VisionManager side: HUD rebuild window alignment ----------------

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
    """A missing key (None) leaves the built-in ~30fps window unchanged."""
    vm = _bare_vm()
    vm.set_display_cadence_ms(None)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_cadence_zero_restores_builtin_window():
    """0 = unthrottled presentation, but the HUD still rebuilds at the
    built-in 33ms; rebuilding every frame is too costly.
    """
    vm = _bare_vm()
    vm.set_display_cadence_ms(100)
    vm.set_display_cadence_ms(0)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_cadence_matches_present_interval():
    """A 100ms presentation cadence gives a 100ms HUD window: one drawn frame, one shown frame."""
    vm = _bare_vm()
    vm.set_display_cadence_ms(100)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.1)


def test_cadence_is_floored_at_builtin_window():
    """16ms (60fps) must not make the vision thread fall back to a per-frame
    copy; the window is clamped to 33ms.
    """
    vm = _bare_vm()
    vm.set_display_cadence_ms(16)
    assert vm._DISPLAY_REBUILD_INTERVAL_S == pytest.approx(0.033)


def test_render_disabled_skips_composition():
    """display_enabled=false: stop composing display frames (no consumer, saves copy+redraw)."""
    vm = _bare_vm()
    vm.set_render_enabled(False)
    vm._update_display_frame()          # returns at once, without touching _pipelines
    assert vm._render_enabled is False

    vm.set_render_enabled(True)
    assert vm._render_enabled is True


def test_render_none_keeps_current_state():
    """A missing key (None) must not turn rendering off."""
    vm = _bare_vm()
    vm.set_render_enabled(None)
    assert vm._render_enabled is True
