# -*- coding: utf-8 -*-
"""显示输出策略（display_enabled / display_show_interval_ms）与 [PERF] 分项计时测试。

背景（2026-10-08 台架实测）:
  - `display.show()` 单次 **23.4ms**，且 C++ 编码期间**全程持 GIL**；
    原实现每 2 帧 show 一次，20fps 下就是每秒 ~117ms 的 GIL 独占。
  - 同一次台架测量里，"取帧 + find_blobs"单独跑能到 **59.7Hz**，而带显示线程的
    真机 `[FPS] pipeline` 只有 **20.3Hz** —— 差额 ~35ms/帧不在识别里。
  - `project_config.yaml` 的 `display_enabled` 此前是**死配置**（全仓库无代码读它），
    于是"关了显示"只是心理安慰。

本文件钉住的不变量:
  1. `display_enabled=False` -> 一次都不 show（但 `_display_frame` 照常更新，
     触摸退出判定依赖它的宽高，见 test_competition_mode.py）；
  2. `display_show_interval_ms > 0` -> 按**时间**限流，与帧率无关；
  3. `== 0` -> 回退到原来的"每 _DISPLAY_SHOW_SKIP 帧一次"帧跳语义（等价改前行为）；
  4. 缺键/None -> 保持当前值，不悄悄改行为；
  5. CMM 内存压力查询降频到 1s 一次（原来每 2 帧一次系统调用）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import MainUiShell, _DISPLAY_SHOW_SKIP


class _SpyDisplay:
    def __init__(self):
        self.shown = []

    def show(self, frame):
        self.shown.append(frame)
        return True

    def width(self):
        return 640

    def height(self):
        return 480


class _StubMachine:
    def __init__(self):
        self.display = _SpyDisplay()


@pytest.fixture()
def ui():
    """最小 MainUiShell：PC 端 _open_touch 拿不到 maix，自动退化为 None。"""
    shell = MainUiShell(machine=_StubMachine(), pages=None,
                        factory=None, vision_manager=None)
    return shell


def _show(ui, counter):
    ui._display_counter = counter
    ui._maybe_show(frame="FRAME")


def test_disabled_never_shows(ui):
    """display_enabled=False: 一次都不上屏（这是"真关显示"的判据）。"""
    ui.set_display_policy(enabled=False)
    for c in range(1, 7):
        _show(ui, c)
    assert ui._machine.display.shown == []


def test_interval_zero_keeps_frame_skip_semantics(ui):
    """interval=0 -> 沿用帧跳: 每 _DISPLAY_SHOW_SKIP 帧一次（等价改前行为）。"""
    ui.set_display_policy(enabled=True, show_interval_ms=0)
    for c in range(1, _DISPLAY_SHOW_SKIP * 3 + 1):
        _show(ui, c)
    assert len(ui._machine.display.shown) == 3


def test_interval_throttles_by_time_not_frame_count(ui):
    """interval>0 -> 时间限流: 连续两次调用只上屏一次。

    帧跳是按**帧数**跳（帧率一变节流比例就变），时间限流与帧率无关 ——
    这正是加这个开关的目的，所以两种情况的行为必须区分开。
    """
    ui.set_display_policy(enabled=True, show_interval_ms=10000)
    _show(ui, 2)          # 第一次: 距上次(0.0)足够久 -> 放行
    _show(ui, 2)          # 立刻第二次: 被 10s 间隔挡住
    _show(ui, 2)
    assert len(ui._machine.display.shown) == 1


def test_show_calls_reset_throttle_clock(ui):
    """放行后必须刷新时钟，否则限流会退化成"第一次之后永久封禁"。"""
    ui.set_display_policy(enabled=True, show_interval_ms=10000)
    _show(ui, 2)
    first = ui._last_show_time
    ui._last_show_time = 0.0          # 模拟"已经过了很久"
    _show(ui, 2)
    assert len(ui._machine.display.shown) == 2
    assert ui._last_show_time > first or ui._last_show_time > 0.0


def test_missing_config_keys_keep_current_behaviour(ui):
    """缺键（None）不得改变行为 —— 老配置文件必须原样能用。"""
    before_enabled = ui._display_enabled
    before_interval = ui._show_interval_s
    ui.set_display_policy(None, None)
    assert ui._display_enabled is before_enabled
    assert ui._show_interval_s == before_interval


def test_cmm_pressure_checked_at_most_once_per_second(ui, monkeypatch):
    """CMM 查询降频: 1s 内多次调用只真正查一次（原来每 2 帧查一次）。"""
    calls = {"n": 0}
    monkeypatch.setattr("app.main._check_cmm_pressure",
                        lambda: calls.__setitem__("n", calls["n"] + 1) or False)
    ui.set_display_policy(enabled=True, show_interval_ms=0)
    for c in range(1, _DISPLAY_SHOW_SKIP * 10 + 1):
        _show(ui, c)
    assert calls["n"] == 1, "同一秒内不该重复查询 CMM 压力"


def test_show_perf_is_accumulated(ui):
    """show() 开销必须被统计 —— 否则台架无法量化"显示吃掉多少 GIL"。"""
    ui.set_display_policy(enabled=True, show_interval_ms=0)
    _show(ui, _DISPLAY_SHOW_SKIP)
    assert ui._show_n == 1
    assert ui._show_ms_sum >= 0.0


# ---------------- MainUiShell.tick 相关（含一个已修的静默 bug）----------------

class _StubPages:
    def __init__(self, name="MAIN"):
        self.name = name

    def is_main(self):
        return self.name == "MAIN"

    def select(self, _name):
        return True


def test_debug_page_factory_has_reset():
    """回归: DebugPageFactory 必须有 reset()。

    MainUiShell.tick() 在 MAIN 页每一拍都调 `self._factory.reset()`, 而这个方法
    曾经**不存在** —— 于是每拍抛 AttributeError, 被主循环的 except 吞掉:
      · 每拍格式化一次完整 traceback 并写盘（实测 tick=27.7ms, 是主循环最大耗时项）;
      · 紧跟其后的 _push_coordinator_status() 永远执行不到, DebugConsole 状态栏
        恒为 "-"。
    两条都不报错, 所以只有测试能钉住它。
    """
    from app.main import DebugPageFactory

    factory = DebugPageFactory(machine=None, vision_manager=None, pages=None)
    assert callable(getattr(factory, "reset", None)), \
        "DebugPageFactory.reset 缺失会让 MAIN 页每拍抛异常并被静默吞掉"
    factory._active = "TARGET"
    factory.reset()
    assert factory._active is None
    assert factory.page_active() is False


def test_main_tick_does_not_raise_and_pushes_status(ui, monkeypatch):
    """MAIN 页 tick 必须走通, 且真的把状态推给 DebugConsole（限频内）。"""
    pushed = []
    monkeypatch.setattr("app.main._push_coordinator_status",
                        lambda coord: pushed.append(coord))

    from app.main import DebugPageFactory
    ui._factory = DebugPageFactory(machine=None, vision_manager=None, pages=None)
    ui._pages = _StubPages("MAIN")
    ui._last_status_push = 0.0

    ui.tick(coordinator="COORD")          # 不该抛（抛了会被主循环静默吞掉）
    assert pushed == ["COORD"], "MAIN 页必须把状态推给调试台"

    for _ in range(5):                    # 限频期内: 不再重复推
        ui.tick(coordinator="COORD")
    assert pushed == ["COORD"], "状态推送应被限频, 不该每拍都推"


def test_status_push_throttle_interval(ui, monkeypatch):
    """限频后可恢复推送 —— 不能退化成"首拍之后再无状态"。"""
    pushed = []
    monkeypatch.setattr("app.main._push_coordinator_status",
                        lambda coord: pushed.append(coord))
    from app.main import DebugPageFactory
    ui._factory = DebugPageFactory(machine=None, vision_manager=None, pages=None)
    ui._pages = _StubPages("MAIN")

    ui._last_status_push = 0.0
    ui.tick(coordinator="A")
    ui._last_status_push = 0.0          # 模拟已经过了限频窗口
    ui.tick(coordinator="B")
    assert pushed == ["A", "B"]
