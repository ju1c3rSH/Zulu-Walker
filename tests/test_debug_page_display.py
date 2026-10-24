# -*- coding: utf-8 -*-
"""调试页上屏节流（DebugPageFactory）测试。

背景（2026-10-15 台架 + 用户反馈）:
  激光调试页 fps 很低, 且控制台反复刷:

      CVI_VENC_GetStream failed with 0xc0078012
      -- [E] compress image failed

  根因（应用层）: 调试页的 `disp.show()` 由页面自己调（DebugPageFactory.step_active
  -> page.step(...) -> disp.show）, 而 `MainUiShell._maybe_show` 的
  display_show_interval_ms 节流**只在 MAIN 页生效**（_display_loop 对非 MAIN 页
  直接 continue 让位）。于是调试页上 show() 退化为"每个相机帧一次"≈58Hz, 而
  MAIN 页是 10Hz —— 高约 6 倍, 且每次 show 都要走一遍 MaixCAM1 的 VPU/VENC
  JPEG 编码（show 同时要压缩一份发给 MaixVision）。

  官方 FAQ（MaixPy docs/doc/zh/faq.md）对这个错误码的解释正是: "上次程序没有正常
  退出, 导致 venc 模块资源没有释放, 再次启动应用就拿不到 venc 的资源了", 并给出
  "关掉 maixvision 的预览视图, 或者切换到 png 流" 作为缓解手段 —— 也就是
  VENC 通道被高频占用是必须从应用层收敛的。

本文件钉住的不变量:
  1. interval > 0 -> 调试页 show 按**时间**限流, 与帧率无关;
  2. interval == 0 -> 回退"每 _DISPLAY_SHOW_SKIP 帧一次"帧跳（= 改前行为）;
  3. display_enabled=False -> 一次都不上屏, 但页面 step（触摸/识别）照常执行;
  4. 缺键/None -> 保持当前值, 不悄悄改行为;
  5. 进页清节流相位 -> 新页首帧一定上屏;
  6. show 抛异常不冒泡（否则 step_active 的 except 会把页面踢回 MAIN）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import DebugPageFactory, _DISPLAY_SHOW_SKIP


class _SpyDisplay:
    """假 display: 记录 show 次数, 可选抛异常。"""

    def __init__(self, raise_on_show=False):
        self.shown = []
        self._raise = raise_on_show

    def show(self, frame):
        self.shown.append(frame)
        if self._raise:
            raise RuntimeError("boom")
        return True

    def width(self):
        return 552

    def height(self):
        return 368


class _SpyPage:
    """假调试页: 与真实页面一致 —— 用**传进来的** disp 上屏, 并记录 step 次数。

    ⚠ 不能持有自己的 display 引用: 真实页面 (threshold/rescue/laser tuner)
      的 step(disp, ts, frame) 一律用形参 disp, 代理才拦得住 show。
    """

    def __init__(self):
        self.steps = 0

    def step(self, disp, ts, frame_bgr):
        self.steps += 1
        disp.show("FRAME")

    def exit_requested(self):
        return False


def _factory(display=None, pages=None):
    return DebugPageFactory(machine=None, vision_manager=None, pages=pages)


def _call(factory, show_fn):
    """直接打 _maybe_show（节流判定的唯一出口）。"""
    return factory._maybe_show(show_fn)


# ---------------- 1/2. 时间限流 & 帧跳回退 ----------------

def test_interval_throttles_debug_page_show_by_time():
    """interval=10s: 连打 3 次只应上屏 1 次 —— 证明不再"每帧一次"。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=10_000)
    disp = _SpyDisplay()
    for _ in range(3):
        _call(f, lambda: disp.show("F"))
    assert len(disp.shown) == 1


def test_interval_zero_keeps_frame_skip_semantics():
    """interval=0 -> 回退"每 _DISPLAY_SHOW_SKIP 帧一次"（与 MAIN 页同一语义）。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=0)
    disp = _SpyDisplay()
    n = _DISPLAY_SHOW_SKIP * 3
    for _ in range(n):
        _call(f, lambda: disp.show("F"))
    assert len(disp.shown) == n // _DISPLAY_SHOW_SKIP


def test_show_calls_reset_throttle_clock():
    """上屏后重新计时: 同一次 interval 内不会因为后续调用再放行。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=10_000)
    disp = _SpyDisplay()
    assert _call(f, lambda: disp.show("F")) is True
    assert _call(f, lambda: disp.show("F")) is False
    assert len(disp.shown) == 1


# ---------------- 3. 代理只拦 show, 不拦 step ----------------

class _StubCam:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return "FRAME"


class _StubVM:
    def __init__(self, cam):
        self._cam = cam

    def get_pipeline_camera(self):
        return self._cam


class _StubPages:
    name = "LASER"

    def __init__(self):
        self.selected = []

    def select(self, name):
        self.selected.append(name)
        return True


def _stepping_factory(display, page):
    """装一个能真跑 step_active 的 factory（只替换外部依赖）。"""
    f = DebugPageFactory(machine=type("M", (), {"display": display})(),
                         vision_manager=_StubVM(_StubCam()),
                         pages=_StubPages())
    f._instances["LASER"] = page
    f._active = "LASER"
    # 触摸不参与本测试（PC 端拿不到 maix）: 直接给个假 ts
    f._ensure_touchscreen = lambda: object()
    return f


def test_disabled_never_shows_but_page_still_steps():
    """关显示时一次都不 show, 但 page.step 每拍照跑（触摸/识别不受影响）。

    这是本改动最容易踩的坑: 若把整个 step() 包进节流, 页面会既没画面
    又无法响应触摸退出 —— 变成死页。
    """
    disp = _SpyDisplay()
    page = _SpyPage()
    f = _stepping_factory(disp, page)
    f.set_display_policy(enabled=False, show_interval_ms=0)
    for _ in range(6):
        f.step_active("LASER")
    assert disp.shown == []          # 一次都没上屏
    assert page.steps == 6           # 但交互链路未被节流


def test_step_active_throttles_show_only():
    """step_active 走真实代理: 多拍只上屏一次, 但 step 仍每拍执行。"""
    disp = _SpyDisplay()
    page = _SpyPage()
    f = _stepping_factory(disp, page)
    f.set_display_policy(enabled=True, show_interval_ms=10_000)
    for _ in range(5):
        f.step_active("LASER")
    assert page.steps == 5
    assert len(disp.shown) == 1


def test_step_active_tolerates_none_frame():
    """cam.read() 返回 None 时 step_active 必须安静退让（不能空转抛异常）。"""
    disp = _SpyDisplay()
    page = _SpyPage()

    class _NoneVM:
        def get_pipeline_camera(self):
            return type("C", (), {"read": lambda self: None})()

    f = DebugPageFactory(machine=type("M", (), {"display": disp})(),
                         vision_manager=_NoneVM(), pages=_StubPages())
    f._instances["LASER"] = page
    f._active = "LASER"
    f._ensure_touchscreen = lambda: object()
    for _ in range(3):
        f.step_active("LASER")
    assert page.steps == 0
    assert disp.shown == []


# ---------------- 4. 缺键不改行为 ----------------

def test_missing_config_keys_keep_current_behaviour():
    """None -> 保持当前值（默认 enabled=True / interval=0 = 改前帧跳行为）。"""
    f = _factory()
    before = (f._display_enabled, f._show_interval_s)
    f.set_display_policy(None, None)
    assert (f._display_enabled, f._show_interval_s) == before


# ---------------- 5. 进页清相位 ----------------

def test_enter_page_resets_show_clock():
    """进页必须清节流相位, 否则新页首帧会被上一页残留的 _last_show_time 吞掉。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=10_000)
    disp = _SpyDisplay()
    _call(f, lambda: disp.show("F"))       # 先建立相位
    assert f._last_show_time > 0.0
    f._reset_show_clock()
    assert f._last_show_time == 0.0
    assert _call(f, lambda: disp.show("F")) is True   # 首帧必上屏
    assert len(disp.shown) == 2


def test_reset_clears_show_clock():
    """回 MAIN（factory.reset()）也要清相位, 下次进页首帧照常上屏。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=10_000)
    _call(f, lambda: None)
    assert f._last_show_time > 0.0
    f.reset()
    assert f._last_show_time == 0.0


# ---------------- 6. show 异常不冒泡 ----------------

def test_show_exception_does_not_raise():
    """show 抛异常必须被吞掉: 冒泡出去会被 step_active 当成 step 失败踢回 MAIN。"""
    f = _factory()
    f.set_display_policy(enabled=True, show_interval_ms=0)
    disp = _SpyDisplay(raise_on_show=True)
    for _ in range(_DISPLAY_SHOW_SKIP * 2):
        _call(f, lambda: disp.show("F"))     # 不应抛
    assert len(disp.shown) > 0
