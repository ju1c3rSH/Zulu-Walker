# -*- coding: utf-8 -*-
"""C3+C4: PresenterService 统一上屏出口、tap 分发、推流迁出视觉线程。

钉住:
  1. 限流语义与旧实现逐条等价（enabled / interval / 帧跳 / 相位重置 / CMM 降频）;
  2. 两个线程同时 submit 不丢不重、不交错进入 show;
  3. 异常隔离: tap 或 display.show() 抛异常都不外泄;
  4. JpegTap 的 attached 语义（无客户端就不编码）与复用现有 JpegStreamer;
  5. 推流独立于 LCD 节流（关屏也供帧）;
  6. DisplaySource 的 seq 语义 + 按需创建 Frame。
"""
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.display.policy import PresentPolicy
from framework.display.service import PresenterService
from framework.display.sources import DisplaySource
from framework.display.taps.base import BaseTap
from framework.display.taps.jpeg import JpegTap
from framework.display.taps.lcd import LcdTap
from framework.display.taps.null import NullTap
from framework.display.types import DisplayCaps, PixelFormat


# ---------------- fakes ----------------

class _SpyDisplay:
    def __init__(self, raise_on_show=False):
        self.shown = []
        self._raise = raise_on_show
        self.concurrent = 0
        self.max_concurrent = 0
        self._in_show = False

    def show(self, frame):
        if self._in_show:
            self.max_concurrent = max(self.max_concurrent, 2)
        self._in_show = True
        try:
            if self._raise:
                raise RuntimeError("boom")
            time.sleep(0.0001)
            self.shown.append(frame)
        finally:
            self._in_show = False
        return True

    def width(self):
        return 552

    def height(self):
        return 368

    def caps(self):
        return DisplayCaps(552, 368, PixelFormat.RGB888,
                           frozenset({PixelFormat.NATIVE, PixelFormat.RGB888}),
                           encode_in_show=True)


class _SpyTap(BaseTap):
    name = "spy"

    def __init__(self):
        super().__init__()
        self.got = []

    def offer(self, frame, seq):
        self.got.append((frame, seq))


class _BoomTap(BaseTap):
    name = "boom"

    def offer(self, frame, seq):
        raise RuntimeError("tap exploded")


def _presenter(**policy_kw):
    disp = _SpyDisplay()
    pol = PresentPolicy(**policy_kw)
    return PresenterService(disp, policy=pol), disp


# ---------------- 1. 限流语义 ----------------

def test_disabled_never_shows():
    p, disp = _presenter(enabled=False)
    for _ in range(6):
        assert p.submit("F") is False
    assert disp.shown == []


def test_interval_zero_is_frame_skip():
    """interval=0 -> 每 frame_skip 帧一次（与旧 _DISPLAY_SHOW_SKIP 同语义）。"""
    p, disp = _presenter(interval_ms=0, frame_skip=2)
    shown = [p.submit("F%d" % i) for i in range(6)]
    assert shown == [False, True, False, True, False, True]
    assert len(disp.shown) == 3


def test_interval_throttles_by_time_not_frame_count():
    """interval>0 -> 连续两次只上屏一次; 过了间隔再放行（与帧率无关）。"""
    p, disp = _presenter(interval_ms=100)
    assert p.submit("A") is True
    assert p.submit("B") is False
    time.sleep(0.11)
    assert p.submit("C") is True
    assert len(disp.shown) == 2


def test_reset_phase_lets_first_frame_through():
    """进页/回 MAIN 清相位 -> 首帧必上屏, 否则用户以为没切过去。"""
    p, disp = _presenter(interval_ms=10_000)
    p.submit("A")
    assert p.submit("B") is False
    p.reset_phase()
    assert p.submit("C") is True
    assert len(disp.shown) == 2


def test_set_policy_none_keeps_values():
    """缺配置键（None）不得改行为。"""
    p, _ = _presenter()
    p.set_policy(enabled=False, interval_ms=100)
    p.set_policy(None, None)
    assert p.policy.enabled is False
    assert p.policy.interval_ms == 100.0


def test_cmm_probe_checked_at_most_once_per_second():
    """CMM 查询降频到 1s 一次（原实现每 2 帧一次系统调用）。"""
    calls = []
    disp = _SpyDisplay()
    p = PresenterService(disp, policy=PresentPolicy(interval_ms=0, frame_skip=1),
                         cmm_probe=lambda: calls.append(1) or False)
    for _ in range(20):
        p.submit("F")
    assert len(calls) == 1


def test_cmm_pressure_blocks_show():
    disp = _SpyDisplay()
    p = PresenterService(disp, policy=PresentPolicy(frame_skip=1),
                         cmm_probe=lambda: True)
    assert p.submit("F") is False
    assert disp.shown == []


# ---------------- 2. 线程安全 ----------------

def test_submit_is_thread_safe_across_two_threads():
    """submit() 由显示线程与主循环线程同时调用 —— 单锁串行, 不丢不重。"""
    disp = _SpyDisplay()
    p = PresenterService(disp, policy=PresentPolicy(interval_ms=0, frame_skip=1))
    n = 200

    def worker(tag):
        for i in range(n):
            p.submit("%s-%d" % (tag, i))

    threads = [threading.Thread(target=worker, args=("a",)),
               threading.Thread(target=worker, args=("b",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert len(disp.shown) == 2 * n          # 一帧都不丢
    assert disp.max_concurrent == 0          # 没有两帧同时进入 show()
    assert p.stats()["submitted"] == 2 * n


# ---------------- 3. 异常隔离 ----------------

def test_show_exception_does_not_propagate():
    """display.show() 抛异常必须被吞 —— 冒泡会被 step_active 当成页面失败踢回 MAIN。"""
    disp = _SpyDisplay(raise_on_show=True)
    p = PresenterService(disp, policy=PresentPolicy(interval_ms=0, frame_skip=1))
    for _ in range(4):
        p.submit("F")            # 不应抛


def test_tap_exception_does_not_propagate_or_block_other_taps():
    p, disp = _presenter(interval_ms=0, frame_skip=1)
    good = _SpyTap()
    p.add_tap(_BoomTap())
    p.add_tap(good)
    assert p.submit("F") is True
    assert len(good.got) == 1     # 前一个 tap 炸了不影响后面的
    assert len(disp.shown) == 1


def test_null_frame_is_ignored():
    p, disp = _presenter(frame_skip=1)
    assert p.submit(None) is False
    assert disp.shown == []


# ---------------- 4. JpegTap attached 语义 ----------------

class _FakeStreamer:
    def __init__(self, running=True):
        self._running = running
        self.pushed = []

    @property
    def is_running(self):
        return self._running

    def push_frame(self, img):
        self.pushed.append(img)


def test_jpeg_tap_attached_follows_streamer_running():
    st = _FakeStreamer(running=True)
    tap = JpegTap(st)
    assert tap.attached() is True
    st._running = False
    assert tap.attached() is False, "无客户端连接时不得编码"


def test_jpeg_tap_detach_and_null_tap():
    st = _FakeStreamer()
    tap = JpegTap(st)
    tap.detach()
    assert tap.attached() is False
    assert NullTap().attached() is False


def test_jpeg_tap_reuses_streamer_push_frame():
    """推流编码路径不重写: tap 只把帧交给 JpegStreamer.push_frame。"""
    st = _FakeStreamer()
    tap = JpegTap(st)
    tap.offer("IMG", 7)
    assert st.pushed == ["IMG"]


def test_detached_tap_gets_nothing_attached_gets_every_frame():
    p, disp = _presenter(interval_ms=0, frame_skip=1)
    att, det = _SpyTap(), _SpyTap()
    det.detach()
    p.add_tap(att)
    p.add_tap(det)
    for i in range(5):
        p.submit("F%d" % i)
    assert len(att.got) == 5
    assert det.got == []
    assert len(disp.shown) == 5


# ---------------- 5. 推流独立于 LCD 节流 ----------------

def test_stream_keeps_frames_when_lcd_is_off():
    """关了屏仍要供帧（fps 节流在 streamer 内做）—— 否则推流跟着黑屏。"""
    p, disp = _presenter(enabled=False, frame_skip=1)
    jt = _SpyTap()
    p.add_tap(jt)
    for _ in range(4):
        p.submit("F")
    assert disp.shown == []
    assert len(jt.got) == 4


def test_stream_can_be_disabled_independently():
    """competition_mode 关推流, 但 LCD 照常（两个开关解耦）。"""
    p, disp = _presenter(interval_ms=0, frame_skip=1, stream_enabled=False)
    jt = _SpyTap()
    p.add_tap(jt)
    for _ in range(3):
        p.submit("F")
    assert len(disp.shown) == 3
    assert jt.got == []


def test_lcd_tap_is_the_only_show_path():
    """全系统只有一个对象调 display.show(): LCD tap。"""
    p, disp = _presenter(interval_ms=0, frame_skip=1)
    lcd_taps = [t for t in p._taps if isinstance(t, LcdTap)]
    assert len(lcd_taps) == 1
    assert lcd_taps[0].display is disp


# ---------------- 6. 兼容垫片: 调试页的 disp 用法 ----------------

def test_presenter_works_as_page_display():
    """page.step(disp, ...) 直接用 presenter: show/width/height/caps 都要有。"""
    p, disp = _presenter(interval_ms=0, frame_skip=1)
    assert p.show("IMG") is True      # == submit(source="page")
    assert (p.width(), p.height()) == (552, 368)
    assert p.caps().native_format is PixelFormat.RGB888
    assert disp.shown == ["IMG"]


def test_present_show_fn_exception_does_not_propagate():
    p, _ = _presenter(frame_skip=1)
    assert p.present(lambda: (_ for _ in ()).throw(RuntimeError("boom"))) is True


# ---------------- 7. DisplaySource ----------------

def test_source_publishes_every_frame_and_dedupes_take():
    src = DisplaySource()
    assert src.take_new() is None          # 还没发布过
    src.publish("F1")
    src.publish("F2")
    assert src.take_new() == "F2"          # 只拿最新
    assert src.take_new() is None          # 同一帧不重复上屏
    src.publish("F3")
    assert src.take_new() == "F3"


def test_source_never_publishes_none_and_seq_starts_unset():
    src = DisplaySource()
    assert src.seq() == -1
    assert src.publish(None) == -1
    assert src.take_new() is None


def test_source_snapshot_builds_frame_on_demand():
    """Frame 只在快照时创建（热路径零分配）。"""
    class _Img:
        def width(self):
            return 640

        def height(self):
            return 480

    src = DisplaySource()
    src.publish(_Img())
    fr = src.snapshot()
    assert (fr.w, fr.h) == (640, 480)
    assert fr.seq == 0
    assert fr.origin == "camera"
    assert fr.fmt is None                  # PC 无 maix, 判不出格式
