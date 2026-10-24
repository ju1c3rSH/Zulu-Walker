# -*- coding: utf-8 -*-
"""PresenterService: 全系统**唯一**调 display.show() 的出口。

⚠ submit() 会被两个线程调用（MAIN 用显示线程, 调试页用主循环）——内部单锁串行化。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from framework.display.metrics import Metrics
from framework.display.policy import PresentPolicy
from framework.display.taps.lcd import LcdTap
from framework.display.types import DisplayCaps, PixelFormat
from utils.boot_timing import thread_cpu
from utils.log_util import log_print, perf_detail_enabled, perf_note

_CMM_PRESSURE_THRESHOLD = 0.80
_CMM_TOTAL_FALLBACK = 256 * 1024 * 1024


def _default_cmm_probe() -> bool:
    """CMM 占用是否超阈值; 无 maix 时返回 False（PC/测试）。"""
    try:
        from maix import sys as maix_sys
        info = maix_sys.memory_info()
        used = info.get("cmm_used", 0)
        total = info.get("cmm_total", _CMM_TOTAL_FALLBACK)
        return used > int(total * _CMM_PRESSURE_THRESHOLD)
    except Exception:
        return False


class PresenterService:
    """按策略限流并分发到 taps; `show()` 垫片让调试页把它当 display 用。"""

    def __init__(self, display: Any, taps: Optional[List[Any]] = None,
                 policy: Optional[PresentPolicy] = None,
                 cmm_probe: Optional[Callable[[], bool]] = None) -> None:
        self._lock = threading.Lock()
        self.policy = policy or PresentPolicy()
        self._cmm_probe = cmm_probe or _default_cmm_probe
        self._lcd = LcdTap(display)
        self._taps: List[Any] = [self._lcd] + list(taps or ())
        self.frame_counter = 0          # interval_ms == 0 时按帧跳
        self.last_show_time = 0.0
        self._phase_reset = False       # 进/出调试页后首帧必上屏
        self._cmm_check_time = 0.0
        self._cmm_pressure = False
        self.metrics: Dict[str, Metrics] = {"main": Metrics("lcd"),
                                            "page": Metrics("page")}
        self._last_fmt = PixelFormat.NATIVE
        self.last_show_ms = 0.0         # 最近一次上屏的墙上耗时（调试页归因用）
        self._skipped = 0
        self._submitted = 0

    # ---------------- 策略 ----------------

    def set_policy(self, enabled=None, interval_ms=None, hud_interval_ms=None,
                   stream_enabled=None) -> None:
        """None = 保持当前值（缺配置键绝不悄悄改行为）。"""
        with self._lock:
            if enabled is not None:
                self.policy.enabled = bool(enabled)
            if interval_ms is not None:
                self.policy.interval_ms = max(0.0, float(interval_ms))
            if hud_interval_ms is not None:
                self.policy.hud_interval_ms = max(0.0, float(hud_interval_ms))
            if stream_enabled is not None:
                self.policy.stream_enabled = bool(stream_enabled)
            on, ms, skip = (self.policy.enabled, self.policy.interval_ms,
                            self.policy.frame_skip)
        if ms > 0.0:
            throttle = "%.0fms 限流" % ms
        else:
            throttle = "每 %d 帧一次(帧跳)" % skip
        log_print("[DISPLAY] lcd=%s, %s" % ("ON" if on else "OFF", throttle))

    def reset_phase(self) -> None:
        """进/出调试页清节流相位 —— 新页首帧必须上屏。"""
        with self._lock:
            self.last_show_time = 0.0
            self.frame_counter = 0
            self._phase_reset = True

    def add_tap(self, tap: Any) -> None:
        with self._lock:
            self._taps.append(tap)

    def attach_tap(self, name: str, attached: bool = True) -> None:
        with self._lock:
            for tap in self._taps:
                if getattr(tap, "name", "") == name:
                    tap.attach() if attached else tap.detach()

    def tap_attached(self, name: str) -> bool:
        with self._lock:
            for tap in self._taps:
                if getattr(tap, "name", "") == name:
                    try:
                        return bool(tap.attached())
                    except Exception:
                        return False
        return False

    # ---------------- 唯一出口 ----------------

    def submit(self, frame: Any, *, source: str = "main",
               show_fn: Optional[Callable[[], Any]] = None) -> bool:
        """按策略限流后分发; 返回是否真的上屏。任何异常都不外泄。

        ⚠ 三个入口（MAIN / 调试页 / 显式 show_fn）共用这一条 gate, 不许各写一份。
        """
        if frame is None and show_fn is None:
            return False
        with self._lock:
            self._submitted += 1
            self.frame_counter += 1
            if self._gate(time.monotonic()):
                self._render(frame, source, show_fn)
                shown = True
            else:
                self._skipped += 1
                shown = False
            # ⚠ 推流独立于 LCD 节流: 关了屏也要继续供帧（fps 节流在 streamer 内）
            if self.policy.stream_enabled:
                self._offer_taps(frame)
            return shown

    def show(self, frame: Any, *args, **kwargs) -> bool:
        """调试页把 presenter 当 display 用: show(img) == submit(source="page")。"""
        return self.submit(frame, source="page")

    def _render(self, frame: Any, source: str,
                show_fn: Optional[Callable[[], Any]]) -> None:
        """锁内真正上屏: show_fn 优先（旧调用点自带 show 闭包）, 否则走 LCD tap。"""
        if show_fn is not None:
            t0 = time.perf_counter()
            try:
                show_fn()
            except Exception:
                pass        # ⚠ 上屏异常不得冒泡: 会打断调试页 step 被踢回 MAIN
            self._note_show((time.perf_counter() - t0) * 1000.0, frame, source)
            return
        if not self._lcd.attached():
            return
        t0 = time.perf_counter()
        cpu0 = thread_cpu()          # ⚠ 必须在 show 之前采样
        self._lcd.offer(frame, self.frame_counter)
        cpu = (thread_cpu() - cpu0) * 1000.0
        self._note_show((time.perf_counter() - t0) * 1000.0, frame, source, cpu)

    def present(self, show_fn: Callable[[], Any], frame: Any = None,
                *, source: str = "page") -> bool:
        """旧 `_maybe_show(show_fn)` 的等价入口: 自定义上屏闭包 + 同一条 gate。"""
        return self.submit(frame if frame is not None else show_fn,
                           source=source, show_fn=show_fn)

    def _gate(self, now: float) -> bool:
        """锁内调用: 策略判定 + CMM 降频; 通过时刷新时间戳。"""
        if not self.policy.enabled:
            return False
        if self._phase_reset:           # 进页/回 MAIN 首帧必须可见
            self._phase_reset = False
        else:
            ms = self.policy.interval_ms
            if ms > 0.0:
                if now - self.last_show_time < ms / 1000.0:
                    return False
            else:
                skip = self.policy.frame_skip
                if skip > 1 and self.frame_counter % skip != 0:
                    return False
        if self._cmm_pressure_throttled(now):
            return False
        self.last_show_time = now
        return True

    def _cmm_pressure_throttled(self, now: float) -> bool:
        """CMM 查询降频到 1s 一次（原每 2 帧一次系统调用）。"""
        if now - self._cmm_check_time >= 1.0:
            self._cmm_check_time = now
            try:
                self._cmm_pressure = bool(self._cmm_probe())
            except Exception:
                self._cmm_pressure = False
        return self._cmm_pressure

    def _offer_lcd(self, frame: Any, source: str) -> None:
        t0 = time.perf_counter()
        cpu0 = thread_cpu()          # ⚠ 必须在 show 之前采样
        self._lcd.offer(frame, self.frame_counter)
        cpu = (thread_cpu() - cpu0) * 1000.0
        self._note_show((time.perf_counter() - t0) * 1000.0, frame, source, cpu)

    def take_last_show_ms(self) -> float:
        """读走最近一次上屏耗时并清零（调试页逐帧归因用, 不会被重复累加）。"""
        with self._lock:
            ms = self.last_show_ms
            self.last_show_ms = 0.0
            return ms

    def _note_show(self, wall_ms: float, frame: Any, source: str,
                   cpu_ms: float = 0.0) -> None:
        """`[PERF] lcd|page show ... fmt=` 汇总; 未到窗口时只写槽位, 不格式化。"""
        self.last_show_ms = wall_ms
        self.last_show_cpu_ms = cpu_ms
        self._last_fmt = PixelFormat.of(frame) or PixelFormat.NATIVE
        if perf_detail_enabled():
            perf_note("fmt", wall_ms, "fmt=" + self._last_fmt.name)
        m = self.metrics.get(source) or self.metrics["main"]
        if m.add(wall_ms, cpu_ms):
            log_print(m.line(self._last_fmt.name))

    def _offer_taps(self, frame: Any) -> None:
        """非 LCD tap 逐个投递; attached()=False 的直接跳过（零编码成本）。"""
        seq = self.frame_counter
        for tap in self._taps:
            if tap is self._lcd:
                continue
            try:
                if tap.attached():
                    tap.offer(frame, seq)
            except Exception:
                pass

    # ---------------- 兼容垫片: 调试页的 disp 用法 ----------------

    def show(self, frame: Any, *args, **kwargs) -> bool:
        return self.submit(frame, source="page")

    def width(self) -> int:
        try:
            return int(self._lcd.display.width())
        except Exception:
            return 0

    def height(self) -> int:
        try:
            return int(self._lcd.display.height())
        except Exception:
            return 0

    def caps(self) -> DisplayCaps:
        try:
            return self._lcd.display.caps()
        except Exception:
            return DisplayCaps(width=self.width(), height=self.height(),
                               native_format=PixelFormat.NATIVE,
                               accepted=frozenset(), encode_in_show=False)

    def stats(self) -> dict:
        with self._lock:
            return {"submitted": self._submitted,
                    "shows": self._submitted - self._skipped,
                    "skipped": self._skipped, "enabled": self.policy.enabled,
                    "interval_ms": self.policy.interval_ms,
                    "fmt": self._last_fmt.name}
