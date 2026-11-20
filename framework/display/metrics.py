# -*- coding: utf-8 -*-
"""show() 开销统计: 预分配环形缓冲, 热路径零分配。

输出行沿用旧格式 `[PERF] lcd show ...` / `[PERF] page show ...`（用户按此对日志）。
"""
from __future__ import annotations

import time


class Metrics:
    __slots__ = ("label", "window_s", "_cap", "_wall", "_cpu", "_idx", "_n",
                 "_wall_sum", "_cpu_sum", "_t0", "_elapsed", "_pend", "_shows",
                 "_last_n", "_last_wall", "_last_cpu")

    def __init__(self, label: str, window_s: float = 1.0,
                 capacity: int = 120) -> None:
        self.label = label
        self.window_s = window_s
        self._cap = max(1, int(capacity))
        self._wall = [0.0] * self._cap      # 预分配, 热路径只写槽位
        self._cpu = [0.0] * self._cap
        self._idx = 0
        self._n = 0
        self._wall_sum = 0.0
        self._cpu_sum = 0.0
        self._t0 = 0.0
        self._elapsed = 0.0
        self._pend = 0                      # 本窗口内 add 次数(超过 cap 也照数)
        self._shows = 0
        self._last_n = 0                    # 最近一次报告的窗口快照
        self._last_wall = 0.0
        self._last_cpu = 0.0

    def add(self, wall_ms: float, cpu_ms: float = 0.0) -> bool:
        """记一次 show; 返回 True 表示该打一条汇总了（每秒最多一次）。

        报告后必须清空本窗口的计数/累计, 否则窗口一过就每次调用都返回
          True（日志洪水）, 且 avg= 变成累计均值而不可与历史日志比较。
        """
        self._shows += 1
        self._pend += 1
        if self._n < self._cap:             # 环形缓冲只服务 p95, 跨窗口保留
            self._wall[self._n] = wall_ms
            self._cpu[self._n] = cpu_ms
            self._n += 1
        self._wall_sum += wall_ms
        self._cpu_sum += cpu_ms
        now = time.monotonic()
        if self._t0 == 0.0:
            self._t0 = now
            return False
        elapsed = now - self._t0
        if elapsed < self.window_s:
            return False
        self._elapsed = elapsed             # line() 读窗口值, 随后清零
        self._last_n = self._pend
        self._last_wall = self._wall_sum
        self._last_cpu = self._cpu_sum
        self._t0 = now
        self._pend = 0
        self._wall_sum = 0.0
        self._cpu_sum = 0.0
        return True

    def line(self, fmt_tag: str = "") -> str:
        """只在 add() 返回 True 后调用 —— 未打印时不做任何格式化。"""
        n = self._last_n or 1
        elapsed = self._elapsed if self._elapsed > 0.0 else 1.0
        fmt = " fmt=" + fmt_tag if fmt_tag else ""
        return ("[PERF] %s show %.1fHz avg=%.1fms (cpu %.1fms)%s "
                "-> 每秒占墙上 %.0fms / CPU %.0fms"
                % (self.label, n / elapsed, self._last_wall / n,
                   self._last_cpu / n, fmt,
                   self._last_wall / elapsed, self._last_cpu / elapsed))

    @property
    def shows(self) -> int:
        return self._shows

    @property
    def wall_sum(self) -> float:
        return self._wall_sum

    def p95_ms(self) -> float:
        """最近一窗的墙上 P95（诊断用, 未 flush 时也可取）。"""
        if self._n == 0:
            return 0.0
        s = sorted(self._wall[:self._n])
        return s[min(self._n - 1, int(round(0.95 * (self._n - 1))))]

    def reset(self) -> None:
        self._idx = 0
        self._n = 0
        self._wall_sum = 0.0
        self._cpu_sum = 0.0
        self._t0 = time.monotonic()
        self._elapsed = 0.0
        self._pend = 0
