# -*- coding: utf-8 -*-
"""启动计时打点 — 把"启动慢"从猜测变成可读数字。

## 为什么需要它

MaixCAM-Pro 从 `main()` 第一行到 `[UART CFG]` 曾隔了 8 秒, 中间**一个时间戳都没有**。
那 8 秒全部落在 `Machine.create()` 的五步串行构造里(相机/显示/串口/AI), 但每一步
都不打点, 所以"到底是相机慢还是抢 UART0 慢"只能猜。

⚠ **不要用 `logger` 打这类点**:`app/main.py` 的 `basicConfig` 钉了
`filename="logs/app.log"` 且 root level=WARNING, `logger.info/warning` **只进
文件、不进控制台**。台架看的是 MaixVision/串口的 stdout, 于是这类关键状态在
控制台完全不可见 —— 要么用 `log_print`(走独立写盘线程 + stdout 兜底),
要么用本模块。

## 用法

```python
from utils.boot_timing import boot_stage

with boot_stage("camera"):
    cam = factory(...)
```

或不需要 with 语法时:

```python
mark = BootTimer("startup")
mark.mark("camera")     # 逐点打
print(mark.summary())   # 收尾汇总
```

设计约束:
- **零依赖**: 只用 time + log_print, 不 import maix, PC 端也能 import 通过
  (测试要能在无设备环境跑)。
- **不抛异常**: 打点自身绝不能把启动搞挂, 计时失败就静默退化成无打点。
- 走 `log_print` 而非 print: 与既有台架日志同通道, 且不与 rich 的
  stdout 捕获打架。
"""
from __future__ import annotations

import time
from typing import Optional

try:
    from utils.log_util import log_print
except Exception:  # pragma: no cover - 极端情况下 log_util 不可用
    def log_print(msg: str = "") -> None:  # type: ignore
        pass

# 慢于该毫秒数就在行尾标一个标记, 便于一眼看出"谁吃掉了时间"。
# 阈值取得��保守(200ms): 目标是标出**异常段**, 而给每一步都打 "慢" 反而
# 淹没有效信息。
_SLOW_MS = 200.0


class BootTimer:
    """累计式计时器: 每个 mark 打一条"距上一 mark + 距起点"的日志。

    用法:
        t = BootTimer("boot")
        t.mark("camera")
        t.mark("display")
        t.mark("total", since_start=True)   # 只看总耗时
    """

    def __init__(self, name: str = "boot", enabled: bool = True) -> None:
        self._name = name
        self._enabled = enabled
        self._t0 = time.monotonic()
        self._prev = self._t0
        self._rows = []

    def mark(self, stage: str, since_start: bool = False) -> float:
        """记录一个阶段耗时, 返回毫秒数。

        *since_start*=True 时报"距起点"而非"距上一个 mark"—— 用于总计。
        打点失败(计时器禁用/时间倒流)返回 0.0, 不抛。
        """
        if not self._enabled:
            return 0.0
        try:
            now = time.monotonic()
            # 单调时钟理论上不回退, 但设备上NTP 校时/挂起唤醒偶发异常,
            # 算出负数会打印 "-3ms" 这种读起来像 bug 的东西。
            delta_ms = max(0.0, (now - self._prev) * 1000.0)
            start_ms = max(0.0, (now - self._t0) * 1000.0)
            self._prev = now
            self._rows.append((stage, delta_ms))
            shown = start_ms if since_start else delta_ms
            slow = "  <-- SLOW" if shown >= _SLOW_MS else ""
            log_print(f"[BOOT] {self._name} {stage}: {shown:.0f}ms{slow}")
            return shown
        except Exception:
            return 0.0

    def summary(self, stage: str = "TOTAL", since_start: bool = True) -> float:
        """收尾汇总。启动路径跑完调一次, 给出总账。"""
        return self.mark(stage, since_start=since_start)

    @property
    def elapsed_ms(self) -> float:
        try:
            return max(0.0, (time.monotonic() - self._t0) * 1000.0)
        except Exception:
            return 0.0

    @property
    def rows(self):
        return list(self._rows)


class _StageContext:
    """`with boot_stage("camera"): ...` 的上下文对象。

    用法上比 BootTimer.mark 更适合包住"一个构造调用":
        with boot_stage("camera"):
            cam = MaixCam1Camera(...)
    退出时自动打点, **异常路径也打**(标 <FAILED>), 这样"某步炸了导致后面
    全慢"和"某步单纯慢"在日志上能区分开。
    """

    __slots__ = ("_label", "_t0", "_timer")

    def __init__(self, label: str, timer: Optional[BootTimer] = None) -> None:
        self._label = label
        self._timer = timer
        self._t0 = time.monotonic()

    def __enter__(self) -> "_StageContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            ms = max(0.0, (time.monotonic() - self._t0) * 1000.0)
            if exc_type is not None:
                # 失败也要留痕: 失败步骤的耗时往往正是"启动卡住"的真凶,
                # 且失败后后续步骤的耗时会被污染, 不打就没法分清。
                log_print(f"[BOOT] {self._label}: {ms:.0f}ms  <-- FAILED "
                          f"({exc_type.__name__}: {exc})")
            elif self._timer is not None:
                self._timer.mark(self._label)
            else:
                slow = "  <-- SLOW" if ms >= _SLOW_MS else ""
                log_print(f"[BOOT] {self._label}: {ms:.0f}ms{slow}")
        except Exception:
            pass
        # **返回 False**: 打点失败绝不能吞掉业务异常。
        return False


def boot_stage(label: str, timer: Optional[BootTimer] = None) -> _StageContext:
    """包住一个启动步骤。异常路径同样打点, 且不吞异常。"""
    return _StageContext(label, timer)


def thread_cpu() -> float:
    """当前线程累计 CPU 时间（秒）。

    ⚠ 把"真在算"和"在等 GIL"分开的唯一手段:
    `time.perf_counter()` 量到的是**墙上时间** —— 线程被 GIL 挡住、被调度器
    挂起的时间都算在里面, 于是一个**纯等待**的线程也会显示"很忙"。
    报耗时的地方一律 **墙上 + CPU 两个都报**, 只报一个必然被误读。
    """
    try:
        return time.thread_time()
    except AttributeError:      # 老平台没有 thread_time
        return time.process_time()
