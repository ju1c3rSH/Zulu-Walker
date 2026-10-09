# -*- coding: utf-8 -*-
"""LazyFrame — 延迟把 maix Image 转成 BGR numpy 的透明代理。

## 为什么需要

设备端相机每帧都做 `maix.image.image2cv(raw, copy=True)` —— 640x480x3 一次拷
921KB。台架真机实测这份拷贝在 `[PERF] vision ... conv=` 里占 **9~16ms**, 是
除 find_blobs 之外最大的一块。

但**它大多数时候是白做的**: TARGET/EXPLOSIVE 走 C++ `find_blobs` 直接吃
`camera.last_raw`(零拷贝), RESCUE 走 `_raw=True` 也直接吃 maix Image。这份 BGR
numpy 只服务两处 —— cv2 回退路径(maix 不可用) 与调参页。真机上 maix 可用时,
这份拷贝属于"算了不用"。

## 做法

相机不再急切转换, 改为交出一个 `LazyFrame` 代理:

  · `.shape` / `.ndim` / `.size` / `.dtype`  **从 maix Image 元数据直接给**
    (`width()/height()`), 零拷贝 —— 处理器里的 `h, w = frame.shape[:2]` 白拿。
  · 任何**真正的像素访问**(切片 / `__array__` / `.copy()` / `.numpy()`)
    才触发一次 `image2cv`, 结果缓存, 同一帧只转一次。

于是: 设备 + maix 可用 + 走 C++ 路径 => **一次拷贝都不发生**;
      cv2 回退 / 调参页 / RESCUE 传 numpy => 与原行为逐位一致。

## 契约

- `LazyFrame` 必须能**冒充 `np.ndarray`** 通过 `frame is None` 之外的判断。
  处理器里唯一对 frame 的"非像素"用法就是 `.shape`, 已覆盖。
- 鸭子类型: 只要不 `isinstance(frame, np.ndarray)`。全局搜过: 处理器/协处理器
  都没有对 frame 做 isinstance 判断（`frame is None` / `.shape` / 切片 / 传库）。
- `np.asarray(lazy)` 等价于 `lazy.numpy()` —— 供第三方库吞 numpy 的场合。
- ⚠⚠ **OpenCV 的 Python 绑定不收本代理**（2026-10-09 真机+PC 实测,
  OpenCV 4.9/5.0 同样报 "src is not a numpy array, neither a scalar"）:
  `cv2.resize/cvtColor` 等函数的入参转换**不走** `__array__` /
  `__array_interface__`, 只认 `PyArray_Check` 为真的真 ndarray。
  所以"把 LazyFrame 直接喂 cv2"是错的 —— 需要 cv2 的消费者（调参页）
  必须先 `.numpy()`/`.copy()` 拿真数组; 相机侧 `read()`（调参页入口）
  已按返回契约收敛为真 numpy。

PC 端(无 maix)不构造本类: `LazyFrame.for_raw()` 在 raw 为 None 或转换函数不可用时
直接返回现有 numpy / None, 不引入任何 maix 依赖。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class LazyFrame:
    """把"原始 maix Image + 转换函数"包成看起来像 BGR numpy 的只读代理。"""

    __slots__ = ("_raw", "_convert", "_numpy", "_shape", "_converted")

    def __init__(self, raw, convert: Callable[[object], np.ndarray],
                 shape_hw: Optional[tuple] = None) -> None:
        self._raw = raw
        self._convert = convert        # convert(raw) -> BGR numpy
        self._numpy: Optional[np.ndarray] = None
        self._shape = shape_hw         # (h, w) —— 来自 raw 元数据, 零拷贝
        self._converted = False

    # ---- 元数据: 零拷贝, 不触发转换 ----

    @property
    def shape(self):
        h, w = self._shape if self._shape else (0, 0)
        return (h, w, 3)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    @property
    def size(self) -> int:
        h, w = self._shape if self._shape else (0, 0)
        return h * w * 3

    # ---- 真正的像素访问: 触发一次转换并缓存 ----

    def copy(self) -> np.ndarray:
        """ndarray.copy() 透传 —— 转换一次并返回其副本。

        ⚠ 这是 2026-10-09 CAM1 真机"按键切到调参页 LCD 直接冻住"的根因修复:
        调参页 `threshold_tuner._highlight` 第一行就是 `frame_bgr.copy()`,
        LazyFrame 没这个方法 -> 主循环 tick 里每帧抛 AttributeError, 被
        module_manager 的 `except Exception: logger.exception(...)` 吞进
        logs/app.log（控制台看不见）-> 页面的 disp.show() 永远执行不到,
        而显示线程对非 MAIN 页整体让位 -> LCD 冻结在按键前最后一帧。
        修法: 补上代理方法, 语义与 ndarray.copy() 一致（独立副本）。
        """
        return self.numpy().copy()

    def numpy(self) -> np.ndarray:
        """拿到真实 BGR numpy（首次调用做 image2cv, 之后命中缓存）。"""
        if not self._converted:
            self._converted = True
            try:
                arr = self._convert(self._raw)
            except Exception:
                # 转换抛异常也退化为空数组 —— 理由同下: 这帧"没有像素可用"是
                # 可降级状态, 不该把整帧打断（单任务 try 兜底会记 error 并丢结果）。
                arr = None
            if arr is None:
                # 转换失败: 退化成空数组而非抛异常 —— 处理器里对 None 的处理是
                # "零值心跳不断流", 而抛异常会打断整帧(单任务 try 兜底会记 error)。
                # 空数组让 `.shape[:2]` 与切片都安全返回, 等价于"这一帧没有像素可用"。
                arr = np.zeros((0, 0, 3), np.uint8)
            self._numpy = arr
            # 转换后若拿到了真实尺寸, 用它校正 shape（初值可能来自估算/缺失）
            if self._shape is None and arr.ndim == 3:
                self._shape = (arr.shape[0], arr.shape[1])
        return self._numpy

    def __array__(self, dtype=None):
        arr = self.numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def __getitem__(self, item):
        return self.numpy()[item]

    def __iter__(self):
        return iter(self.numpy())

    def __len__(self) -> int:
        h, _w = self._shape if self._shape else (0, 0)
        return h

    def __repr__(self) -> str:
        h, w = self._shape if self._shape else (0, 0)
        return f"<LazyFrame {w}x{h} converted={self._converted}>"

    # ---- 构造入口 ----

    @classmethod
    def for_raw(cls, raw, convert: Callable[[object], np.ndarray],
                fallback: Optional[np.ndarray] = None,
                shape_hw: Optional[tuple] = None):
        """按平台能力选择: 有 raw 且有转换函数 -> LazyFrame; 否则退回 fallback。

        `fallback` 是"没有 maix 时本来会返回的东西"（通常是 None, 或已转好的
        真实 numpy）。linux/mock 平台不调用本类, 保持原 `read()` 语义。
        """
        if raw is None or convert is None:
            return fallback
        return cls(raw, convert, shape_hw=shape_hw)
