# -*- coding: utf-8 -*-
"""LazyFrame — 把 maix Image 转成 BGR numpy 这件事，拖到真要用像素时才做。

设备端相机原本每帧都调 `maix.image.image2cv(raw, copy=True)`，640x480x3 一次拷
921KB，台架真机实测要 9~16ms，是 find_blobs 之外最大的一块。多数时候这份拷贝白做了：
TARGET/EXPLOSIVE 走 C++ `find_blobs`，直接吃 `camera.last_raw`（零拷贝）；RESCUE 走
`_raw=True`，也直接吃 maix Image。只有 maix 不可用时回退 cv2，以及调参页，才真需要
这份 BGR numpy。

所以相机改成交出一个 `LazyFrame` 代理，不再一开始就转换：

  · `.shape` / `.ndim` / `.size` / `.dtype` 从 maix Image 的元数据
    (`width()/height()`) 直接给，零拷贝，处理器里的 `h, w = frame.shape[:2]` 白拿。
  · 切片、`__array__`、`.copy()`、`.numpy()` 这些真正碰像素的操作，才触发一次
    `image2cv`，结果缓存起来，同一帧只转一次。

于是设备上 maix 可用、又走 C++ 路径时，一次拷贝都不发生；cv2 回退、调参页、RESCUE
传 numpy 这几种情况，结果和原来逐位一致。

代理要满足几条：能像 `np.ndarray` 一样通过 `frame is None` 之外的判断（处理器里唯一
不碰像素的用法是 `.shape`）；不要求 `isinstance(frame, np.ndarray)`，全仓库查过，
处理器和协处理器都没做这种判断；`np.asarray(lazy)` 和 `lazy.numpy()` 等价，方便
第三方库接手。

OpenCV 的 Python 绑定不吃这个代理。OpenCV 4.9/5.0 都会报
"src is not a numpy array, neither a scalar"：`cv2.resize/cvtColor` 的入参转换不走
`__array__` / `__array_interface__`，只认 `PyArray_Check` 为真的真 ndarray。所以调参页
这类要喂 cv2 的地方，得先 `.numpy()`/`.copy()` 拿真数组；相机侧 `read()` 是调参页的
取帧入口，最后统一返回真 numpy。

PC 端没有 maix，不构造本类：`LazyFrame.for_raw()` 在 raw 为 None 或转换函数不可用时
直接返回现有 numpy / None，不会引入 maix 依赖。
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

        缺这个方法会让调参页 LCD 直接冻住: 调参页 `threshold_tuner._highlight`
        第一行就是 `frame_bgr.copy()`, 缺方法 -> 每帧抛 AttributeError 被吞进
        logs/app.log（控制台看不见）-> disp.show() 永远执行不到 -> LCD 冻结在
        按键前最后一帧。效果跟 ndarray.copy() 一样（独立副本）。
        """
        return self.numpy().copy()

    def numpy(self) -> np.ndarray:
        """拿到真实 BGR numpy（首次调用做 image2cv, 之后命中缓存）。"""
        if not self._converted:
            self._converted = True
            try:
                arr = self._convert(self._raw)
            except Exception:
                # 转换抛异常也退化为空数组 —— 这帧"没有像素可用"还能继续跑,
                # 不该把整帧打断（单任务 try 的异常处理会记 error 并丢结果）。
                arr = None
            if arr is None:
                # 转换失败: 退化成空数组而非抛异常 —— 空数组让 `.shape[:2]` 与
                # 切片都安全返回, 等价于"这一帧没有像素可用"。
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
        真实 numpy）。linux/mock 平台不调用本类, 保持原 `read()` 的行为。
        """
        if raw is None or convert is None:
            return fallback
        return cls(raw, convert, shape_hw=shape_hw)
