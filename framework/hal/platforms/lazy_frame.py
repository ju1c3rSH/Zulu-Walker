# -*- coding: utf-8 -*-
"""Defer maix Image -> BGR ndarray conversion until pixels are actually read.

The camera once called ``maix.image.image2cv(raw, copy=True)`` on every frame:
a ~921 KB copy for 640x480x3 costing 9-16 ms on device, the largest expense
besides ``find_blobs``. It was usually wasted, since C++ processors read
``camera.last_raw`` (zero-copy) and those needing a maix Image use ``_raw=True``
(also zero-copy). Only the cv2 fallback and the tuning page need a real ndarray.

The camera now hands out a ``LazyFrame`` proxy. Metadata (``shape``, ``ndim``,
``size``, ``dtype``) comes from the raw frame's width/height without copying,
so ``h, w = frame.shape[:2]`` is free. Pixel access (slicing, ``__array__``,
``copy()``, ``numpy()``) triggers one ``image2cv``, cached per frame. The C++
maix path thus copies nothing, while fallback paths return results bit-identical
to the old eager conversion.

Contract: the proxy passes truthiness checks without touching pixels (``.shape``
is the only pixel-free processor use); no caller uses ``isinstance(frame,
np.ndarray)``; ``np.asarray(lazy)`` equals ``lazy.numpy()``. OpenCV's Python
bindings require a real ndarray (``PyArray_Check``) and reject the proxy, so cv2
consumers such as the tuning page call ``numpy()`` or ``copy()`` first, and the
camera's ``read()`` returns a real ndarray. Without maix, ``for_raw()`` returns
the existing numpy/None fallback and never imports maix.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class LazyFrame:
    """Read-only proxy wrapping a raw maix Image and converter as a BGR ndarray."""

    __slots__ = ("_raw", "_convert", "_numpy", "_shape", "_converted")

    def __init__(self, raw, convert: Callable[[object], np.ndarray],
                 shape_hw: Optional[tuple] = None) -> None:
        self._raw = raw
        self._convert = convert        # convert(raw) -> BGR ndarray
        self._numpy: Optional[np.ndarray] = None
        self._shape = shape_hw         # (h, w) from raw metadata, zero-copy
        self._converted = False

    # ---- Metadata: zero-copy, no conversion ----

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

    # ---- Pixel access: trigger one conversion and cache ----

    def copy(self) -> np.ndarray:
        """Return an independent copy, mirroring ndarray.copy().

        Without it the tuning page's LCD freezes: ``threshold_tuner._highlight``
        starts with ``frame_bgr.copy()``, so the missing method raises
        AttributeError every frame, is swallowed into logs/app.log (invisible on
        the console), and prevents ``disp.show()`` from ever running, leaving
        the LCD on the last frame before the keypress.
        """
        return self.numpy().copy()

    def numpy(self) -> np.ndarray:
        """Return the real BGR ndarray (first call converts, later calls hit the cache)."""
        if not self._converted:
            self._converted = True
            try:
                arr = self._convert(self._raw)
            except Exception:
                # A conversion error also degrades to an empty array: "no pixels"
                # must not abort the frame (the task-level try logs and drops it).
                arr = None
            if arr is None:
                # Empty array instead of an exception: shape[:2] and slicing stay
                # safe, meaning "no pixels available for this frame".
                arr = np.zeros((0, 0, 3), np.uint8)
            self._numpy = arr
            # Correct shape from the real size when available (the initial value
            # may have been estimated or missing).
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

    # ---- Constructors ----

    @classmethod
    def for_raw(cls, raw, convert: Callable[[object], np.ndarray],
                fallback: Optional[np.ndarray] = None,
                shape_hw: Optional[tuple] = None):
        """Return a LazyFrame when raw and convert exist, else the fallback.

        ``fallback`` is what would have been returned without maix (usually None,
        or an already-converted ndarray). Linux/mock platforms never construct
        this class, preserving their original ``read()`` behaviour.
        """
        if raw is None or convert is None:
            return fallback
        return cls(raw, convert, shape_hw=shape_hw)
