# -*- coding: utf-8 -*-
"""显示前台公共类型: 帧格式 / 帧 / 平台能力 / 呈现结果。

BGR888 故意留成显式的"慢路径"标记: MaixCAM1 show() 只白名单 RGB888 等少数格式,
其它格式会在 show() 内部整帧 to_format(FMT_RGB888)（堆分配 + 逐像素通道交换）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, Optional

import numpy as np


class PixelFormat(Enum):
    """帧的真实字节格式; NATIVE = 平台画布原生, 零转换。"""

    NATIVE = "native"
    RGB888 = "rgb888"
    BGR888 = "bgr888"      # ⚠ 慢路径标记: maix show() 内部要整帧转 RGB888
    GRAY8 = "gray8"
    JPEG = "jpeg"

    @staticmethod
    def of(frame: Any) -> Optional["PixelFormat"]:
        """判帧格式; 判不出（如 LazyFrame）返回 None, 由调用方拒绝。"""
        if isinstance(frame, np.ndarray):
            return PixelFormat.BGR888
        fmt_fn = getattr(frame, "format", None)
        if callable(fmt_fn):
            try:
                return _FMT_MAP.get(fmt_fn())
            except Exception:
                return None
        return None


@dataclass(frozen=True, slots=True)
class Frame:
    """一帧的句柄 + 元数据; 只带引用, 不持有像素。"""

    handle: Any
    fmt: PixelFormat
    w: int
    h: int
    seq: int = 0
    origin: str = "camera"     # "camera" | "canvas"
    ts: float = 0.0


@dataclass(frozen=True, slots=True)
class DisplayCaps:
    """平台显示能力; accepted 之外的格式会在 show() 内被整帧救一次。"""

    width: int
    height: int
    native_format: PixelFormat
    accepted: FrozenSet[PixelFormat]
    encode_in_show: bool


@dataclass(frozen=True, slots=True)
class PresentResult:
    ok: bool
    quit_requested: bool = False
    elapsed_ms: float = 0.0
    reason: str = ""


# 名字 -> 枚举, 只建一次; 逐帧 dir(Format) 会吃掉探针全部收益。
_BY_FMT_NAME = {
    "RGB888": PixelFormat.RGB888,
    "BGR888": PixelFormat.BGR888,
    "GRAYSCALE": PixelFormat.GRAY8,
    "JPEG": PixelFormat.JPEG,
}


def _build_format_map() -> dict:
    """一次性建 maix.image.Format 值 -> PixelFormat 表; PC 无 maix 则空表。"""
    try:
        from maix import image as _mi
        members = getattr(_mi, "Format", None)
        if members is None:
            return {}
        return {getattr(members, a): _BY_FMT_NAME[a[4:]]
                for a in dir(members)
                if a.startswith("FMT_") and a[4:] in _BY_FMT_NAME}
    except Exception:
        return {}


_FMT_MAP = _build_format_map()
