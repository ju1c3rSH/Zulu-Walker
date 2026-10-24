# -*- coding: utf-8 -*-
"""显示前台公共层: 帧/格式类型 + 格式闸门（与平台 display HAL 解耦）。"""
from framework.display.gate import warn_slow_format
from framework.display.types import (
    DisplayCaps,
    Frame,
    PixelFormat,
    PresentResult,
)

__all__ = [
    "DisplayCaps",
    "Frame",
    "PixelFormat",
    "PresentResult",
    "warn_slow_format",
]
