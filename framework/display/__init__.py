# -*- coding: utf-8 -*-
"""显示前台公共层: 类型 / 格式检查 / 上屏服务 / taps。"""
from framework.display.gate import warn_slow_format
from framework.display.policy import PresentPolicy
from framework.display.service import PresenterService
from framework.display.sources import DisplaySource
from framework.display.types import (
    DisplayCaps,
    Frame,
    PixelFormat,
    PresentResult,
)

__all__ = [
    "DisplayCaps",
    "DisplaySource",
    "Frame",
    "PixelFormat",
    "PresentPolicy",
    "PresenterService",
    "PresentResult",
    "warn_slow_format",
]
