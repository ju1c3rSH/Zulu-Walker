# -*- coding: utf-8 -*-
"""上屏 tap: LCD / JPEG 推流 / 空实现。"""
from framework.display.taps.base import BaseTap, Tap
from framework.display.taps.jpeg import JpegTap
from framework.display.taps.lcd import LcdTap
from framework.display.taps.null import NullTap

__all__ = ["BaseTap", "JpegTap", "LcdTap", "NullTap", "Tap"]
