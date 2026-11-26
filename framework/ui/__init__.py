"""Panel-space UI primitives shared by the device and desktop backends."""

from __future__ import annotations

from framework.ui.canvas import Canvas, Rgb, scale_for
from framework.ui.panel import Gesture, Panel, Rect, TapGate, TouchEvent
from framework.ui.widgets import Button, Slider, find_button, find_slider

__all__ = [
    "Button",
    "Canvas",
    "Gesture",
    "Panel",
    "Rect",
    "Rgb",
    "Slider",
    "TapGate",
    "TouchEvent",
    "find_button",
    "find_slider",
    "scale_for",
]
