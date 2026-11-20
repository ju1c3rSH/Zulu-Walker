# -*- coding: utf-8 -*-
"""上屏策略: MAIN 与调试页共用同一份, 不再各存一份状态。"""
from __future__ import annotations

from dataclasses import dataclass

# interval_ms == 0 时回退的帧跳步长（改前行为）。
DEFAULT_FRAME_SKIP = 2


@dataclass(slots=True)
class PresentPolicy:
    enabled: bool = True
    interval_ms: float = 0.0        # 0 = 每 frame_skip 帧一次
    hud_interval_ms: float = 0.0
    frame_skip: int = DEFAULT_FRAME_SKIP
    stream_enabled: bool = True     # competition_mode 关推流, 不受关显示影响
