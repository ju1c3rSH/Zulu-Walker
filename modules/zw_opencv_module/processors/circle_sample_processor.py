# -*- coding: utf-8 -*-
"""教学样板: 最小化 Processor 实现。

演示如何接入 registry:
  1. 继承 Processor, 实现 process()
  2. 用 @register_processor("名字") 注册
  3. 启用: from modules.zw_opencv_module.processors import enable_optional
           enable_optional(["CircleTargetProcessor"])

新生任务可参照本文件编写自己的 rect / circle 检测 Processor。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..detectors.circle_target_detector import CircleTargetDetector
from ..models import Color
from .base import Processor, VisionResult
from .registry import register_processor
from utils.log_util import log_print


@register_processor("CircleTargetProcessor")
class CircleSampleProcessor(Processor):
    """圆形目标检测: 检出 -> 取最大目标 -> 输出中心偏差。"""

    def __init__(self, name: str = "circle_sample"):
        super().__init__(name)
        self.detector = CircleTargetDetector(name)
        self.target_color: Optional[Color] = None
        self._frame_w, self._frame_h = 640, 480

    def set_target_color(self, color: Optional[Color]) -> None:
        """None 表示不按颜色过滤。"""
        self.target_color = color

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        if frame is None:
            return VisionResult(self.name, success=False, error_message="empty frame")

        self._frame_h, self._frame_w = frame.shape[:2]
        color_str = self.target_color.name.lower().capitalize() if self.target_color else None
        targets = self.detector.detect_circle_targets(frame, target_color=color_str)

        if not targets.targets:
            return VisionResult(
                self.name,
                result_data={"targets": targets, "target_found": False},
                success=False,
                error_message="target not found",
            )

        target = max(targets.targets, key=lambda t: t.radius or 0)
        cx, cy = target.center_coordinates
        error_x = cx - self._frame_w // 2
        error_y = cy - self._frame_h // 2
        return VisionResult(
            self.name,
            result_data={
                "targets": targets,
                "target": target,
                "target_found": True,
                "center": (cx, cy),
                "error_x": error_x,
                "error_y": error_y,
            },
            success=True,
        )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if result.result_data and result.result_data.get("target_found"):
            cx, cy = result.result_data["center"]
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.drawMarker(frame, (self._frame_w // 2, self._frame_h // 2),
                           (255, 255, 255), cv2.MARKER_CROSS, 40, 1)
        return frame
