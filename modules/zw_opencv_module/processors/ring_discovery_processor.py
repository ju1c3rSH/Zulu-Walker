import cv2
import numpy as np
from typing import Optional
from .base import Processor, VisionResult, ColorTrackable
from ..models.color import Color
from ..detectors.ring_detector import RingDetector


class RingDiscoveryProcessor(Processor):
    """色环发现处理器。

    输出 result_data:
      percent_error_x: int, [-5000, 5000], 0=画面中心, 负=偏左, 正=偏右
      percent_error_y: int, [-5000, 5000], 0=画面中心, 负=偏上, 正=偏下
      coordinate: 色环中心像素坐标
      confidence: float 置信度
    """

    def __init__(self, name: str = "ring_discovery"):
        super().__init__(name)
        self.detector = RingDetector()
        self.target_color: Optional[Color] = None

    def set_target_color(self, color: Optional[Color]):
        self.target_color = color

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        if self.target_color is None:
            return self._no_target()

        ring = self.detector.detect_ring(frame, self.target_color)

        if ring is None:
            return self._no_target()

        cx, cy = ring.coordinate
        h, w = frame.shape[:2]
        pe_x = int(((cx - w / 2.0) / (w / 2.0)) * 5000.0)
        pe_y = int(((cy - h / 2.0) / (h / 2.0)) * 5000.0)

        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": True,
                "percent_error_x": pe_x,
                "percent_error_y": pe_y,
                "coordinate": ring.coordinate,
                "confidence": ring.confidence,
            },
        )

    def _no_target(self) -> VisionResult:
        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": False,
                "percent_error_x": 0,
                "percent_error_y": 0,
                "coordinate": None,
                "confidence": 0.0,
            },
        )
