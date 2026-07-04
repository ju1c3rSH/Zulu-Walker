import cv2
import numpy as np
from typing import Optional
from .base import Processor, VisionResult
from ..models import Color


class RingTrackProcessor(Processor):
    """色环对准处理器（stub — 后续实现）"""

    def __init__(self, name: str = "ring_track"):
        super().__init__(name)
        self.target_color: Optional[Color] = None

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        return VisionResult(
            task_name=self.name,
            success=False,
            error_message="Not implemented",
        )

    def set_target_color(self, color: Optional[Color]):
        self.target_color = color
