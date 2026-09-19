from abc import ABC, abstractmethod
from typing import Optional
import cv2
import numpy as np

from modules.zw_opencv_module.models.ring import RingTarget
from modules.zw_opencv_module.models.color import Color


TARGET_W = 640
TARGET_H = 480


class BaseRingDetectionMethod(ABC):

    def __init__(self, name: str = "base_ring_detection", detector=None):
        self.name = name
        self.detector = detector

    @abstractmethod
    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        pass

    def _scale_frame(self, frame: np.ndarray, target_width: int = TARGET_W):
        h, w = frame.shape[:2]
        if w == target_width:
            scale = 1.0
            small = frame
        else:
            scale = target_width / w
            new_h = int(h * scale)
            small = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
        return small, scale, small.shape[:2]

    def _decide_roi(self, ts, scale: float, small_hw: tuple):
        if ts.last_center is None:
            return None
        roi_size = getattr(self.detector, "roi_size", 160)
        if ts.roi_miss_count >= getattr(self.detector, "max_roi_miss", 30):
            return None
        small_h, small_w = small_hw
        cx, cy = ts.last_center
        cx_s = int(cx * scale)
        cy_s = int(cy * scale)
        half = int(roi_size * scale // 2)
        x1 = max(0, cx_s - half)
        y1 = max(0, cy_s - half)
        x2 = min(small_w, cx_s + half)
        y2 = min(small_h, cy_s + half)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
