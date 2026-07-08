from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget


class EdgeDrawingRingMethod(BaseRingDetectionMethod):

    def __init__(self, detector=None):
        super().__init__(name="edge_drawing_ring", detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None or self.detector.ed is None:
            return None

        small, scale, small_hw = self._scale_frame(frame)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bk = getattr(self.detector, "blur_kernel", 5)
        bs = getattr(self.detector, "blur_sigma", 1.5)
        blurred = cv2.GaussianBlur(gray, (bk, bk), bs)

        self.detector.ed.detectEdges(blurred)
        edges = self.detector.ed.getEdgeImage()
        self.detector._last_edge_preview = edges.copy()

        ek = getattr(self.detector, "edge_morph_kernel", 3)
        ei = getattr(self.detector, "edge_morph_iterations", 1)
        kernel = np.ones((ek, ek), np.uint8)
        dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=ei)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = getattr(self.detector, "min_area", 100)
        best = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            try:
                (cx, cy), (a, b), _ = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            axis_ratio = max(a, b) / max(min(a, b), 1)
            if axis_ratio > 1.5:
                continue

            score = 0.5 * (1.0 / max(axis_ratio - 1.0, 0.01)) + 0.5 * (area / 10000.0)
            if score > best_score:
                best_score = score
                orig_cx = cx / scale
                orig_cy = cy / scale
                best = (orig_cx, orig_cy)

        if best is None:
            return None

        return create_ring_target(best, target_color, 100.0)
