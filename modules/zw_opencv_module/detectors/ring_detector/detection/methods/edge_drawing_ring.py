from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget


class EdgeDrawingRingMethod(BaseRingDetectionMethod):
    """Edge-drawing ring detection with ROI-based color verification.

    Runs EdgeDrawing on grayscale (fast, C++ edge detector), fits ellipses
    to dilated edge contours, then verifies target color ONLY on a small
    ROI around each candidate ellipse.  No full-frame HSV, no full-frame
    mask allocation, no Kalman.
    """

    AXIS_RATIO_MAX = 1.5
    COLOR_MATCH_MIN_PX = 15

    def __init__(self, detector=None):
        super().__init__(name="edge_drawing_ring", detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None or self.detector.ed is None:
            return None

        small, scale, _ = self._scale_frame(frame)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bk = getattr(self.detector, "blur_kernel", 3)
        if bk % 2 == 0:
            bk += 1
        bs = getattr(self.detector, "blur_sigma", 1.5)
        blurred = cv2.GaussianBlur(gray, (bk, bk), bs)

        self.detector.ed.detectEdges(blurred)
        edges = self.detector.ed.getEdgeImage()

        if edges is None or cv2.countNonZero(edges) < 30:
            return None

        ek = getattr(self.detector, "edge_morph_kernel", 3)
        ei = getattr(self.detector, "edge_morph_iterations", 1)
        if ek % 2 == 0:
            ek += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=ei)
        self.detector._last_edge_preview = dilated

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = getattr(self.detector, "min_area", 150)
        best = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            try:
                (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            axis_ratio = max(a, b) / max(min(a, b), 1)
            if axis_ratio > self.AXIS_RATIO_MAX:
                continue

            color_px = self._verify_color_on_roi(
                small, int(cx), int(cy),
                int(a / 2), int(b / 2), angle,
                target_color,
            )
            if color_px < self.COLOR_MATCH_MIN_PX:
                continue

            shape_term = 1.0 / max(axis_ratio - 1.0, 0.01)
            area_term = area / 10000.0
            color_term = min(color_px / 200.0, 1.0)
            score = 0.35 * shape_term + 0.35 * area_term + 0.30 * color_term

            if score > best_score:
                best_score = score
                best = (cx / scale, cy / scale)

        if best is None:
            return None

        return create_ring_target(best, target_color, 100.0)

    def _verify_color_on_roi(self, bgr_frame, cx, cy, ra, rb, angle, target_color):
        h, w = bgr_frame.shape[:2]
        pad = 4
        half = max(ra, rb) + pad

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 <= x1 or y2 <= y1:
            return 0

        roi_bgr = bgr_frame[y1:y2, x1:x2]
        local_cx = cx - x1
        local_cy = cy - y1

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        ranges = self.detector.color_ranges.get(target_color, [])
        if not ranges:
            return 0
        roi_mask = None
        for lower, upper in ranges:
            chunk = cv2.inRange(roi_hsv, lower, upper)
            roi_mask = chunk if roi_mask is None else cv2.bitwise_or(roi_mask, chunk)
        if roi_mask is None or cv2.countNonZero(roi_mask) == 0:
            return 0

        roi_h, roi_w = roi_hsv.shape[:2]
        ellipse_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        try:
            cv2.ellipse(ellipse_mask, (int(local_cx), int(local_cy)),
                        (ra, rb), angle, 0, 360, 255, -1)
        except cv2.error:
            return 0

        intersection = cv2.bitwise_and(roi_mask, roi_mask, mask=ellipse_mask)
        return cv2.countNonZero(intersection)
