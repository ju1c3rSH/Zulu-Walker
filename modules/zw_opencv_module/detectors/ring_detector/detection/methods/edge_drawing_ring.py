from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget


class EdgeDrawingRingMethod(BaseRingDetectionMethod):
    """Edge-drawing ring detection with color verification.

    Finds ellipse-shaped edges, then verifies the candidate contains the
    target color.  Uses absolute pixel count (>=15px) to avoid the
    hollow-center penalty inherent in fill-ratio thresholds.
    """

    AXIS_RATIO_MAX = 1.5
    COLOR_MATCH_MIN_PX = 15  # absolute pixels of target color inside ellipse

    def __init__(self, detector=None):
        super().__init__(name="edge_drawing_ring", detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None or self.detector.ed is None:
            return None

        small, scale, small_hw = self._scale_frame(frame)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bk = getattr(self.detector, "blur_kernel", 5)
        if bk % 2 == 0:
            bk += 1
        bs = getattr(self.detector, "blur_sigma", 1.5)
        blurred = cv2.GaussianBlur(gray, (bk, bk), bs)

        self.detector.ed.detectEdges(blurred)
        edges = self.detector.ed.getEdgeImage()
        self.detector._last_edge_preview = edges.copy()

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        color_mask = self._build_color_mask(hsv, target_color)

        ek = getattr(self.detector, "edge_morph_kernel", 3)
        ei = getattr(self.detector, "edge_morph_iterations", 1)
        if ek % 2 == 0:
            ek += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=ei)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = getattr(self.detector, "min_area", 100)
        best = None
        best_score = 0.0
        total_candidates = 0

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
            total_candidates += 1

            color_px = self._count_color_in_ellipse(color_mask, int(cx), int(cy),
                                                    int(a / 2), int(b / 2), angle)
            if color_px < self.COLOR_MATCH_MIN_PX:
                continue

            shape_term = 1.0 / max(axis_ratio - 1.0, 0.01)
            area_term = area / 10000.0
            color_term = min(color_px / 200.0, 1.0)
            score = 0.35 * shape_term + 0.35 * area_term + 0.30 * color_term

            if score > best_score:
                best_score = score
                orig_cx = cx / scale
                orig_cy = cy / scale
                best = (orig_cx, orig_cy)

        if not hasattr(self.detector, '_ring_log_frame'):
            self.detector._ring_log_frame = 0
        self.detector._ring_log_frame += 1
        frame_no = self.detector._ring_log_frame

        if frame_no % 60 == 0:
            if best is not None:
                print(f"[EdgeDrawRing] target={target_color.name} "
                      f"contours={len(contours)} candidates={total_candidates} "
                      f"best: score={best_score:.2f} center=({best[0]:.0f},{best[1]:.0f})")
            else:
                mask_px = cv2.countNonZero(color_mask) if color_mask is not None else 0
                print(f"[EdgeDrawRing] target={target_color.name} "
                      f"contours={len(contours)} candidates={total_candidates} "
                      f"color_mask={mask_px}px no match")

        if best is None:
            return None

        return create_ring_target(best, target_color, 100.0)

    def _build_color_mask(self, hsv: np.ndarray, color: Color) -> Optional[np.ndarray]:
        ranges = self.detector.color_ranges.get(color, [])
        if not ranges:
            return None
        mask = None
        for lower, upper in ranges:
            chunk = cv2.inRange(hsv, lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)
        return mask

    def _count_color_in_ellipse(self, color_mask, cx, cy, ra, rb, angle):
        if color_mask is None:
            return 0
        h, w = color_mask.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(roi_mask, (cx, cy), (ra, rb), angle, 0, 360, 255, -1)
        intersection = cv2.bitwise_and(color_mask, color_mask, mask=roi_mask)
        return cv2.countNonZero(intersection)
