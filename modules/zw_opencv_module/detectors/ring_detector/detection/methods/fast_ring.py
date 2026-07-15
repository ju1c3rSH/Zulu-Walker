from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget
from modules.zw_opencv_module.detectors._shared.kalman_utils import kalman_update


class FastRingMethod(BaseRingDetectionMethod):

    def __init__(self, detector=None):
        super().__init__(name="fast_ring", detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None:
            return None
        ts = self.detector._get_tracking(target_color)
        small, scale, small_hw = self._scale_frame(frame)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = self._build_color_mask(hsv, target_color)
        morphed = self._morph_mask(mask)
        self.detector._last_mask = mask.copy()
        self.detector._last_morphed = morphed.copy()

        result = self._try_roi(small, hsv, morphed, ts, target_color, scale, small_hw)
        if result is None:
            result = self._try_global(small, hsv, morphed, ts, target_color, scale)
        if result is None:
            result = self._fallback_predict(ts, target_color)
        return result

    def _build_color_mask(self, hsv: np.ndarray, color: Color) -> np.ndarray:
        ranges = self.detector.color_ranges.get(color, [])
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        return mask

    def _morph_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return morphed

    def _try_roi(self, frame, hsv, morphed, ts, target_color, scale, small_hw):
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None
        x1, y1, x2, y2 = roi
        roi_morphed = morphed[y1:y2, x1:x2]
        result = self._detect_ring(roi_morphed, target_color, scale, offset=(x1, y1))
        if result is not None:
            return self._finalize(result, target_color, ts, scale)
        ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
        return None

    def _try_global(self, frame, hsv, morphed, ts, target_color, scale):
        result = self._detect_ring(morphed, target_color, scale, offset=(0, 0))
        if result is not None:
            return self._finalize(result, target_color, ts, scale)
        return None

    def _detect_ring(self, morphed, target_color, scale, offset):
        contours, hierarchy = cv2.findContours(
            morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or hierarchy is None:
            if not hasattr(self.detector, '_ring_log_frame'):
                self.detector._ring_log_frame = 0
            self.detector._ring_log_frame += 1
            if self.detector._ring_log_frame % 60 == 0:
                print(f"[FastRing] target={target_color.name} no contours")
            return None

        outer_candidates = []
        hierarchy = hierarchy[0]
        min_area = getattr(self.detector, "min_area", 100)
        for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
            if h[3] != -1:
                continue
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            try:
                (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            axis_ratio = max(a, b) / max(min(a, b), 1)
            if axis_ratio > 1.5:
                continue
            ox, oy = offset
            orig_cx = (cx + ox) / scale
            orig_cy = (cy + oy) / scale
            outer_candidates.append(((orig_cx, orig_cy), area, axis_ratio))

        if not hasattr(self.detector, '_ring_log_frame'):
            self.detector._ring_log_frame = 0
        self.detector._ring_log_frame += 1
        frame = self.detector._ring_log_frame

        if not outer_candidates:
            if frame % 60 == 0:
                print(f"[FastRing] target={target_color.name} "
                      f"contours={len(contours)} outer_candidates=0")
            return None

        outer_candidates.sort(key=lambda x: x[1], reverse=True)
        center = outer_candidates[0][0]

        if frame % 60 == 0:
            print(f"[FastRing] target={target_color.name} "
                  f"contours={len(contours)} outer_candidates={len(outer_candidates)} "
                  f"best: area={outer_candidates[0][1]:.0f} axis_ratio={outer_candidates[0][2]:.2f}")

        return center, target_color, 100.0

    def _finalize(self, result, target_color, ts, scale):
        center, color, conf = result
        cx, cy = center

        smoothed = self._kalman_predict(ts, (cx, cy))
        ts.last_center = (cx, cy)
        ts.roi_miss_count = 0
        ts._center_history.append((cx, cy))
        final = smoothed if smoothed is not None else (cx, cy)

        return create_ring_target(final, target_color, conf)

    def _kalman_predict(self, ts, center):
        cx, cy = center
        result, ts.tracking_initialized, ts.lost_frames, ts._last_predict_time = kalman_update(
            ts.kf, (cx, cy),
            ts.tracking_initialized, ts.lost_frames,
            getattr(self.detector, "max_lost_frames", 30),
            getattr(self.detector, "q_base", 0.1),
            getattr(self.detector, "q_vel_base", 0.01),
            ts._last_predict_time,
        )
        return result

    def _fallback_predict(self, ts, target_color):
        result, ts.tracking_initialized, ts.lost_frames, ts._last_predict_time = kalman_update(
            ts.kf, None,
            ts.tracking_initialized, ts.lost_frames,
            getattr(self.detector, "max_lost_frames", 30),
            getattr(self.detector, "q_base", 0.1),
            getattr(self.detector, "q_vel_base", 0.01),
            ts._last_predict_time,
        )
        if result is not None:
            cx, cy = result
            ts.last_center = (cx, cy)
            ts._center_history.append((cx, cy))
            return create_ring_target(result, target_color, 100.0)

        if ts._center_history:
            cx = sum(p[0] for p in ts._center_history) / len(ts._center_history)
            cy = sum(p[1] for p in ts._center_history) / len(ts._center_history)
            return create_ring_target((cx, cy), target_color, 100.0)

        return None
