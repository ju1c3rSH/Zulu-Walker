from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget
from modules.zw_opencv_module.detectors._shared.kalman_utils import kalman_update


class HeuristicRingMethod(BaseRingDetectionMethod):
    """Two-stage ring detection for target-shaped rings (colored line, hollow center).

    Stage A (Heuristic, confidence=60):
        Color mask -> largest contour -> moments centroid.
        Handles partial/fragmented ring arcs. Guides robot toward target color.

    Stage B (Complete Ring, confidence=100):
        fitEllipse succeeds + axis_ratio <= 2.0.
        Ring is fully visible -> precise center for final confirmation.

    Auto-transitions between stages every frame based on contour quality.
    Stage transitions are logged immediately; steady-state every 60 frames.
    """

    AXIS_RATIO_MAX = 2.0
    CIRCULARITY_MIN = 0.40
    MORPH_KERNEL = 3
    MORPH_CLOSE_ITER = 2
    _LOG_INTERVAL = 60
    _COLOR_MATCH_MIN_PX = 15

    def __init__(self, detector=None):
        super().__init__(name="heuristic_ring", detector=detector)
        self._prev_stage: Optional[str] = None

    # ================================================================
    #  public entry
    # ================================================================

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None:
            return None

        ts = self.detector._get_tracking(target_color)
        small, scale, small_hw = self._scale_frame(frame)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = self._build_color_mask(hsv, target_color)
        if mask is None:
            self._info(f"[HeuristicRing] {target_color.name}: no color ranges configured")
            return self._fallback_predict(ts, target_color)

        mask_px = cv2.countNonZero(mask)
        if mask_px < 10:
            self._info(f"[HeuristicRing] target={target_color.name} mask={mask_px}px (too sparse)")
            return self._fallback_predict(ts, target_color)

        morphed = self._morph_mask(mask)
        self.detector._last_mask = mask.copy()
        self.detector._last_morphed = morphed.copy()

        result = self._try_roi(small, morphed, ts, target_color, scale, small_hw)
        if result is not None:
            return result

        result = self._try_global(morphed, ts, target_color, scale)
        if result is not None:
            return result

        return self._fallback_predict(ts, target_color)

    # ================================================================
    #  color mask
    # ================================================================

    def _build_color_mask(self, hsv: np.ndarray, color: Color) -> Optional[np.ndarray]:
        ranges = self.detector.color_ranges.get(color, [])
        if not ranges:
            return None
        mask = None
        for lower, upper in ranges:
            chunk = cv2.inRange(hsv, lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)
        return mask

    def _morph_mask(self, mask: np.ndarray) -> np.ndarray:
        k = self.MORPH_KERNEL
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel,
                                   iterations=self.MORPH_CLOSE_ITER)
        return closed

    # ================================================================
    #  ROI / global dispatch
    # ================================================================

    def _try_roi(self, frame, morphed, ts, target_color, scale, small_hw):
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            max_miss = getattr(self.detector, "max_roi_miss", 5)
            self._info(f"[HeuristicRing] ROI skipped: miss={ts.roi_miss_count}/{max_miss} "
                       f"center={'none' if ts.last_center is None else 'known'}")
            ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None

        x1, y1, x2, y2 = roi
        roi_morphed = morphed[y1:y2, x1:x2]
        return self._detect_and_finalize(roi_morphed, ts, target_color, scale,
                                         offset=(x1, y1), context="ROI")

    def _try_global(self, morphed, ts, target_color, scale):
        return self._detect_and_finalize(morphed, ts, target_color, scale,
                                         offset=(0, 0), context="global")

    # ================================================================
    #  core: two-stage detection
    # ================================================================

    def _detect_and_finalize(self, binary, ts, target_color, scale,
                              offset, context):
        min_area = getattr(self.detector, "min_area", 100)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"mask_morphed={cv2.countNonZero(binary)}px no contours")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_area:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"largest_area={area:.0f} < min_area={min_area} "
                       f"total_contours={len(contours)}")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None

        center, conf = self._evaluate_contour(largest, target_color, scale, offset)

        self._log_detection(context, target_color, center, area, conf)

        if context == "ROI":
            ts.roi_miss_count = 0

        return self._finalize(center, conf, ts, target_color, scale)

    def _evaluate_contour(self, contour, target_color, scale, offset):
        """Try Stage B (fitEllipse) first; fall back to Stage A (moments)."""
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        ox, oy = offset

        # --- Stage B: complete ring ---
        if len(contour) >= 5 and circularity > self.CIRCULARITY_MIN:
            try:
                ellipse = cv2.fitEllipse(contour)
                (ecx, ecy), (ea, eb), _ = ellipse
                if ea > 0 and eb > 0:
                    axis_ratio = max(ea, eb) / min(ea, eb)
                    if axis_ratio <= self.AXIS_RATIO_MAX:
                        center = ((ecx + ox) / scale, (ecy + oy) / scale)
                        return center, 100.0
            except cv2.error:
                pass

        # --- Stage A: heuristic centroid ---
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = ((cx + ox) / scale, (cy + oy) / scale)
        return center, 60.0

    # ================================================================
    #  finalize / kalman / fallback
    # ================================================================

    def _finalize(self, center, conf, ts, target_color, scale):
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
            self._info("[HeuristicRing] fallback=kalman")
            return create_ring_target(result, target_color, 60.0)

        if ts._center_history:
            cxs = [p[0] for p in ts._center_history]
            cys = [p[1] for p in ts._center_history]
            center = (sum(cxs) / len(cxs), sum(cys) / len(cys))
            self._info("[HeuristicRing] fallback=history_avg")
            return create_ring_target(center, target_color, 40.0)

        return None

    # ================================================================
    #  logging
    # ================================================================

    def _tick(self) -> int:
        if not hasattr(self.detector, '_ring_log_frame'):
            self.detector._ring_log_frame = 0
        self.detector._ring_log_frame += 1
        return self.detector._ring_log_frame

    def _info(self, msg: str, force: bool = False) -> None:
        frame = self._tick()
        if force or frame % self._LOG_INTERVAL == 0:
            print(msg)

    def _log_detection(self, context, target_color, center, area, conf):
        stage = "COMPLETE" if conf >= 100 else "HEURISTIC"
        prev = self._prev_stage
        self._prev_stage = stage
        force = (prev is not None and prev != stage)
        msg = (f"[HeuristicRing] {context}: {stage} "
               f"target={target_color.name} "
               f"center=({center[0]:.0f},{center[1]:.0f}) "
               f"area={area:.0f} conf={conf}")
        self._info(msg, force=force)
