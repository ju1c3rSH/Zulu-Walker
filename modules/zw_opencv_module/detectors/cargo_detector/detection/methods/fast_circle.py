import cv2
import numpy as np
from typing import Optional, Tuple

from .base import BaseDetectionMethod
from .....models.color import Color
from .....models.cargo import CargoItem
from .. import kalman_filter
from utils.log_util import log_print



class FastCircleDetectionWithColorMethod(BaseDetectionMethod):
    TARGET_W = 640
    TARGET_H = 480

    def __init__(self, name: str = "fast_circle", detector=None):
        super().__init__(name=name, detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[CargoItem]:
        ts = self.detector._get_tracking(target_color)
        small, scale = self._scale_frame(frame)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        result = self._try_roi(small, hsv, ts, target_color, scale)
        if result is not None:
            return result

        result = self._try_global(small, hsv, ts, target_color, scale)
        if result is not None:
            return result

        return self._fallback_predict(ts, target_color, scale)

    def _scale_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame, 1.0
        s = min(self.TARGET_H / h, self.TARGET_W / w, 1.0)
        if s < 1.0:
            return cv2.resize(
                frame, (int(w * s), int(h * s)),
                interpolation=cv2.INTER_AREA
            ), s
        return frame, 1.0

    def _decide_roi(self, frame: np.ndarray, ts, scale: float
                    ) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        if ts.last_center is None:
            return None
        if ts.roi_miss_count >= self.detector.max_roi_miss:
            return None
        cx = int(ts.last_center[0] * scale)
        cy = int(ts.last_center[1] * scale)
        half = self.detector.roi_size // 2
        h, w = frame.shape[:2]
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2], (x1, y1)

    def _try_roi(self, full_frame: np.ndarray, full_hsv: np.ndarray,
                 ts, target_color: Color, scale: float) -> Optional[CargoItem]:
        roi = self._decide_roi(full_frame, ts, scale)
        if roi is None:
            return None
        sub_img, offset = roi
        sub_hsv = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)

        mask = self._adaptive_color_segment(sub_hsv, ts, target_color)
        if mask is None:
            ts.roi_miss_count += 1
            return None

        morphed = self._morph_process(mask)
        result = self._find_best_contour(morphed)
        if result is None:
            ts.roi_miss_count += 1
            return None

        center, radius = result
        center = (center[0] + offset[0], center[1] + offset[1])
        return self._finalize(center, radius, ts, target_color, scale)

    def _try_global(self, frame: np.ndarray, hsv: np.ndarray,
                    ts, target_color: Color, scale: float) -> Optional[CargoItem]:
        mask = self._adaptive_color_segment(hsv, ts, target_color)
        if mask is None:
            return None

        morphed = self._morph_process(mask)
        self.detector._last_mask = mask
        self.detector._last_morphed = morphed

        result = self._find_best_contour(morphed)
        if result is None:
            if self.detector._fast_log_frame % 60 == 0:
                log_print(f"[FastCircle] global: no contour matched")
            return None

        center, radius = result
        return self._finalize(center, radius, ts, target_color, scale)

    def _adaptive_color_segment(self, hsv: np.ndarray, ts,
                                target_color: Color) -> Optional[np.ndarray]:
        if target_color not in self.detector.color_ranges:
            return None

        ranges = self.detector.color_ranges[target_color]

        mask_coarse = None
        for lower, upper in ranges:
            wide_upper = np.array([upper[0], 255, 255], dtype=np.uint8)
            wide_lower = np.array([lower[0], 0, 0], dtype=np.uint8)
            chunk = cv2.inRange(hsv, wide_lower, wide_upper)
            mask_coarse = chunk if mask_coarse is None else cv2.bitwise_or(mask_coarse, chunk)

        if mask_coarse is None or cv2.countNonZero(mask_coarse) < self.detector.coarse_min_pixels:
            if not hasattr(self.detector, '_fast_log_frame'):
                self.detector._fast_log_frame = 0
            self.detector._fast_log_frame += 1
            if self.detector._fast_log_frame % 60 == 0:
                coarse_px = cv2.countNonZero(mask_coarse) if mask_coarse is not None else 0
                log_print(f"[FastCircle] target={target_color.name} coarse_mask={coarse_px}px -> too few pixels, skipping")
            return None

        coarse_px = cv2.countNonZero(mask_coarse)
        h, w = hsv.shape[:2]
        total_px = h * w
        coarse_ratio = coarse_px / total_px

        if coarse_ratio > self.detector.coarse_ratio_threshold:
            if not hasattr(self.detector, '_fast_log_frame'):
                self.detector._fast_log_frame = 0
            self.detector._fast_log_frame += 1
            if self.detector._fast_log_frame % 60 == 0:
                log_print(f"[FastCircle] target={target_color.name} coarse={coarse_px}px "
                      f"({coarse_ratio:.0%} of frame) -> too large, skipping adaptive SV, "
                      f"falling back to hardcoded ranges")
            return self._build_hardcoded_mask(hsv, target_color)

        s_raw, v_raw = self._compute_adaptive_sv(hsv, ts, mask_coarse)
        s_low = max(s_raw, self.detector.color_ranges[target_color][0][0][1])
        v_low = max(v_raw, self.detector.color_ranges[target_color][0][0][2])

        mask_fine = None
        for lower, upper in ranges:
            adjusted_lower = np.array([lower[0], s_low, v_low], dtype=np.uint8)
            adjusted_upper = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
            chunk = cv2.inRange(hsv, adjusted_lower, adjusted_upper)
            mask_fine = chunk if mask_fine is None else cv2.bitwise_or(mask_fine, chunk)

        if not hasattr(self.detector, '_fast_log_frame'):
            self.detector._fast_log_frame = 0
        self.detector._fast_log_frame += 1
        if self.detector._fast_log_frame % 60 == 0:
            fine_px = cv2.countNonZero(mask_fine) if mask_fine is not None else 0
            log_print(f"[FastCircle] target={target_color.name} coarse={coarse_px}px "
                  f"s_raw={s_raw} s_clamped={s_low} v_raw={v_raw} v_clamped={v_low} "
                  f"fine={fine_px}px")

        return mask_fine

    def _build_hardcoded_mask(self, hsv: np.ndarray, target_color: Color) -> np.ndarray:
        ranges = self.detector.color_ranges[target_color]
        mask = None
        for lower, upper in ranges:
            chunk = cv2.inRange(hsv, lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)
        return mask

    def _compute_adaptive_sv(self, hsv: np.ndarray, ts,
                             mask: np.ndarray) -> Tuple[int, int]:
        s_channel = hsv[:, :, 1][mask > 0]
        v_channel = hsv[:, :, 2][mask > 0]

        if len(s_channel) < self.detector.sv_min_samples:
            return (self.detector.sv_fallback_s, self.detector.sv_fallback_v)

        s_low = int(np.percentile(s_channel, self.detector.sv_percentile))
        v_low = int(np.percentile(v_channel, self.detector.sv_percentile))

        if ts._ema_s is None:
            ts._ema_s = s_low
            ts._ema_v = v_low
        else:
            alpha = self.detector.ema_alpha
            ts._ema_s = int(alpha * s_low + (1 - alpha) * ts._ema_s)
            ts._ema_v = int(alpha * v_low + (1 - alpha) * ts._ema_v)

        return (ts._ema_s, ts._ema_v)

    def _morph_process(self, mask: np.ndarray) -> np.ndarray:
        k_open = self.detector.kernel_open
        if k_open % 2 == 0:
            k_open += 1
        k_close = self.detector.kernel_close
        if k_close % 2 == 0:
            k_close += 1

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        return closed

    def _find_best_contour(self, binary: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.detector.min_area:
            if self.detector._fast_log_frame % 60 == 0:
                log_print(f"[FastCircle] largest contour area={area:.1f} < min_area={self.detector.min_area}")
            return None

        peri = cv2.arcLength(largest, True)
        if peri == 0:
            return None

        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < self.detector.min_circularity:
            if self.detector._fast_log_frame % 60 == 0:
                log_print(f"[FastCircle] circularity={circularity:.3f} < threshold={self.detector.min_circularity}")
            return None

        if self.detector._fast_log_frame % 60 == 0:
            log_print(f"[FastCircle] found: area={area:.1f} circularity={circularity:.3f}")

        if len(largest) >= self.detector.ellipse_min_contour_points:
            try:
                (ecx, ecy), (ea, eb), _ = cv2.fitEllipse(largest)
                if ea > 0 and eb > 0:
                    axis_ratio = max(ea, eb) / min(ea, eb)
                    if axis_ratio <= self.detector.ellipse_max_axis_ratio:
                        radius = (ea + eb) / 4.0
                        return ((float(ecx), float(ecy)), float(radius))
            except cv2.error:
                pass

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        inside = cv2.pointPolygonTest(largest, (float(cx), float(cy)), False)
        if inside < 0:
            return None

        radius = np.sqrt(area / np.pi)
        return ((float(cx), float(cy)), float(radius))

    def _finalize(self, center: Tuple[float, float], radius: float,
                  ts, target_color: Color, scale: float) -> Optional[CargoItem]:
        if scale < 1.0:
            center = (center[0] / scale, center[1] / scale)
            radius = radius / scale

        smoothed = self._kalman_predict(ts, center)

        ts.last_center = center
        ts.roi_miss_count = 0

        ts._center_history.append(center)
        ts._radius_history.append(radius)

        final_center = smoothed if smoothed is not None else center
        return self._build_cargo_item(final_center, target_color, radius)

    def _kalman_predict(self, ts,
                        measurement: Optional[Tuple[float, float]]
                        ) -> Optional[Tuple[float, float]]:
        result, ts.tracking_initialized, ts.lost_frames, ts._last_predict_time = (
            kalman_filter.kalman_update(
                ts.kf, measurement,
                ts.tracking_initialized,
                ts.lost_frames,
                self.detector.max_lost_frames,
                self.detector._kf_q_base,
                self.detector._kf_q_vel_base,
                ts._last_predict_time,
            )
        )
        return result

    def _fallback_predict(self, ts, target_color: Color,
                          scale: float) -> Optional[CargoItem]:
        center = self._kalman_predict(ts, None)
        if center is not None:
            return self._build_cargo_item(center, target_color, is_predicted=True)

        if len(ts._center_history) > 0:
            avg = (
                sum(c[0] for c in ts._center_history) / len(ts._center_history),
                sum(c[1] for c in ts._center_history) / len(ts._center_history),
            )
            return self._build_cargo_item(avg, target_color, is_predicted=True)

        return None

    def _build_cargo_item(self, center: Tuple[float, float], target_color: Color,
                          radius: Optional[float] = None,
                          is_predicted: bool = False) -> CargoItem:
        from ..target_creation import create_cargo_item
        return create_cargo_item(center, target_color, radius=radius,
                                 is_predicted=is_predicted)
