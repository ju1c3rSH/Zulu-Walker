import time
import cv2
import numpy as np
from typing import Optional, Tuple, List

from .base import BaseDetectionMethod
from .....models.color import Color
from .....models.cargo import CargoItem
from .. import kalman_filter


class EdgeDrawingCircleMethod(BaseDetectionMethod):
    """基于 EdgeDrawing 边缘检测的颜色圆识别方法。

    适用于俯视正拍、纯色圆形物料、边缘存在反光的场景。
    亚像素中心采用"边缘拟合椭圆 + 颜色 mask 矩"的混合策略。
    """

    TARGET_W = 640
    TARGET_H = 480
    MAX_AXIS_RATIO = 1.5

    def __init__(self, name: str = "edge_drawing_circle", detector=None):
        super().__init__(name=name, detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[CargoItem]:
        print(f"[ED] detect() called, ed={self.detector.ed is not None}, method={self.detector.detect_method}")
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

        center, radius = self._detect_circle(sub_img, sub_hsv, ts, target_color)
        if center is None:
            ts.roi_miss_count += 1
            return None

        center = (center[0] + offset[0], center[1] + offset[1])
        return self._finalize(center, radius, ts, target_color, scale)

    def _try_global(self, frame: np.ndarray, hsv: np.ndarray,
                    ts, target_color: Color, scale: float) -> Optional[CargoItem]:
        center, radius = self._detect_circle(frame, hsv, ts, target_color)
        if center is None:
            return None

        return self._finalize(center, radius, ts, target_color, scale)

    def _detect_circle(self, bgr: np.ndarray, hsv: np.ndarray,
                       ts, target_color: Color) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        print(f"[ED] _detect_circle() called, ed={self.detector.ed is not None}")
        if self.detector.ed is None:
            print("[ED] ed is None — returning early!")
            return None, None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        blur_k = self.detector.blur_kernel
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), self.detector.blur_sigma)

        try:
            self.detector.ed.detectEdges(blurred)
            edges = self.detector.ed.getEdgeImage()
            if edges is None or cv2.countNonZero(edges) < 20:
                print("EdgeDrawing: No edges detected or too few edges.")
                return None, None
        except cv2.error:
            print("EdgeDrawing: Error occurred while detecting edges.")
            return None, None

        morph_k = self.detector.edge_morph_kernel
        if morph_k % 2 == 0:
            morph_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        morphed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel,
            iterations=self.detector.edge_morph_iterations
        )

        self.detector._last_edge_preview = morphed.copy()

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("EdgeDrawing: No contours detected.")
            return None, None

        color_mask = self._build_color_mask(hsv, ts, target_color)
        if color_mask is None:
            return None, None

        candidates = []
        for contour in contours:
            candidate = self._evaluate_contour(contour, color_mask, hsv, ts)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None, None

        best = self._select_best_candidate(candidates)
        return best["center"], best["radius"]

    SV_PERCENTILE = 5
    EMA_ALPHA = 0.3
    COARSE_MIN_PIXELS = 50

    def _build_color_mask(self, hsv: np.ndarray, ts,
                          target_color: Color) -> Optional[np.ndarray]:
        print(f"[ED] _build_color_mask() called, target={target_color}")
        if target_color not in self.detector.color_ranges:
            print(f"EdgeDrawing: Color {target_color} is not in the supported color ranges.")
            return None
        raw_ranges = self.detector.color_ranges[target_color]
        if not raw_ranges:
            print(f"EdgeDrawing: Empty color_ranges for {target_color}.")
            return None

        # Step 1 — coarse mask: H only, full S/V
        mask_coarse = None
        for lower, upper in raw_ranges:
            coarse_lower = np.array([lower[0], 0, 0], dtype=np.uint8)
            coarse_upper = np.array([upper[0], 255, 255], dtype=np.uint8)
            chunk = cv2.inRange(hsv, coarse_lower, coarse_upper)
            mask_coarse = chunk if mask_coarse is None else cv2.bitwise_or(mask_coarse, chunk)

        if mask_coarse is None or cv2.countNonZero(mask_coarse) < self.COARSE_MIN_PIXELS:
            # 粗 mask 太少像素 → 回退原始硬编码范围
            mask = None
            for lower, upper in raw_ranges:
                chunk = cv2.inRange(hsv, lower, upper)
                mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)
            return mask

        # Step 2 — compute adaptive S/V lower bounds (histogram O(n), no sort)
        s_ch = hsv[:, :, 1][mask_coarse > 0]
        v_ch = hsv[:, :, 2][mask_coarse > 0]
        _t0 = time.perf_counter()
        s_bins = np.bincount(s_ch.astype(np.int32), minlength=256)
        v_bins = np.bincount(v_ch.astype(np.int32), minlength=256)
        s_cum = np.cumsum(s_bins, dtype=np.float64)
        v_cum = np.cumsum(v_bins, dtype=np.float64)
        s_low = int(np.searchsorted(s_cum, s_cum[-1] * 0.05))
        v_low = int(np.searchsorted(v_cum, v_cum[-1] * 0.05))
        _t1 = time.perf_counter()
        if _t1 - _t0 > 0.01:
            print(f"[ED] bincount percentile took {_t1-_t0:.3f}s, pixels={len(s_ch)}")

        # Step 3 — EMA smoothing across frames
        if ts._ema_s is None:
            ts._ema_s = s_low
            ts._ema_v = v_low
        else:
            ts._ema_s = int(self.EMA_ALPHA * s_low + (1 - self.EMA_ALPHA) * ts._ema_s)
            ts._ema_v = int(self.EMA_ALPHA * v_low + (1 - self.EMA_ALPHA) * ts._ema_v)

        # Step 4 — fine mask with adjusted S/V
        mask = None
        for lower, upper in raw_ranges:
            adj_lower = np.array([
                lower[0],
                max(int(lower[1]), ts._ema_s),
                max(int(lower[2]), ts._ema_v),
            ], dtype=np.uint8)
            chunk = cv2.inRange(hsv, adj_lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)

        return mask

    def _evaluate_contour(self, contour: np.ndarray, color_mask: np.ndarray,
                          hsv: np.ndarray, ts) -> Optional[dict]:
        area = cv2.contourArea(contour)
        if area < self.detector.min_area:
            return None

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return None

        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < self.detector.min_circularity:
            return None

        if len(contour) < 5:
            return None

        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            return None

        center = ellipse[0]
        axes = ellipse[1]
        if axes[0] == 0 or axes[1] == 0:
            return None

        axis_ratio = max(axes) / min(axes)
        if axis_ratio > self.MAX_AXIS_RATIO:
            return None

        radius = (axes[0] + axes[1]) / 4.0

        refined_center = self._refine_center_with_color_moments(
            center, axes, ellipse[2], color_mask
        )
        if refined_center is None:
            print(
                f"EdgeDrawing: Failed to refine center with color moments.\n"
                f"  contour: area={area:.1f} peri={peri:.1f} "
                f"circularity={circularity:.4f} axis_ratio={axis_ratio:.3f}\n"
                f"  ellipse: center=({center[0]:.1f},{center[1]:.1f}) "
                f"axes=({axes[0]:.1f},{axes[1]:.1f}) angle={ellipse[2]:.1f}"
            )
            refined_center = center

        color_score = self._compute_color_score(
            refined_center, radius, color_mask
        )
        _base_threshold = self.detector.color_match_threshold
        _ema_v = ts._ema_v if ts._ema_v is not None else 80
        _effective_threshold = max(0.15, min(_base_threshold, _ema_v / 80.0 * _base_threshold))
        if color_score < _effective_threshold:
            _h, _w = color_mask.shape[:2]
            _x, _y = int(refined_center[0]), int(refined_center[1])
            _r = int(radius)
            _x1 = max(_x - _r, 0)
            _y1 = max(_y - _r, 0)
            _x2 = min(_x + _r, _w)
            _y2 = min(_y + _r, _h)
            _roi = color_mask[_y1:_y2, _x1:_x2]
            _roi_r = min(_r, _roi.shape[1] // 2, _roi.shape[0] // 2) if _roi.size > 0 else 0
            _matched = 0
            _total_px = 0.0
            if _roi_r > 0:
                _cm = np.zeros_like(_roi)
                cv2.circle(_cm, (_roi.shape[1] // 2, _roi.shape[0] // 2), _roi_r, 255, -1)
                _matched = cv2.countNonZero(cv2.bitwise_and(_roi, _roi, mask=_cm))
                _total_px = np.pi * float(_roi_r) * float(_roi_r)
            print(
                f"EdgeDrawing: Color score is below the threshold.\n"
                f"  score={color_score:.3f} < threshold={_effective_threshold:.3f} "
                f"(base={_base_threshold:.3f})  matched={_matched}/{_total_px:.0f} "
                f"({_matched / _total_px * 100:.1f}%)\n"
                f"  light: ema_s={ts._ema_s} ema_v={_ema_v}\n"
                f"  center=({refined_center[0]:.1f},{refined_center[1]:.1f}) "
                f"radius={radius:.1f}\n"
                f"  contour: area={area:.1f} peri={peri:.1f} "
                f"circularity={circularity:.4f} axis_ratio={axis_ratio:.3f} "
                f"ellipse_axes=({axes[0]:.1f},{axes[1]:.1f})"
            )
            return None

        return {
            "center": refined_center,
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "color_score": color_score,
            "axis_ratio": axis_ratio,
        }

    def _refine_center_with_color_moments(self, center: Tuple[float, float],
                                          axes: Tuple[float, float],
                                          angle: float,
                                          color_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        h, w = color_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (int(center[0]), int(center[1])),
                    (int(axes[0] / 2), int(axes[1] / 2)), angle, 0, 360, 255, -1)

        obj_mask = cv2.bitwise_and(color_mask, mask)
        M = cv2.moments(obj_mask)
        if M["m00"] <= 0:
            ellipse_px = cv2.countNonZero(mask)
            masked_px = cv2.countNonZero(obj_mask)
            print(
                f"EdgeDrawing: _refine_center_with_color_moments failed.\n"
                f"  ellipse: center=({center[0]:.1f},{center[1]:.1f}) "
                f"axes=({axes[0]:.1f},{axes[1]:.1f}) angle={angle:.1f}\n"
                f"  mask: ellipse_area={ellipse_px}px "
                f"color_intersection={masked_px}px m00={M['m00']:.0f}\n"
                f"  cause: no color mask pixels inside the fitted ellipse"
            )
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)

    def _compute_color_score(self, center: Tuple[float, float], radius: float,
                             color_mask: np.ndarray) -> float:
        h, w = color_mask.shape[:2]
        x, y = int(center[0]), int(center[1])
        r = int(radius)

        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, w)
        y2 = min(y + r, h)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = color_mask[y1:y2, x1:x2]
        circle_mask = np.zeros_like(roi)
        roi_r = min(r, roi.shape[1] // 2, roi.shape[0] // 2)
        if roi_r <= 0:
            return 0.0
        cv2.circle(circle_mask, (roi.shape[1] // 2, roi.shape[0] // 2), roi_r, 255, -1)
        masked_roi = cv2.bitwise_and(roi, roi, mask=circle_mask)
        matched = cv2.countNonZero(masked_roi)
        total = np.pi * float(roi_r) * float(roi_r)
        return float(matched) / total if total > 0 else 0.0

    def _select_best_candidate(self, candidates: List[dict]) -> dict:
        max_area = max(c["area"] for c in candidates)

        best = None
        best_score = -1.0
        for c in candidates:
            area_norm = c["area"] / max_area if max_area > 0 else 0.0
            score = (
                0.5 * c["color_score"]
                + 0.3 * c["circularity"]
                + 0.2 * area_norm
            )
            if score > best_score:
                best_score = score
                best = c

        return best

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
        return self._build_cargo_item(final_center, target_color)

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
            return self._build_cargo_item(center, target_color)

        if len(ts._center_history) > 0:
            avg = (
                sum(c[0] for c in ts._center_history) / len(ts._center_history),
                sum(c[1] for c in ts._center_history) / len(ts._center_history),
            )
            return self._build_cargo_item(avg, target_color)

        return None

    def _build_cargo_item(self, center: Tuple[float, float], target_color: Color) -> CargoItem:
        from ..target_creation import create_cargo_item
        return create_cargo_item(center, target_color)
