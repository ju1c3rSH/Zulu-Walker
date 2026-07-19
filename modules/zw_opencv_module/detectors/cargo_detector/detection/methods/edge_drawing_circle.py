import cv2
import numpy as np
from typing import Optional, Tuple, List

from .base import BaseDetectionMethod
from .....models.color import Color
from .....models.cargo import CargoItem
from utils.log_util import log_print



class EdgeDrawingCircleMethod(BaseDetectionMethod):
    TARGET_W = 640
    TARGET_H = 480

    def __init__(self, name: str = "edge_drawing_circle", detector=None):
        super().__init__(name=name, detector=detector)

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[CargoItem]:
        ts = self.detector._get_tracking(target_color)
        small, scale = self._scale_frame(frame)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        fs = self.detector.force_stage if hasattr(self.detector, 'force_stage') else 0
        if fs == 0 or fs == 1:
            result = self._try_roi(small, hsv, ts, target_color, scale)
            if result is not None and fs == 1:
                return result
        else:
            result = None
        if result is None and (fs == 0 or 2 <= fs <= 4):
            result = self._try_global(small, hsv, ts, target_color, scale)
            if result is not None and (2 <= fs <= 4):
                return result
        if result is None and (fs == 0 or fs == 5):
            result = self._fallback_predict(ts, target_color, scale)
        return result

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

        center, radius, conf = self._detect_circle(sub_img, sub_hsv, target_color)
        if center is None:
            ts.roi_miss_count += 1
            return None

        center = (center[0] + offset[0], center[1] + offset[1])
        return self._finalize(center, radius, conf, ts, target_color, scale)

    def _try_global(self, frame: np.ndarray, hsv: np.ndarray,
                    ts, target_color: Color, scale: float) -> Optional[CargoItem]:
        center, radius, conf = self._detect_circle(frame, hsv, target_color)
        if center is None:
            return None

        return self._finalize(center, radius, conf, ts, target_color, scale)

    def _run_edge_pipeline(self, bgr: np.ndarray, hsv: np.ndarray,
                           target_color: Color
                           ) -> Optional[Tuple[List, np.ndarray]]:
        """Edge detection -> morph -> contours -> color_mask.
        Returns (contours, color_mask) or None on failure.
        """
        if self.detector.ed is None:
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self.detector._clahe.apply(gray)

        blur_k = self.detector.blur_kernel
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), self.detector.blur_sigma)

        try:
            self.detector.ed.detectEdges(blurred)
            edges = self.detector.ed.getEdgeImage()
            if edges is None or cv2.countNonZero(edges) < self.detector.edge_min_pixels:
                log_print("EdgeDrawing: No edges detected or too few edges.")
                return None
        except cv2.error:
            log_print("EdgeDrawing: Error occurred while detecting edges.")
            return None

        morph_k = self.detector.edge_morph_kernel
        if morph_k % 2 == 0:
            morph_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        morphed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel,
            iterations=self.detector.edge_morph_iterations
        )

        self.detector._last_edge_preview = morphed.copy()

        color_mask = self._build_color_mask(hsv, target_color)
        if color_mask is not None:
            self.detector._last_mask = color_mask
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.detector._last_morphed = cv2.morphologyEx(
                color_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=2)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log_print("EdgeDrawing: No contours detected.")
            return None

        if color_mask is None:
            return None

        return contours, color_mask

    def _detect_circle(self, bgr: np.ndarray, hsv: np.ndarray,
                       target_color: Color
                       ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float]]:
        result = self._run_edge_pipeline(bgr, hsv, target_color)
        if result is None:
            return None, None, None

        contours, color_mask = result

        candidates = []
        for contour in contours:
            candidate = self._evaluate_contour(contour, color_mask, hsv, target_color)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            if not hasattr(self.detector, '_ed_log_frame'):
                self.detector._ed_log_frame = 0
            self.detector._ed_log_frame += 1
            if self.detector._ed_log_frame % 60 == 0:
                mask_px = cv2.countNonZero(color_mask)
                log_print(f"[EdgeDraw] target={target_color.name} color_mask={mask_px}px "
                      f"contours={len(contours)} candidates=0")
            return None, None, None

        best = self._select_best_candidate(candidates)
        if not hasattr(self.detector, '_ed_log_frame'):
            self.detector._ed_log_frame = 0
        self.detector._ed_log_frame += 1
        if self.detector._ed_log_frame % 60 == 0:
            mask_px = cv2.countNonZero(color_mask)
            log_print(f"[EdgeDraw] target={target_color.name} color_mask={mask_px}px "
                  f"contours={len(contours)} candidates={len(candidates)} "
                  f"best: area={best['area']:.1f} circularity={best['circularity']:.3f} "
                  f"color_score={best['color_score']:.3f}")
        return best["center"], best["radius"], 100.0

    def _build_color_mask(self, hsv: np.ndarray,
                          target_color: Color) -> Optional[np.ndarray]:
        if target_color not in self.detector.color_ranges:
            log_print(f"EdgeDrawing: Color {target_color} is not in the supported color ranges.")
            return None
        raw_ranges = self.detector.color_ranges[target_color]
        if not raw_ranges:
            return None

        # Pass 1 — use the original hardcoded ranges (same as original code)
        mask = None
        for lower, upper in raw_ranges:
            chunk = cv2.inRange(hsv, lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)

        if mask is not None and cv2.countNonZero(mask) >= self.detector.low_light_min_pixels:
            return mask

        # Pass 2 — mask too sparse (low light), relax S/V
        relaxed = None
        for lower, upper in raw_ranges:
            rel_lower = np.array([
                lower[0],
                max(int(lower[1]) // self.detector.low_light_s_divider, self.detector.relaxed_s),
                max(int(lower[2]) // self.detector.low_light_v_divider, self.detector.relaxed_v),
            ])
            chunk = cv2.inRange(hsv, rel_lower, upper)
            relaxed = chunk if relaxed is None else cv2.bitwise_or(relaxed, chunk)

        if relaxed is not None and cv2.countNonZero(relaxed) > 0:
            log_print(
                f"EdgeDrawing: Low-light fallback activated. "
                f"original={cv2.countNonZero(mask) if mask is not None else 0}px, "
                f"relaxed={cv2.countNonZero(relaxed)}px"
            )
            return relaxed

        return mask

    def _evaluate_contour(self, contour: np.ndarray, color_mask: np.ndarray,
                          hsv: np.ndarray, target_color: Color = None) -> Optional[dict]:
        area = cv2.contourArea(contour)
        if area < self.detector.min_area:
            return None

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return None

        circularity = 4.0 * np.pi * area / (peri * peri)

        if len(contour) < self.detector.ellipse_min_contour_points:
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
        if axis_ratio > self.detector.ellipse_max_axis_ratio:
            return None

        radius = (axes[0] + axes[1]) / 4.0

        color_score = self._compute_color_score(
            center, radius, color_mask
        )
        thresh = self.detector.color_match_threshold
        if target_color is not None:
            thresh = self.detector.color_match_threshold_per_color.get(target_color, thresh)
        if color_score < thresh:
            _h, _w = color_mask.shape[:2]
            _x, _y = int(center[0]), int(center[1])
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
            log_print(
                f"EdgeDrawing: Color score is below the threshold.\n"
                f"  score={color_score:.3f} < threshold={thresh:.3f}  "
                f"matched={_matched}/{_total_px:.0f} ({_matched / _total_px * 100:.1f}%)\n"
                f"  center=({center[0]:.1f},{center[1]:.1f}) "
                f"radius={radius:.1f}\n"
                f"  contour: area={area:.1f} peri={peri:.1f} "
                f"circularity={circularity:.4f} axis_ratio={axis_ratio:.3f} "
                f"ellipse_axes=({axes[0]:.1f},{axes[1]:.1f})"
            )
            return None

        return {
            "center": center,
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
            log_print(
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
                self.detector.score_weight_color * c["color_score"]
                + self.detector.score_weight_circularity * c["circularity"]
                + self.detector.score_weight_area * area_norm
            )
            if score > best_score:
                best_score = score
                best = c

        return best

    def _finalize(self, center: Tuple[float, float], radius: float,
                  confidence: float, ts, target_color: Color,
                  scale: float) -> Optional[CargoItem]:
        if scale < 1.0:
            center = (center[0] / scale, center[1] / scale)
            radius = radius / scale

        ts.last_center = center
        ts.roi_miss_count = 0
        ts._center_history.append(center)
        ts._radius_history.append(radius)

        self._store_cargo_meta(target_color, center, radius,
                               confidence=confidence)
        return self._build_cargo_item(center, target_color, radius,
                                      confidence=confidence)

    def _fallback_predict(self, ts, target_color: Color,
                          scale: float) -> Optional[CargoItem]:
        if len(ts._center_history) > 0:
            avg = (
                sum(c[0] for c in ts._center_history) / len(ts._center_history),
                sum(c[1] for c in ts._center_history) / len(ts._center_history),
            )
            return self._build_cargo_item(avg, target_color, is_predicted=True,
                                          confidence=0.0)
        return None

    def _build_cargo_item(self, center: Tuple[float, float], target_color: Color,
                          radius: Optional[float] = None,
                          is_predicted: bool = False,
                          confidence: float = 100.0) -> CargoItem:
        from ..target_creation import create_cargo_item
        return create_cargo_item(center, target_color, radius=radius,
                                 is_predicted=is_predicted, confidence=confidence)
