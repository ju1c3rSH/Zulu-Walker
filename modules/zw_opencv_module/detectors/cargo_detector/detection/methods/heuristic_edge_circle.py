import cv2
import numpy as np
from typing import Optional, Tuple

from .edge_drawing_circle import EdgeDrawingCircleMethod
from .....models.color import Color



class HeuristicEdgeCircleMethod(EdgeDrawingCircleMethod):
    """Three-stage cargo circle detection.

    Stage 1 (confidence=100):  fitEllipse from EdgeDrawing contours
        - EdgeDrawing -> contours -> fitEllipse (center used directly, no color-moments refinement)
        - Color score verified but NOT used to shift center

    Stage 2 (confidence=60):  color blob centroid
        - findContours directly on raw color_mask
        - Moments centroid from largest color blob
        - Pure area-based, no edge dependency

    Stage 3 (confidence=40):  heuristic moments from edge contours
        - Moments centroid from existing edge contours
        - Relaxed color threshold
        - Handles partial/fragmented circles

    Fallback:
        Kalman prediction or history average (is_predicted=True, confidence=0)
    """

    COLOR_BLOB_MIN_AREA = 2000
    STAGE1_CONFIDENCE = 100.0
    STAGE2_CONFIDENCE = 60.0
    STAGE3_CONFIDENCE = 40.0

    def __init__(self, name: str = "heuristic_edge", detector=None):
        super().__init__(name=name, detector=detector)

    def _detect_circle(self, bgr: np.ndarray, hsv: np.ndarray,
                       target_color: Color
                       ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float]]:
        fs = self.detector.force_stage if hasattr(self.detector, 'force_stage') else 0
        if fs == 3:
            return self._try_color_blob_from_hsv(hsv, target_color)

        result = self._run_edge_pipeline(bgr, hsv, target_color)
        if result is None:
            return self._try_color_blob_from_hsv(hsv, target_color)

        contours, color_mask = result

        # === Stage 1: fitEllipse evaluation ===
        candidates = []
        for contour in contours:
            candidate = self._evaluate_contour(contour, color_mask, hsv, target_color)
            if candidate is not None:
                candidates.append(candidate)

        if candidates:
            best = self._select_best_candidate(candidates)
            return best["center"], best["radius"], self.STAGE1_CONFIDENCE

        # === Stage 2: color blob centroid ===
        blob_result = self._try_color_blob(color_mask)
        if blob_result is not None:
            return blob_result[0], blob_result[1], self.STAGE2_CONFIDENCE

        # === Stage 3: heuristic moments from edge contours ===
        stage3_candidates = []
        for contour in contours:
            candidate = self._evaluate_heuristic(contour, color_mask, target_color)
            if candidate is not None:
                stage3_candidates.append(candidate)

        if stage3_candidates:
            best = max(stage3_candidates, key=lambda c: c["area"])
            return best["center"], best["radius"], self.STAGE3_CONFIDENCE

        return None, None, None

    def _try_color_blob(self, color_mask: np.ndarray
                        ) -> Optional[Tuple]:
        blob_contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not blob_contours:
            return None
        best = max(blob_contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.COLOR_BLOB_MIN_AREA:
            return None
        M = cv2.moments(best)
        if M["m00"] <= 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        radius = np.sqrt(area / np.pi)
        return (cx, cy), radius, self.STAGE2_CONFIDENCE

    def _try_color_blob_from_hsv(self, hsv: np.ndarray, target_color: Color
                                 ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float]]:
        color_mask = self._build_color_mask(hsv, target_color)
        if color_mask is None:
            return None, None, None
        result = self._try_color_blob(color_mask)
        if result is not None:
            return result
        return None, None, None

    def _evaluate_heuristic(self, contour: np.ndarray,
                            color_mask: np.ndarray, target_color: Color = None) -> Optional[dict]:
        area = cv2.contourArea(contour)
        if area < self.detector.min_area * self.detector.stage2_min_area_ratio:
            return None

        peri = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0

        M = cv2.moments(contour)
        if M["m00"] <= 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        radius = np.sqrt(area / np.pi)

        color_score = self._compute_color_score((cx, cy), radius, color_mask)
        thresh = self.detector.stage2_color_threshold
        if target_color is not None:
            thresh = self.detector.stage2_color_threshold_per_color.get(target_color, thresh)
        if color_score < thresh:
            return None

        return {
            "center": (cx, cy),
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "color_score": color_score,
        }
