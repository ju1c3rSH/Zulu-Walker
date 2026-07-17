import cv2
import numpy as np
from typing import Optional, Tuple

from .edge_drawing_circle import EdgeDrawingCircleMethod
from .....models.color import Color



class HeuristicEdgeCircleMethod(EdgeDrawingCircleMethod):
    """Two-stage cargo circle detection using EdgeDrawing + Heuristic fallback.

    Stage 1 (confidence=100):
        EdgeDrawing -> contours -> fitEllipse + color moments + color score.
        Same as EdgeDrawingCircleMethod.

    Stage 2 (confidence=60):
        When Stage 1 yields no valid candidate:
        - Moments centroid from existing contours
        - Relaxed color threshold
        - Handles partial/fragmented circles

    Fallback:
        Kalman prediction or history average (is_predicted=True, confidence=0)
    """

    def __init__(self, name: str = "heuristic_edge", detector=None):
        super().__init__(name=name, detector=detector)

    def _detect_circle(self, bgr: np.ndarray, hsv: np.ndarray,
                       target_color: Color
                       ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float]]:
        result = self._run_edge_pipeline(bgr, hsv, target_color)
        if result is None:
            return None, None, None

        contours, color_mask = result

        # === Stage 1: full ellipse evaluation ===
        candidates = []
        for contour in contours:
            candidate = self._evaluate_contour(contour, color_mask, hsv)
            if candidate is not None:
                candidates.append(candidate)

        if candidates:
            best = self._select_best_candidate(candidates)
            return best["center"], best["radius"], 100.0

        # === Stage 2: heuristic moments fallback ===
        stage2_candidates = []
        for contour in contours:
            candidate = self._evaluate_heuristic(contour, color_mask)
            if candidate is not None:
                stage2_candidates.append(candidate)

        if stage2_candidates:
            best = max(stage2_candidates, key=lambda c: c["area"])
            return best["center"], best["radius"], 60.0

        return None, None, None

    def _evaluate_heuristic(self, contour: np.ndarray,
                            color_mask: np.ndarray) -> Optional[dict]:
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
        if color_score < self.detector.stage2_color_threshold:
            return None

        return {
            "center": (cx, cy),
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "color_score": color_score,
        }
