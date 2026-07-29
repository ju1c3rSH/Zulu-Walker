import cv2
import numpy as np

from .base import DetectionMethod


class EdgeContourEllipseMethod(DetectionMethod):
    def detect(self, frame: np.ndarray, target_color: str) -> None:
        import time

        h, w = frame.shape[:2]
        target_w, target_h = 640, 480
        scale = min(target_h / h, target_w / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        result = None
        morphed_edges = None
        detected_quad = None
        detected_quad_center = None

        fallback_result = self.detector._fallback_ellipse_detection(small, gray, scale, hsv)

        if fallback_result is not None:
            ellipse, contour, quad, morphed_edges, quad_center = fallback_result
            detected_quad = quad
            detected_quad_center = quad_center
            if ellipse is not None:
                center = ellipse[0]
                radius = max(ellipse[1]) / 2
                result = (center, radius, quad, quad_center)

        final_center = None
        final_radius = None
        final_quad = detected_quad
        final_quad_center = detected_quad_center

        if result is not None:
            center, radius, quad, quad_center = result
            final_radius = radius
            smoothed_center = self.detector._kalman_update(center)
            if smoothed_center is not None:
                final_center = smoothed_center
            else:
                final_center = center
        else:
            predicted = self.detector._kalman_update(None)
            if predicted is not None:
                final_center = predicted
                final_radius = None
            elif detected_quad_center is not None:
                final_center = detected_quad_center
                final_radius = None

        if final_center is not None:
            color_name = target_color if target_color else "Red"
            if final_radius is not None:
                target_item = self.detector._create_target_item_from_center(
                    final_center, final_radius, scale, color_name, final_quad
                )
            else:
                target_item = self.detector._create_target_item_from_center(
                    final_center, 20.0, scale, color_name, final_quad
                )
            self.detector.circle_target.add_target(target_item)

        if morphed_edges is not None:
            self.detector._last_canny_preview = morphed_edges
        else:
            try:
                if self.detector.ed is not None:
                    self.detector.ed.detectEdges(gray)
                    edges = self.detector.ed.getEdgeImage()
                    if edges is not None:
                        self.detector._last_canny_preview = self.detector._apply_morphology(edges)
                    else:
                        self.detector._last_canny_preview = np.zeros(
                            (small.shape[0], small.shape[1]), dtype=np.uint8
                        )
                else:
                    self.detector._last_canny_preview = np.zeros(
                        (small.shape[0], small.shape[1]), dtype=np.uint8
                    )
            except Exception:
                self.detector._last_canny_preview = np.zeros(
                    (small.shape[0], small.shape[1]), dtype=np.uint8
                )
