import cv2
import numpy as np

from .base import DetectionMethod


class TestLineQuadMethod(DetectionMethod):
    def detect(self, frame: np.ndarray, target_color: str) -> None:
        import time

        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        kernel_size = self.detector.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.detector.blur_sigma)

        result = None

        try:
            from ..line_geometry import extract_lines_ed_lsd, find_quad_from_lines
            from ..quad_geometry import compute_quad_center, validate_center_alignment

            lines = extract_lines_ed_lsd(
                self.detector.ed, self.detector.lsd, blurred
            )

            if lines is not None and len(lines) >= 4 and len(lines) <= 200:
                quad = find_quad_from_lines(
                    lines, small.shape,
                    order_quad_points_fn=self.detector._order_quad_points,
                    min_area_threshold_quad=self.detector.min_area_threshold_quad,
                )

                if quad is not None:
                    circle_result = self.detector._detect_circle_in_quad(blurred, quad)

                    if circle_result is not None:
                        circle_center, radius = circle_result
                        quad_center = compute_quad_center(quad)
                        max_offset = 20.0 * scale
                        if validate_center_alignment(quad_center, circle_center, max_offset):
                            result = (circle_center, radius, quad)

            if result is None:
                fallback_result = self.detector._fallback_ellipse_detection(
                    small, gray, scale, hsv
                )
                if fallback_result is not None:
                    ellipse, contour, quad, morphed, quad_center = fallback_result
                    if ellipse is not None:
                        center = ellipse[0]
                        radius = max(ellipse[1]) / 2
                        result = (center, radius, quad)

        except Exception as e:
            print(f"[TEST_LINE_QUAD] Error: {e}")
            import traceback
            traceback.print_exc()
            try:
                fallback_result = self.detector._fallback_ellipse_detection(
                    small, gray, scale, hsv
                )
                if fallback_result is not None:
                    ellipse, contour, quad, morphed, quad_center = fallback_result
                    if ellipse is not None:
                        center = ellipse[0]
                        radius = max(ellipse[1]) / 2
                        result = (center, radius, quad)
            except Exception as e2:
                print(f"[TEST_LINE_QUAD] Fallback error: {e2}")

        final_center = None
        final_radius = None

        if result is not None:
            center, radius, quad = result
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

        if final_center is not None and final_radius is not None:
            color_name = target_color if target_color else "Red"
            target_item = self.detector._create_target_item_from_center(
                final_center, final_radius, scale, color_name
            )
            self.detector.circle_target.add_target(target_item)

        try:
            self.detector.ed.detectEdges(gray)
            edge_img = self.detector.ed.getEdgeImage()
            if edge_img is not None:
                self.detector._last_canny_preview = edge_img
            else:
                self.detector._last_canny_preview = np.zeros(
                    (small.shape[0], small.shape[1]), dtype=np.uint8
                )
        except Exception:
            self.detector._last_canny_preview = np.zeros(
                (small.shape[0], small.shape[1]), dtype=np.uint8
            )
