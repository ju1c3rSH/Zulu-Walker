import cv2
import numpy as np

from .base import DetectionMethod


class EdgeDrawingQuadsMethod(DetectionMethod):
    def detect(self, frame: np.ndarray, target_color: str) -> None:
        if self.detector.ed is None:
            return

        self.detector.is_uv_spot_detected = False
        self.detector.uv_spot_center = None

        h, w = frame.shape[:2]
        target_w, target_h = 640, 480
        scale = min(target_h / h, target_w / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) if self.detector.enable_color_filter else None
        kernel_size = self.detector.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.detector.blur_sigma)

        try:
            self.detector.ed.detectEdges(blurred)
            edges = self.detector.ed.getEdgeImage()
            if edges is None or cv2.countNonZero(edges) < 100:
                self.detector._last_canny_preview = np.zeros(
                    (small.shape[0], small.shape[1]), dtype=np.uint8
                )
                return
        except cv2.error:
            self.detector._last_canny_preview = np.zeros(
                (small.shape[0], small.shape[1]), dtype=np.uint8
            )
            return

        morphed = self.detector._apply_morphology(edges)
        self.detector._last_canny_preview = morphed

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        from ..quad_geometry import find_quadrilaterals_from_contours, get_quad_center_perspective
        from ..target_creation import create_quad_target_item
        from ..uv_detection import detect_uv_spot_with_search_contour

        quadrilaterals = find_quadrilaterals_from_contours(
            contours, scale,
            min_area_threshold_quad=self.detector.min_area_threshold_quad,
            quad_aspect_ratio=self.detector.quad_aspect_ratio,
            enable_color_filter=self.detector.enable_color_filter,
            hsv=hsv,
            detect_color_func=self.detector._detect_contour_color,
        )
        if not quadrilaterals:
            return

        sorted_quads = sorted(quadrilaterals, key=lambda x: x[0], reverse=True)[:3]

        best_quad = None
        best_quad_center = None
        best_area = 0
        has_last_center = self.detector.last_best_quad_center is not None

        for area, quad in sorted_quads:
            quad_center = get_quad_center_perspective(quad, is_ordered=True)
            if quad_center is None:
                continue

            if has_last_center:
                dist = np.linalg.norm(
                    np.array(quad_center) - np.array(self.detector.last_best_quad_center)
                )
                if dist > 150:
                    continue
                if dist < 30:
                    best_quad = quad
                    best_quad_center = quad_center
                    best_area = area
                    break
                score = area * (1.0 - dist / 200.0)
                if score > best_area:
                    best_area = score
                    best_quad = quad
                    best_quad_center = quad_center
            else:
                if area > best_area:
                    best_area = area
                    best_quad = quad
                    best_quad_center = quad_center

        if best_quad is not None:
            self.detector.last_best_quad_center = best_quad_center
            self.detector.last_best_quad = best_quad

        if best_quad is None or best_quad_center is None:
            self.detector.is_detected_quad = False
            self.detector.lost_frames += 1
            if self.detector.lost_frames > self.detector.max_lost_frames:
                self.detector.last_best_quad_center = None
                self.detector.last_best_quad = None
                self.detector.tracking_initialized = False
            return
        else:
            self.detector.lost_frames = 0

        self.detector.is_detected_quad = True

        uv_config = {
            "color_ranges": self.detector.color_ranges,
            "uv_min_area": self.detector.uv_min_area,
            "uv_adaptive_enabled": self.detector.uv_adaptive_enabled,
            "uv_s_gate": self.detector.uv_s_gate,
            "uv_s_min": self.detector.uv_s_min,
            "uv_v_floor": self.detector.uv_v_floor,
            "uv_v_percentile": self.detector.uv_v_percentile,
            "uv_h_range": self.detector.uv_h_range,
            "uv_contrast_dilate": self.detector.uv_contrast_dilate,
            "uv_contrast_ratio_min": self.detector.uv_contrast_ratio_min,
            "ema_alpha": 0.3,
            "uv_miss_threshold": 10,
        }

        uv_center, self.detector._ema_v_min, self.detector._uv_miss_counter = (
            detect_uv_spot_with_search_contour(
                small, uv_config,
                search_contour=best_quad,
                hsv=hsv,
                gray=gray,
                ema_v_min=self.detector._ema_v_min,
                uv_miss_counter=self.detector._uv_miss_counter,
            )
        )

        if uv_center is not None:
            uv_center_orig = (int(uv_center[0] / scale), int(uv_center[1] / scale))
            smoothed = self.detector._uv_kalman_update(uv_center_orig)
            if smoothed is not None:
                self.detector.uv_spot_center = smoothed
                self.detector.is_uv_spot_detected = True
        else:
            self.detector._uv_kalman_update(None)
            self.detector.is_uv_spot_detected = False
            self.detector.uv_spot_center = None

        smoothed_center = self.detector._kalman_update(best_quad_center)
        if smoothed_center is not None:
            final_center = smoothed_center
        else:
            final_center = best_quad_center

        color_name = target_color if target_color else "Red"
        target_item = create_quad_target_item(final_center, best_quad, scale, color_name)
        self.detector.circle_target.add_target(target_item)
