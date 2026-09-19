from typing import Optional, Tuple, List
import time

import cv2
import numpy as np

try:
    cv2.ximgproc.createEdgeDrawing
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False

from ...models.circle import CircleTargets, CircleTargetItem, ShapeType
from .detection import DetectMethod
from .detection.methods import (
    DetectionMethod,
    ContourEllipseMethod,
    EdgeContourEllipseMethod,
    EdgeDrawingQuadsMethod,
    TestLineQuadMethod,
)
from .detection.quad_geometry import (
    fit_ellipse_in_quad,
    find_quadrilaterals_from_contours,
    get_quad_center_perspective,
    order_quad_points,
    compute_quad_center,
    is_center_aligned,
)


class CircleTargetDetector:
    def __init__(self, name: str = "circle_target"):
        self.name = name

        self.ed = None
        self.lsd = None
        if _HAS_XIMGPROC:
            self.ed = cv2.ximgproc.createEdgeDrawing()
            ed_params = self.ed.Params()
            ed_params.MinPathLength = 50
            ed_params.GradientThresholdValue = 20
            ed_params.NFAValidation = True
            self.ed.setParams(ed_params)
            self.lsd = cv2.createLineSegmentDetector(0)

        self.ed_min_path_length = 50
        self.ed_gradient_threshold = 20
        self.ed_nfa_validation = True

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf_q_base = 0.3
        self._kf_q_vel_base = 0.3

        self.tracking_initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self._last_predict_time = None

        self.color_ranges = {
            "Red": [
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            "Green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            "UV": [
                (np.array([120, 10, 80]), np.array([160, 255, 255])),
                (np.array([90, 5, 90]), np.array([130, 220, 255])),
            ],
        }
        self.circle_target = CircleTargets()
        self.min_area_threshold_quad = 1680
        self.min_area_threshold_ellipse = 100
        self.min_contour_points = 15

        self.max_aspect_ratio = 2.0
        self.min_circularity = 0.4

        self.detect_method = DetectMethod.EDGE_DRAWING_QUADS

        self.quad_participation = False

        self.ed_min_path_length = 164
        self.ed_gradient_threshold = 90
        self.ed_nfa_validation = True
        self._update_ed_params()

        self.morph_type = 4
        self.morph_kernel = 3
        self.morph_iterations = 1
        self._morph_kernel_cache = None

        self.blur_kernel = 5
        self.blur_sigma = 38.0

        self._last_canny_preview: Optional[np.ndarray] = None

        self.color_h_ranges = {
            "Red": (0, 15, 165, 180),
            "Green": (40, 80, 0, 0),
            "Blue": (100, 130, 0, 0),
        }
        self.color_s_min = 60
        self.color_v_min = 50
        self.debug_color = False

        self.quad_aspect_ratio = 1.51
        self.is_detected_quad = False
        self.last_best_quad_center = None
        self.last_best_quad = None
        self.is_uv_spot_detected = False
        self.uv_spot_center = None

        self.uv_min_area = 0

        self.enable_color_filter = True

        self.uv_kalman = cv2.KalmanFilter(4, 2)
        self.uv_kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.uv_kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.uv_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.uv_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self._uv_q_base = 0.5
        self._uv_q_vel_base = 0.5
        self.uv_tracking_initialized = False
        self.uv_lost_frames = 0
        self.uv_max_lost_frames = 10
        self._uv_last_predict_time = None

        self.uv_adaptive_enabled = True
        self.uv_v_percentile = 95
        self.uv_v_floor = 90
        self.uv_s_min = 80
        self.uv_s_gate = 80
        self.uv_h_range = (130, 160)
        self.uv_contrast_ratio_min = 1.15
        self.uv_contrast_dilate = 30
        self._ema_v_min = None
        self._uv_miss_counter = 0

        self._init_detection_methods()

    def _init_detection_methods(self):
        self._methods: dict[DetectMethod, DetectionMethod] = {
            DetectMethod.CONTOUR_ELLIPSE: ContourEllipseMethod(self),
            DetectMethod.EDGE_CONTOUR_ELLIPSE: EdgeContourEllipseMethod(self),
            DetectMethod.EDGE_DRAWING_QUADS: EdgeDrawingQuadsMethod(self),
            DetectMethod.TEST_LINE_QUAD: TestLineQuadMethod(self),
        }

    def set_detect_method(self, method: DetectMethod):
        self.detect_method = method

    def get_detect_method(self) -> DetectMethod:
        return self.detect_method

    @staticmethod
    def get_supported_methods() -> list:
        return [
            DetectMethod.CONTOUR_ELLIPSE,
            DetectMethod.EDGE_CONTOUR_ELLIPSE,
            DetectMethod.EDGE_DRAWING_QUADS,
            DetectMethod.TEST_LINE_QUAD,
        ]

    def get_method_params(self, method: DetectMethod) -> dict:
        method_name = method.value
        params = {
            "ed_min_path_length": self.ed_min_path_length,
            "ed_gradient_threshold": self.ed_gradient_threshold,
            "ed_nfa_validation": self.ed_nfa_validation,
            "morph_type": self.morph_type,
            "morph_kernel": self.morph_kernel,
            "morph_iterations": self.morph_iterations,
            "min_area_threshold_quad": self.min_area_threshold_quad,
            "min_area_threshold_ellipse": self.min_area_threshold_ellipse,
            "min_contour_points": self.min_contour_points,
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_circularity": self.min_circularity,
            "quad_aspect_ratio": self.quad_aspect_ratio,
        }
        if method in (DetectMethod.EDGE_DRAWING_QUADS, DetectMethod.TEST_LINE_QUAD):
            params["uv_min_area"] = self.uv_min_area
            params["enable_color_filter"] = self.enable_color_filter
        return params

    def set_method_params(self, method: DetectMethod, params: dict):
        pass  # handled by update_params

    def update_params(self, params: dict):
        for key in (
            "ed_min_path_length", "ed_gradient_threshold",
            "morph_type", "morph_kernel", "morph_iterations",
            "min_area_threshold_quad", "min_area_threshold_ellipse",
            "min_contour_points", "blur_kernel",
            "uv_min_area", "uv_max_lost_frames",
            "uv_s_gate", "uv_s_min", "uv_v_floor", "uv_v_percentile",
            "uv_contrast_dilate",
        ):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in (
            "blur_sigma", "max_aspect_ratio", "min_circularity",
            "quad_aspect_ratio", "uv_contrast_ratio_min",
        ):
            if key in params:
                setattr(self, key, float(params[key]))
        bool_keys = ("ed_nfa_validation", "enable_color_filter", "uv_adaptive_enabled")
        for key in bool_keys:
            if key in params:
                setattr(self, key, bool(params[key]))
        self._update_ed_params()
        self._morph_kernel_cache = None

    def _update_ed_params(self):
        if self.ed is not None:
            ed_params = self.ed.Params()
            ed_params.MinPathLength = self.ed_min_path_length
            ed_params.GradientThresholdValue = self.ed_gradient_threshold
            ed_params.NFAValidation = self.ed_nfa_validation
            self.ed.setParams(ed_params)

    def get_edge_preview(self, frame: np.ndarray) -> np.ndarray:
        if self._last_canny_preview is not None:
            return self._last_canny_preview
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            self.ed.detectEdges(gray)
            edges = self.ed.getEdgeImage()
            if edges is not None:
                return self._apply_morphology(edges)
        except cv2.error:
            pass
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    def _apply_morphology(self, edges: np.ndarray) -> np.ndarray:
        if self.morph_type == 0:
            return edges
        if self._morph_kernel_cache is None or self._morph_kernel_cache.shape[0] != self.morph_kernel:
            self._morph_kernel_cache = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
        kernel = self._morph_kernel_cache
        if self.morph_type == 1:
            return cv2.dilate(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 2:
            return cv2.erode(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 3:
            return cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 4:
            return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        return edges

    def detect_circle_targets(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ) -> CircleTargets:
        self.circle_target.clear()
        method = self._methods.get(self.detect_method)
        if method is not None:
            method.detect(frame, target_color)
        return self.circle_target

    def _detect_contour_color(self, hsv: np.ndarray, contour, img_shape: tuple) -> Optional[str]:
        from .detection.color_utils import detect_contour_color
        return detect_contour_color(
            hsv, contour, img_shape,
            color_h_ranges=self.color_h_ranges,
            color_s_min=self.color_s_min,
            debug_color=self.debug_color,
        )

    def _order_quad_points(self, quad: np.ndarray) -> np.ndarray:
        from .detection.quad_geometry import order_quad_points
        return order_quad_points(quad)

    def _compute_quad_center(self, quad: np.ndarray) -> Tuple[float, float]:
        from .detection.quad_geometry import compute_quad_center
        return compute_quad_center(quad)

    def _get_quad_center_perspective(self, quad: np.ndarray, is_ordered: bool = False):
        from .detection.quad_geometry import get_quad_center_perspective
        return get_quad_center_perspective(quad, is_ordered)

    def _validate_center_alignment(self, quad_center, circle_center, max_offset):
        from .detection.quad_geometry import validate_center_alignment
        return validate_center_alignment(quad_center, circle_center, max_offset)

    def _is_center_aligned(self, quad_center, ellipse_center, quad, max_offset_ratio=0.15):
        from .detection.quad_geometry import is_center_aligned
        return is_center_aligned(quad_center, ellipse_center, quad, max_offset_ratio)

    def _detect_circle_in_quad(self, gray: np.ndarray, quad: np.ndarray):
        ordered_quad = self._order_quad_points(quad)
        src_points = ordered_quad.reshape(4, 2).astype(np.float32)

        quad_area = cv2.contourArea(src_points)
        target_size = int(np.sqrt(quad_area) * 0.8)
        target_size = max(100, min(400, target_size))

        dst_points = np.array(
            [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        warped = cv2.warpPerspective(gray, M, (target_size, target_size))

        min_radius = int(target_size * 0.15)
        max_radius = int(target_size * 0.4)

        circles = cv2.HoughCircles(
            warped,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=target_size // 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None or len(circles) == 0:
            return None

        center_target = np.array([target_size / 2, target_size / 2])
        best_circle = None
        best_dist = float("inf")

        for circle in circles[0]:
            cx, cy, r = circle
            dist = np.sqrt((cx - center_target[0]) ** 2 + (cy - center_target[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_circle = (cx, cy, r)

        if best_circle is None:
            return None

        cx, cy, r = best_circle

        src_center = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), M_inv
        )[0, 0]

        scale_factor = np.sqrt(quad_area) / target_size
        src_radius = r * scale_factor

        return ((float(src_center[0]), float(src_center[1])), float(src_radius))

    def _fallback_ellipse_detection(self, small, gray, scale, hsv=None):
        if self.ed is None:
            return None

        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

        try:
            self.ed.detectEdges(blurred)
            edges = self.ed.getEdgeImage()
            if edges is None or cv2.countNonZero(edges) < 100:
                return None
        except cv2.error:
            return None

        morphed = self._apply_morphology(edges)

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        quadrilaterals = find_quadrilaterals_from_contours(
            contours, scale,
            min_area_threshold_quad=self.min_area_threshold_quad,
            quad_aspect_ratio=self.quad_aspect_ratio,
            enable_color_filter=False,
            hsv=None,
            detect_color_func=None,
        )

        if not quadrilaterals:
            return None

        largest_quads = sorted(quadrilaterals, key=lambda x: x[0], reverse=True)[:5]

        best_quad = None
        best_quad_center = None

        for area, quad in largest_quads:
            quad_center = get_quad_center_perspective(quad, is_ordered=True)
            if quad_center is not None and best_quad is None:
                best_quad = quad
                best_quad_center = quad_center

            result = fit_ellipse_in_quad(
                morphed, quad, small.shape,
                min_contour_points=self.min_contour_points,
                min_area_threshold_ellipse=self.min_area_threshold_ellipse,
                max_aspect_ratio=self.max_aspect_ratio,
                min_circularity=self.min_circularity,
            )
            if result is not None:
                ellipse, contour, score = result
                ellipse_center = ellipse[0]
                if quad_center is not None:
                    if is_center_aligned(quad_center, ellipse_center, quad, max_offset_ratio=0.15):
                        return (ellipse, contour, quad, morphed, quad_center)

        if best_quad is not None and best_quad_center is not None:
            return (None, None, best_quad, morphed, best_quad_center)

        return None

    def _extract_lines_ed_lsd(self, gray: np.ndarray) -> np.ndarray:
        from .detection.line_geometry import extract_lines_ed_lsd
        return extract_lines_ed_lsd(self.ed, self.lsd, gray)

    def _find_quad_from_lines(self, lines, img_shape):
        from .detection.line_geometry import find_quad_from_lines
        return find_quad_from_lines(
            lines, img_shape,
            order_quad_points_fn=self._order_quad_points,
            min_area_threshold_quad=self.min_area_threshold_quad,
        )

    def _update_transition_matrix(self, kf, dt, q_base, q_vel_base):
        from .detection.kalman_filter import update_transition_matrix
        update_transition_matrix(kf, dt, q_base, q_vel_base)

    def _kalman_update(self, measurement: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        from .detection.kalman_filter import kalman_update
        result, self.tracking_initialized, self.lost_frames, self._last_predict_time = (
            kalman_update(
                self.kf, measurement,
                self.tracking_initialized, self.lost_frames, self.max_lost_frames,
                self._kf_q_base, self._kf_q_vel_base,
                self._last_predict_time,
            )
        )
        return result

    def _uv_kalman_update(self, uv_center: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        from .detection.kalman_filter import uv_kalman_update
        result, self.uv_tracking_initialized, self.uv_lost_frames, self._uv_last_predict_time = (
            uv_kalman_update(
                self.uv_kalman, uv_center,
                self.uv_tracking_initialized, self.uv_lost_frames, self.uv_max_lost_frames,
                self._uv_q_base, self._uv_q_vel_base,
                self._uv_last_predict_time,
            )
        )
        return result

    def _create_target_item(self, ellipse, contour, scale: float, color: str) -> CircleTargetItem:
        from .detection.target_creation import create_ellipse_target_item
        return create_ellipse_target_item(ellipse, contour, scale, color)

    def _create_quad_target_item(self, center, quad, scale, color):
        from .detection.target_creation import create_quad_target_item
        return create_quad_target_item(center, quad, scale, color)

    def _create_target_item_from_center(self, center, radius, scale, color, quad=None):
        from .detection.target_creation import create_target_item_from_center
        return create_target_item_from_center(center, radius, scale, color, quad)
