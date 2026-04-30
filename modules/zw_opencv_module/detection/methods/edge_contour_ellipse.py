# -*- coding: utf-8 -*-
"""边缘检测 + 轮廓 + 椭圆拟合方法"""

from typing import Optional, Tuple
import cv2
import numpy as np

from ...circle import CircleTargets, CircleTargetItem, ShapeType
from ..detect_method import DetectMethod
from ..preprocessing import Preprocessor
from ..color_detection import ColorDetector
from ..quad_detection import QuadDetector
from ..ellipse_detection import EllipseDetector
from ..tracking import KalmanTracker
from .base import DetectionMethodBase


class EdgeContourEllipseMethod(DetectionMethodBase):
    """边缘检测 + 轮廓 + 椭圆拟合"""

    def __init__(
        self,
        preprocessor: Preprocessor,
        color_detector: ColorDetector,
        quad_detector: QuadDetector,
        ellipse_detector: EllipseDetector,
        tracker: KalmanTracker,
        params,
    ):
        super().__init__("edge_contour_ellipse")
        self.preprocessor = preprocessor
        self.color_detector = color_detector
        self.quad_detector = quad_detector
        self.ellipse_detector = ellipse_detector
        self.tracker = tracker
        self.params = params
        self.circle_target = CircleTargets()
        self._last_edge_preview: Optional[np.ndarray] = None

    @property
    def method(self) -> DetectMethod:
        return DetectMethod.EDGE_CONTOUR_ELLIPSE

    def detect(
        self,
        frame: np.ndarray,
        target_color: Optional[str] = None,
    ) -> CircleTargets:
        self.circle_target.clear()

        small, gray, hsv, scale = self.preprocessor.preprocess(frame)
        blurred = self.preprocessor.gaussian_blur(gray)

        result = None
        morphed_edges = None
        detected_quad = None
        detected_quad_center = None

        # 回退检测
        fallback_result = self._fallback_ellipse_detection(small, gray, scale, hsv)
        if fallback_result is not None:
            ellipse, contour, quad, morphed_edges, quad_center = fallback_result
            detected_quad = quad
            detected_quad_center = quad_center
            if ellipse is not None:
                center = ellipse[0]
                radius = max(ellipse[1]) / 2
                result = (center, radius, quad, quad_center)

        # 卡尔曼滤波平滑
        final_center = None
        final_radius = None
        final_quad = detected_quad
        final_quad_center = detected_quad_center

        if result is not None:
            center, radius, quad, quad_center = result
            final_radius = radius
            smoothed_center = self.tracker.update(center)
            final_center = smoothed_center if smoothed_center is not None else center
        else:
            predicted = self.tracker.update(None)
            if predicted is not None:
                final_center = predicted
                final_radius = None
            elif detected_quad_center is not None:
                final_center = detected_quad_center
                final_radius = None

        if final_center is not None:
            color_name = target_color if target_color else 'Red'
            if final_radius is not None:
                target_item = self._create_target_item_from_center(
                    final_center, final_radius, scale, color_name, final_quad
                )
            else:
                target_item = self._create_target_item_from_center(
                    final_center, 20.0, scale, color_name, final_quad
                )
            self.circle_target.add_target(target_item)

        # 缓存边缘图像
        if morphed_edges is not None:
            self._last_edge_preview = morphed_edges
        else:
            self._cache_edge_preview(blurred, small.shape)

        return self.circle_target

    def get_edge_preview(self) -> Optional[np.ndarray]:
        preview = self._last_edge_preview
        self._last_edge_preview = None
        return preview

    def _fallback_ellipse_detection(
        self, small: np.ndarray, gray: np.ndarray, scale: float, hsv: np.ndarray
    ) -> Optional[Tuple]:
        """回退检测方法"""
        blurred = self.preprocessor.gaussian_blur(gray)

        edges = self.preprocessor.detect_edges(blurred)
        if edges is None or np.count_nonzero(edges) < 100:
            return None

        morphed = self.preprocessor.apply_morphology(edges)

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        quadrilaterals = self.quad_detector.find_quadrilaterals(
            contours, scale, hsv, self.color_detector
        )
        if not quadrilaterals:
            return None

        largest_quads = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[:5]

        best_quad = None
        best_quad_center = None

        for quad in largest_quads:
            quad_center = self.quad_detector.get_quad_center_perspective(
                quad, self.params.quad_aspect_ratio
            )
            if quad_center is not None and best_quad is None:
                best_quad = quad
                best_quad_center = quad_center

            result = self.ellipse_detector.fit_ellipse_in_quad(morphed, quad, small.shape)
            if result is not None:
                ellipse, contour, score = result
                ellipse_center = ellipse[0]
                if quad_center is not None:
                    if self.ellipse_detector.is_center_aligned(
                        quad_center, ellipse_center, quad
                    ):
                        return (ellipse, contour, quad, morphed, quad_center)

        if best_quad is not None and best_quad_center is not None:
            return (None, None, best_quad, morphed, best_quad_center)

        return None

    def _cache_edge_preview(self, blurred: np.ndarray, shape: tuple) -> None:
        """缓存边缘预览"""
        try:
            edges = self.preprocessor.detect_edges(blurred)
            if edges is not None:
                self._last_edge_preview = self.preprocessor.apply_morphology(edges)
            else:
                self._last_edge_preview = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        except cv2.error:
            self._last_edge_preview = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    def _create_target_item_from_center(
        self,
        center: Tuple[float, float],
        radius: float,
        scale: float,
        color: str,
        quad: np.ndarray = None,
    ) -> CircleTargetItem:
        """从圆心和半径创建目标项"""
        center_orig = (int(center[0] / scale), int(center[1] / scale))
        radius_orig = radius / scale
        area = np.pi * radius_orig * radius_orig

        contour_points = cv2.ellipse2Poly(
            center_orig,
            (int(radius_orig), int(radius_orig)),
            0, 0, 360, 5
        )

        quad_orig = None
        if quad is not None:
            quad_orig = (quad.astype(np.float32) / scale).astype(np.int32)

        return CircleTargetItem(
            index=0,
            center_coordinates=center_orig,
            radius=radius_orig,
            area=area,
            shape_type=ShapeType.CIRCLE,
            contour_points=contour_points,
            bounding_box=(
                int(center_orig[0] - radius_orig),
                int(center_orig[1] - radius_orig),
                int(2 * radius_orig),
                int(2 * radius_orig),
            ),
            color=color,
            major_axis=2 * radius_orig,
            minor_axis=2 * radius_orig,
            quad_points=quad_orig,
        )