# -*- coding: utf-8 -*-
"""边缘检测 + 四边形检测方法"""

from typing import Optional, Tuple
import cv2
import numpy as np

from ...circle import CircleTargets, CircleTargetItem, ShapeType
from ..detect_method import DetectMethod
from ..preprocessing import Preprocessor
from ..color_detection import ColorDetector
from ..quad_detection import QuadDetector
from ..uv_detection import UVDetector
from ..tracking import KalmanTracker
from .base import DetectionMethodBase


class EdgeDrawingQuadsMethod(DetectionMethodBase):
    """边缘检测 + 四边形检测"""

    def __init__(
        self,
        preprocessor: Preprocessor,
        color_detector: ColorDetector,
        quad_detector: QuadDetector,
        uv_detector: UVDetector,
        tracker: KalmanTracker,
        params,
    ):
        super().__init__("edge_drawing_quads")
        self.preprocessor = preprocessor
        self.color_detector = color_detector
        self.quad_detector = quad_detector
        self.uv_detector = uv_detector
        self.tracker = tracker
        self.params = params
        self.circle_target = CircleTargets()
        self._last_edge_preview: Optional[np.ndarray] = None

        # 检测状态
        self.is_detected_quad: bool = False
        self.is_uv_spot_detected: bool = False
        self.uv_spot_center: Optional[Tuple[int, int]] = None

    @property
    def method(self) -> DetectMethod:
        return DetectMethod.EDGE_DRAWING_QUADS

    def detect(
        self,
        frame: np.ndarray,
        target_color: Optional[str] = None,
    ) -> CircleTargets:
        self.circle_target.clear()

        # 重置 UV 检测状态
        self.is_uv_spot_detected = False
        self.uv_spot_center = None

        small, gray, hsv, scale = self.preprocessor.preprocess(frame)
        blurred = self.preprocessor.gaussian_blur(gray)

        # EdgeDrawing 边缘检测
        edges = self.preprocessor.detect_edges(blurred)
        if edges is None or np.count_nonzero(edges) < 100:
            self._last_edge_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
            return self.circle_target

        # 形态学操作
        morphed = self.preprocessor.apply_morphology(edges)
        self._last_edge_preview = morphed

        # 查找轮廓
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 查找四边形
        quadrilaterals = self.quad_detector.find_quadrilaterals(
            contours, scale, hsv, self.color_detector
        )
        if not quadrilaterals:
            self.is_detected_quad = False
            return self.circle_target

        # 按面积排序
        largest_quads = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[:3]

        best_quad = None
        best_quad_center = None
        best_area = 0

        for quad in largest_quads:
            quad_center = self.quad_detector.get_quad_center_perspective(
                quad, self.params.quad_aspect_ratio
            )
            if quad_center is not None:
                area = cv2.contourArea(quad)
                if area > best_area:
                    best_area = area
                    best_quad = quad
                    best_quad_center = quad_center

        if best_quad is None or best_quad_center is None:
            self.is_detected_quad = False
            return self.circle_target

        self.is_detected_quad = True

        # 检测 UV 点
        uv_center = self.uv_detector.detect(small, best_quad)
        if uv_center is not None:
            uv_center_orig = (int(uv_center[0] / scale), int(uv_center[1] / scale))
            self.uv_spot_center = self.tracker.uv_update(uv_center_orig)
            self.is_uv_spot_detected = True
        else:
            self.is_uv_spot_detected = False
            self.uv_spot_center = None

        # 卡尔曼滤波平滑
        smoothed_center = self.tracker.update(best_quad_center)
        final_center = smoothed_center if smoothed_center is not None else best_quad_center

        # 创建目标项
        color_name = target_color if target_color else 'Red'
        target_item = self._create_quad_target_item(
            final_center, best_quad, scale, color_name
        )
        self.circle_target.add_target(target_item)

        return self.circle_target

    def get_edge_preview(self) -> Optional[np.ndarray]:
        preview = self._last_edge_preview
        self._last_edge_preview = None
        return preview

    def _create_quad_target_item(
        self,
        center: Tuple[float, float],
        quad: np.ndarray,
        scale: float,
        color: str,
    ) -> CircleTargetItem:
        """从四边形创建目标项"""
        center_orig = (int(center[0] / scale), int(center[1] / scale))
        quad_orig = (quad.astype(np.float32) / scale).astype(np.int32)
        area = cv2.contourArea(quad_orig)
        x, y, w_box, h_box = cv2.boundingRect(quad_orig)

        return CircleTargetItem(
            index=0,
            center_coordinates=center_orig,
            radius=0.0,
            area=area,
            shape_type=ShapeType.QUAD,
            contour_points=quad_orig,
            bounding_box=(x, y, w_box, h_box),
            color=color,
            major_axis=None,
            minor_axis=None,
            quad_points=quad_orig,
        )