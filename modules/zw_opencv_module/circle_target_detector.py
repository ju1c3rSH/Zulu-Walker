# -*- coding: utf-8 -*-
"""
圆形/椭圆目标检测器 - 门面类

本模块提供统一的检测接口，内部组合各子模块实现检测功能。
保持向后兼容性，原有调用方无需修改。
"""

from typing import Optional, Tuple, List
import numpy as np

from .circle import CircleTargets
from .detection.detect_method import DetectMethod
from .detection.params import DetectionParams
from .detection.preprocessing import Preprocessor
from .detection.color_detection import ColorDetector
from .detection.quad_detection import QuadDetector
from .detection.ellipse_detection import EllipseDetector
from .detection.uv_detection import UVDetector
from .detection.tracking import KalmanTracker
from .detection.methods.base import DetectionMethodBase
from .detection.methods.contour_ellipse import ContourEllipseMethod
from .detection.methods.edge_contour_ellipse import EdgeContourEllipseMethod
from .detection.methods.edge_drawing_quads import EdgeDrawingQuadsMethod
from .detection.methods.test_line_quad import TestLineQuadMethod


class CircleTargetDetector:
    """
    圆形/椭圆目标检测器

    门面类，组合各子模块提供统一的检测接口。
    保持向后兼容性。
    """

    def __init__(self, name: str = "circle_target"):
        self.name = name

        # 初始化子模块
        self.params = DetectionParams()
        self.preprocessor = Preprocessor()
        self.color_detector = ColorDetector()
        self.quad_detector = QuadDetector()
        self.ellipse_detector = EllipseDetector()
        self.uv_detector = UVDetector()
        self.tracker = KalmanTracker()

        # 检测方法实例
        self._methods: dict = {}
        self._current_method: DetectMethod = DetectMethod.EDGE_DRAWING_QUADS

        # 检测结果
        self.circle_target = CircleTargets()

        # 同步参数到子模块
        self._sync_params_to_modules()

    def _sync_params_to_modules(self) -> None:
        """同步参数到各子模块"""
        # 预处理器参数
        self.preprocessor.set_params(
            ed_min_path_length=self.params.ed_min_path_length,
            ed_gradient_threshold=self.params.ed_gradient_threshold,
            ed_nfa_validation=self.params.ed_nfa_validation,
            morph_type=self.params.morph_type,
            morph_kernel=self.params.morph_kernel,
            morph_iterations=self.params.morph_iterations,
            blur_kernel=self.params.blur_kernel,
            blur_sigma=self.params.blur_sigma,
        )

        # 四边形检测器参数
        self.quad_detector.min_area_threshold = self.params.min_area_threshold_quad
        self.quad_detector.quad_aspect_ratio = self.params.quad_aspect_ratio
        self.quad_detector.enable_color_filter = self.params.enable_color_filter

        # 椭圆检测器参数
        self.ellipse_detector.min_area_threshold = self.params.min_area_threshold_ellipse
        self.ellipse_detector.min_contour_points = self.params.min_contour_points
        self.ellipse_detector.max_aspect_ratio = self.params.max_aspect_ratio
        self.ellipse_detector.min_circularity = self.params.min_circularity

        # UV 检测器参数
        self.uv_detector.min_area = self.params.uv_min_area

        # 颜色检测器参数
        self.color_detector.enable_color_filter = self.params.enable_color_filter

    def _get_method_instance(self, method: DetectMethod) -> DetectionMethodBase:
        """获取或创建检测方法实例"""
        if method not in self._methods:
            if method == DetectMethod.CONTOUR_ELLIPSE:
                self._methods[method] = ContourEllipseMethod(
                    self.preprocessor,
                    self.color_detector,
                    self.params,
                )
            elif method == DetectMethod.EDGE_CONTOUR_ELLIPSE:
                self._methods[method] = EdgeContourEllipseMethod(
                    self.preprocessor,
                    self.color_detector,
                    self.quad_detector,
                    self.ellipse_detector,
                    self.tracker,
                    self.params,
                )
            elif method == DetectMethod.EDGE_DRAWING_QUADS:
                self._methods[method] = EdgeDrawingQuadsMethod(
                    self.preprocessor,
                    self.color_detector,
                    self.quad_detector,
                    self.uv_detector,
                    self.tracker,
                    self.params,
                )
            elif method == DetectMethod.TEST_LINE_QUAD:
                self._methods[method] = TestLineQuadMethod(
                    self.preprocessor,
                    self.color_detector,
                    self.quad_detector,
                    self.ellipse_detector,
                    self.tracker,
                    self.params,
                )

        return self._methods[method]

    # ==================== 公共接口 ====================

    def detect_circle_targets(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ) -> CircleTargets:
        """
        检测图像中指定颜色的圆形/椭圆目标

        Args:
            frame: BGR格式的输入图像
            target_color: 目标颜色 ('Red', 'Green', 'Blue', 'Black')，None 表示检测所有颜色

        Returns:
            CircleTargets: 包含所有检测到的目标信息的对象
        """
        self.circle_target.clear()

        method_instance = self._get_method_instance(self._current_method)
        result = method_instance.detect(frame, target_color)
        self.circle_target = result

        return self.circle_target

    def set_detect_method(self, method: DetectMethod) -> None:
        """设置检测方法"""
        self._current_method = method

    def get_detect_method(self) -> DetectMethod:
        """获取当前检测方法"""
        return self._current_method

    @staticmethod
    def get_supported_methods() -> list:
        """返回支持的方法列表"""
        return list(DetectMethod)

    def get_method_params(self, method: DetectMethod) -> dict:
        """获取指定检测方法的参数"""
        return self.params.get_method_params(method)

    def set_method_params(self, method: DetectMethod, params: dict) -> None:
        """设置指定检测方法的参数"""
        self.params.set_method_params(method, params)
        self._sync_params_to_modules()

    def update_params(self, params: dict) -> None:
        """
        更新边缘检测参数（从调试窗口调用，保持向后兼容）

        Args:
            params: 参数字典
        """
        self.params.update_from_dict(params)
        self._sync_params_to_modules()

    def get_edge_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        获取边缘预览图像（用于调试窗口）

        Args:
            frame: BGR格式的输入图像

        Returns:
            EdgeDrawing边缘图像（灰度）
        """
        method_instance = self._get_method_instance(self._current_method)
        preview = method_instance.get_edge_preview()

        if preview is not None:
            return preview

        return self.preprocessor.get_edge_preview(frame)

    # ==================== 属性访问（向后兼容） ====================

    @property
    def is_detected_quad(self) -> bool:
        """当前是否检测到四边形"""
        method = self._get_method_instance(self._current_method)
        if isinstance(method, EdgeDrawingQuadsMethod):
            return method.is_detected_quad
        return False

    @property
    def is_uv_spot_detected(self) -> bool:
        """是否检测到UV点"""
        method = self._get_method_instance(self._current_method)
        if isinstance(method, EdgeDrawingQuadsMethod):
            return method.is_uv_spot_detected
        return False

    @property
    def uv_spot_center(self) -> Optional[Tuple[int, int]]:
        """UV点中心坐标"""
        method = self._get_method_instance(self._current_method)
        if isinstance(method, EdgeDrawingQuadsMethod):
            return method.uv_spot_center
        return None

    @property
    def detect_method(self) -> DetectMethod:
        """当前检测方法（属性访问）"""
        return self._current_method

    # ==================== 辅助方法（向后兼容） ====================

    def _order_quad_points(self, quad: np.ndarray) -> np.ndarray:
        """将四边形的四个顶点按顺时针顺序排列"""
        return self.quad_detector._order_quad_points(quad)
