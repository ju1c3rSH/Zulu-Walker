# -*- coding: utf-8 -*-
"""检测参数管理"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from .detect_method import DetectMethod


@dataclass
class DetectionParams:
    """检测参数数据类"""

    # 通用参数
    min_area_threshold_quad: int = 1680  # 四边形最小面积阈值
    min_area_threshold_ellipse: int = 100  # 椭圆最小面积阈值
    min_contour_points: int = 15  # 椭圆拟合最少点数

    # EdgeDrawing 参数
    ed_min_path_length: int = 164
    ed_gradient_threshold: int = 90
    ed_nfa_validation: bool = True

    # 形态学操作参数
    morph_type: int = 4  # 0=none, 1=dilate, 2=erode, 3=open, 4=close
    morph_kernel: int = 3
    morph_iterations: int = 1

    # 高斯模糊参数
    blur_kernel: int = 5
    blur_sigma: float = 38.0

    # 椭圆验证参数
    max_aspect_ratio: float = 2.0  # 长短轴比上限
    min_circularity: float = 0.4  # 圆度下限

    # 四边形参数
    quad_aspect_ratio: float = 1.45  # 目标四边形长宽比

    # UV 点参数
    uv_min_area: int = 5  # UV 点最小面积
    enable_color_filter: bool = True  # 颜色过滤开关

    def get_method_params(self, method: DetectMethod) -> dict:
        """获取指定检测方法的参数"""
        params = {
            "min_area_threshold_quad": self.min_area_threshold_quad,
            "min_area_threshold_ellipse": self.min_area_threshold_ellipse,
            "min_contour_points": self.min_contour_points,
        }

        method_name = method.value

        if method_name == "edge_contour_ellipse":
            params.update({
                "ed_min_path_length": self.ed_min_path_length,
                "ed_gradient_threshold": self.ed_gradient_threshold,
                "ed_nfa_validation": self.ed_nfa_validation,
                "morph_type": self.morph_type,
                "morph_kernel": self.morph_kernel,
                "morph_iterations": self.morph_iterations,
                "blur_kernel": self.blur_kernel,
                "blur_sigma": self.blur_sigma,
                "max_aspect_ratio": self.max_aspect_ratio,
                "min_circularity": self.min_circularity,
            })
        elif method_name == "edge_drawing_quads":
            params.update({
                "ed_min_path_length": self.ed_min_path_length,
                "ed_gradient_threshold": self.ed_gradient_threshold,
                "ed_nfa_validation": self.ed_nfa_validation,
                "morph_type": self.morph_type,
                "morph_kernel": self.morph_kernel,
                "morph_iterations": self.morph_iterations,
                "blur_kernel": self.blur_kernel,
                "blur_sigma": self.blur_sigma,
                "quad_aspect_ratio": self.quad_aspect_ratio,
                "uv_min_area": self.uv_min_area,
                "enable_color_filter": self.enable_color_filter,
            })
        elif method_name == "test_line_quad":
            params.update({
                "blur_kernel": self.blur_kernel,
                "blur_sigma": self.blur_sigma,
            })

        return params

    def set_method_params(self, method: DetectMethod, params: dict) -> None:
        """设置指定检测方法的参数"""
        # 通用参数
        if "min_area_threshold_quad" in params:
            self.min_area_threshold_quad = params["min_area_threshold_quad"]
        if "min_area_threshold_ellipse" in params:
            self.min_area_threshold_ellipse = params["min_area_threshold_ellipse"]
        if "min_contour_points" in params:
            self.min_contour_points = params["min_contour_points"]

        method_name = method.value

        if method_name in ("edge_contour_ellipse", "edge_drawing_quads"):
            self._set_edge_params(params)

        if method_name == "edge_drawing_quads":
            if "quad_aspect_ratio" in params:
                self.quad_aspect_ratio = params["quad_aspect_ratio"]
            if "uv_min_area" in params:
                self.uv_min_area = int(params["uv_min_area"])
            if "enable_color_filter" in params:
                self.enable_color_filter = bool(params["enable_color_filter"])

        if method_name == "edge_contour_ellipse":
            if "max_aspect_ratio" in params:
                self.max_aspect_ratio = params["max_aspect_ratio"]
            if "min_circularity" in params:
                self.min_circularity = params["min_circularity"]

        if method_name == "test_line_quad":
            if "blur_kernel" in params:
                self.blur_kernel = params["blur_kernel"]
            if "blur_sigma" in params:
                self.blur_sigma = params["blur_sigma"]

    def _set_edge_params(self, params: dict) -> None:
        """设置边缘检测参数"""
        if "ed_min_path_length" in params:
            self.ed_min_path_length = params["ed_min_path_length"]
        if "ed_gradient_threshold" in params:
            self.ed_gradient_threshold = params["ed_gradient_threshold"]
        if "ed_nfa_validation" in params:
            self.ed_nfa_validation = params["ed_nfa_validation"]
        if "morph_type" in params:
            self.morph_type = params["morph_type"]
        if "morph_kernel" in params:
            self.morph_kernel = params["morph_kernel"]
        if "morph_iterations" in params:
            self.morph_iterations = params["morph_iterations"]
        if "blur_kernel" in params:
            self.blur_kernel = params["blur_kernel"]
        if "blur_sigma" in params:
            self.blur_sigma = params["blur_sigma"]

    def update_from_dict(self, params: dict) -> None:
        """从字典更新所有参数（向后兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
