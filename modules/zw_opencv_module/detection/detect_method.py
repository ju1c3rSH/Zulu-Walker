# -*- coding: utf-8 -*-
"""检测方法枚举定义"""

from enum import Enum


class DetectMethod(Enum):
    """检测方法枚举"""
    CONTOUR_ELLIPSE = "contour_ellipse"  # 轮廓检测 + 椭圆拟合
    EDGE_CONTOUR_ELLIPSE = "edge_contour_ellipse"  # 边缘检测 + 椭圆拟合
    EDGE_DRAWING_QUADS = "edge_drawing_quads"  # 边缘检测 + 四边形检测
    TEST_LINE_QUAD = "test_line_quad"  # 测试：线段+四边形+透视变换+霍夫圆
