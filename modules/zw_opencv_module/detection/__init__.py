# -*- coding: utf-8 -*-
"""
检测模块 - 圆形/椭圆目标检测

提供模块化的检测功能：
- 参数管理
- 图像预处理
- 颜色检测
- 四边形检测
- 椭圆检测
- UV 点检测
- 卡尔曼滤波跟踪
- 多种检测方法
"""

from .detect_method import DetectMethod
from .params import DetectionParams
from .preprocessing import Preprocessor
from .color_detection import ColorDetector
from .quad_detection import QuadDetector
from .ellipse_detection import EllipseDetector
from .uv_detection import UVDetector
from .tracking import KalmanTracker

__all__ = [
    'DetectMethod',
    'DetectionParams',
    'Preprocessor',
    'ColorDetector',
    'QuadDetector',
    'EllipseDetector',
    'UVDetector',
    'KalmanTracker',
]