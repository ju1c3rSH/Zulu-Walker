# -*- coding: utf-8 -*-
"""检测方法模块"""

from .base import DetectionMethodBase
from .contour_ellipse import ContourEllipseMethod
from .edge_contour_ellipse import EdgeContourEllipseMethod
from .edge_drawing_quads import EdgeDrawingQuadsMethod
from .test_line_quad import TestLineQuadMethod

__all__ = [
    'DetectionMethodBase',
    'ContourEllipseMethod',
    'EdgeContourEllipseMethod',
    'EdgeDrawingQuadsMethod',
    'TestLineQuadMethod',
]