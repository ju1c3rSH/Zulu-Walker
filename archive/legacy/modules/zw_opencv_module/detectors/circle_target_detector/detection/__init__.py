from enum import Enum


class DetectMethod(Enum):
    CONTOUR_ELLIPSE = "contour_ellipse"
    EDGE_CONTOUR_ELLIPSE = "edge_contour_ellipse"
    EDGE_DRAWING_QUADS = "edge_drawing_quads"
    TEST_LINE_QUAD = "test_line_quad"
