# -*- coding: utf-8 -*-
"""Shape classifier tests on synthetic silhouettes (frontal view, upright)."""
import cv2
import numpy as np
import pytest

from modules.zw_opencv_module.detectors.eod_shape import classify_contour, classify_id
from modules.zw_uart_module.protocol import (
    SHAPE_CYLINDER, SHAPE_CONE, SHAPE_WAIST_DRUM,
)

pytest.importorskip("cv2")


def contour_of_polygon(points, scale=4):
    """Build a filled polygon and return its largest external contour."""
    pts = np.array(points, dtype=np.int32) * scale
    canvas = np.zeros((600, 600), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def test_cylinder_rectangle():
    # frontal cylinder silhouette = rectangle
    contour = contour_of_polygon([(30, 30), (90, 30), (90, 120), (30, 120)])
    shape_id, conf, feats = classify_contour(contour)
    assert shape_id == SHAPE_CYLINDER
    assert conf > 0.3
    assert feats["top_ratio"] > 0.8 and feats["bottom_ratio"] > 0.8


def test_cone_triangle():
    # frontal cone silhouette = triangle (apex up); frustum 圆台 also tapers
    contour = contour_of_polygon([(60, 30), (95, 120), (25, 120)])
    shape_id, conf, _ = classify_contour(contour)
    assert shape_id == SHAPE_CONE


def test_cone_frustum_trapezoid():
    # 圆台: small top end, wide bottom
    contour = contour_of_polygon([(50, 30), (85, 30), (100, 120), (35, 120)])
    shape_id, _, _ = classify_contour(contour)
    assert shape_id == SHAPE_CONE


def test_waist_drum_concave_sides():
    # 腰鼓: both ends wide, middle dips (concave waist)
    contour = contour_of_polygon([
        (30, 30), (90, 30),          # top edge
        (68, 75),                    # right waist dip
        (90, 120), (30, 120),        # bottom edge
        (52, 75),                    # left waist dip
    ])
    shape_id, conf, feats = classify_contour(contour)
    assert shape_id == SHAPE_WAIST_DRUM
    assert feats["mid_ratio"] < feats["top_ratio"]


def test_unknown_garbage_returns_zero():
    # blob too small to profile (bbox height < 4 px) cannot be classified
    contour = contour_of_polygon([(50, 50), (53, 50), (51, 52)], scale=1)
    shape_id, _, _ = classify_contour(contour)
    assert shape_id == 0


def test_classify_id_convenience():
    contour = contour_of_polygon([(30, 30), (90, 30), (90, 120), (30, 120)])
    assert classify_id(contour) == SHAPE_CYLINDER
