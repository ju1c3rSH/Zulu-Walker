from typing import Optional, Tuple
import cv2
import numpy as np

from ....models.circle import CircleTargetItem, CircleTargets, ShapeType


def create_ellipse_target_item(
    ellipse, contour, scale: float, color: str,
) -> CircleTargetItem:
    center_orig = (int(ellipse[0][0] / scale), int(ellipse[0][1] / scale))
    axes_orig = (ellipse[1][0] / scale, ellipse[1][1] / scale)
    radius = max(axes_orig) / 2
    area = 3.14159 * (axes_orig[0] / 2) * (axes_orig[1] / 2)

    if scale < 1.0:
        contour_orig = (contour * (1.0 / scale)).astype(np.int32)
    else:
        contour_orig = contour

    return CircleTargetItem(
        index=0,
        center_coordinates=center_orig,
        radius=radius,
        area=area,
        shape_type=ShapeType.ELLIPSE,
        contour_points=contour_orig,
        bounding_box=(
            int((center_orig[0] - axes_orig[0] / 2)),
            int((center_orig[1] - axes_orig[1] / 2)),
            int(axes_orig[0]),
            int(axes_orig[1]),
        ),
        color=color,
        major_axis=axes_orig[0],
        minor_axis=axes_orig[1],
    )


def create_quad_target_item(
    center: Tuple[float, float], quad: np.ndarray,
    scale: float, color: str,
) -> CircleTargetItem:
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


def create_target_item_from_center(
    center: Tuple[float, float], radius: float,
    scale: float, color: str, quad: np.ndarray = None,
) -> CircleTargetItem:
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
