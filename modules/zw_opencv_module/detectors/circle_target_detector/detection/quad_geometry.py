from typing import Optional, Tuple, Callable
import cv2
import numpy as np


def order_quad_points(quad: np.ndarray) -> np.ndarray:
    points = quad.reshape(4, 2).astype(np.float32)
    center = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    sums = sorted_points[:, 0] + sorted_points[:, 1]
    top_left_idx = np.argmin(sums)

    ordered = np.roll(sorted_points, -top_left_idx, axis=0)
    return ordered.reshape(4, 1, 2).astype(np.float32)


def check_quad_aspect_ratio(quad: np.ndarray, expected_ratio: float,
                            tolerance: float = 0.7, is_ordered: bool = False) -> bool:
    if not is_ordered:
        ordered = order_quad_points(quad)
    else:
        ordered = quad
    points = ordered.reshape(4, 2).astype(np.float32)

    tl, tr, br, bl = points

    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    estimated_width = (top_width + bottom_width) / 2

    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)
    estimated_height = (left_height + right_height) / 2

    if estimated_height <= 0 or estimated_width <= 0:
        return False

    measured_ratio = estimated_width / estimated_height

    if expected_ratio == 1.0:
        return abs(measured_ratio - 1.0) <= tolerance

    diff_normal = abs(measured_ratio - expected_ratio) / expected_ratio
    if diff_normal <= tolerance:
        return True

    inv_expected_ratio = 1.0 / expected_ratio
    diff_rotated = abs(measured_ratio - inv_expected_ratio) / inv_expected_ratio
    return diff_rotated <= tolerance


def check_angle_constraint(approx: np.ndarray, angle_threshold: float = 15.0,
                           right_angle_tolerance: float = 10.0) -> bool:
    angles = []
    pts = approx.reshape(4, 2)

    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    right_angle_count = sum(1 for a in angles if abs(a - 90) <= right_angle_tolerance)
    obtuse_count = sum(1 for a in angles if a > 90 + angle_threshold)

    if right_angle_count >= 2 and obtuse_count >= 1:
        return False
    return True


def find_quadrilaterals_from_contours(
    contours, scale: float,
    min_area_threshold_quad: float,
    quad_aspect_ratio: float,
    enable_color_filter: bool = False,
    hsv: np.ndarray = None,
    detect_color_func: Callable = None,
) -> list:
    quadrilaterals = []
    min_area_scaled = min_area_threshold_quad * (scale * scale)

    for contour in contours:
        if cv2.contourArea(contour) < min_area_scaled:
            continue

        epsilon = cv2.arcLength(contour, True) * 0.02
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > min_area_scaled:
                if not cv2.isContourConvex(approx):
                    continue

                if enable_color_filter and hsv is not None and detect_color_func is not None:
                    color = detect_color_func(hsv, approx, hsv.shape)
                    if color != "Black":
                        continue

                if check_quad_aspect_ratio(approx, quad_aspect_ratio):
                    ordered_approx = order_quad_points(approx).astype(np.int32)
                    quadrilaterals.append((area, ordered_approx))

    return quadrilaterals


def fit_ellipse_in_quad(
    edges: np.ndarray, quad, img_shape: tuple,
    min_contour_points: int = 15,
    min_area_threshold_ellipse: float = 100,
    max_aspect_ratio: float = 2.0,
    min_circularity: float = 0.4,
) -> Optional[Tuple]:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)

    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    inner_contours, _ = cv2.findContours(
        masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    best_ellipse = None
    best_contour = None
    best_score = 0

    for cnt in inner_contours:
        if len(cnt) < min_contour_points:
            continue

        contour_area = cv2.contourArea(cnt)
        if contour_area < min_area_threshold_ellipse:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        axes = ellipse[1]
        major_axis = max(axes)
        minor_axis = min(axes)

        if minor_axis <= 0 or major_axis <= 0:
            continue

        aspect_ratio = major_axis / minor_axis
        if aspect_ratio > max_aspect_ratio:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

        area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
        ellipse_contour = cv2.ellipse2Poly(
            (int(ellipse[0][0]), int(ellipse[0][1])),
            (int(major_axis / 2), int(minor_axis / 2)),
            int(ellipse[2]),
            0, 360, 5
        )

        score = area * (1.0 - aspect_ratio / max_aspect_ratio * 0.3) * circularity
        if score > best_score:
            best_score = score
            best_ellipse = ellipse
            best_contour = cnt

    if best_ellipse is not None:
        return (best_ellipse, best_contour, best_score)
    return None


def get_quad_center_perspective(quad: np.ndarray, is_ordered: bool = False) -> Optional[Tuple[float, float]]:
    if quad is None or len(quad) != 4:
        return None

    pts = quad.reshape(4, 2).astype(np.float64)
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]

    x1, y1 = p0
    x2, y2 = p2
    x3, y3 = p1
    x4, y4 = p3

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    cx = x1 + t * (x2 - x1)
    cy = y1 + t * (y2 - y1)

    return (float(cx), float(cy))


def compute_quad_center(quad: np.ndarray) -> Tuple[float, float]:
    points = quad.reshape(4, 2)
    center = np.mean(points, axis=0)
    return (float(center[0]), float(center[1]))


def validate_center_alignment(quad_center: Tuple[float, float],
                              circle_center: Tuple[float, float],
                              max_offset: float = 20.0) -> bool:
    dist = np.sqrt((quad_center[0] - circle_center[0]) ** 2 +
                   (quad_center[1] - circle_center[1]) ** 2)
    return dist <= max_offset


def is_center_aligned(quad_center, ellipse_center, quad, max_offset_ratio=0.15):
    ordered = order_quad_points(quad)
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bottom = np.linalg.norm(ordered[2] - ordered[3])
    avg_width = (w_top + w_bottom) / 2.0

    dist = np.hypot(ellipse_center[0] - quad_center[0],
                    ellipse_center[1] - quad_center[1])
    return dist <= avg_width * max_offset_ratio
