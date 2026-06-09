from typing import Optional, Tuple
import cv2
import numpy as np


def extract_lines_ed_lsd(ed, lsd, gray: np.ndarray) -> np.ndarray:
    try:
        ed.detectEdges(gray)
        segments = ed.getSegments()
        if segments is None or len(segments) == 0:
            return np.empty((0, 4), dtype=np.float32)

        lines = segments.squeeze(1)
        return lines.astype(np.float32)
    except cv2.error:
        return np.empty((0, 4), dtype=np.float32)


def merge_collinear_lines(lines: np.ndarray, angle_threshold: float = 5.0,
                          distance_threshold: float = 10.0) -> np.ndarray:
    if len(lines) < 2:
        return lines

    merged = []
    used = set()

    for i in range(len(lines)):
        if i in used:
            continue

        group = [i]
        used.add(i)
        x1, y1, x2, y2 = lines[i]
        dx, dy = x2 - x1, y2 - y1
        ref_angle = np.degrees(np.arctan2(abs(dy), abs(dx))) % 180

        for j in range(i + 1, len(lines)):
            if j in used:
                continue

            x3, y3, x4, y4 = lines[j]
            dx2, dy2 = x4 - x3, y4 - y3
            angle2 = np.degrees(np.arctan2(abs(dy2), abs(dx2))) % 180

            angle_diff = abs(ref_angle - angle2)
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff > angle_threshold:
                continue

            mid1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            mid2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])
            dist = np.linalg.norm(mid1 - mid2)

            if dist <= distance_threshold:
                group.append(j)
                used.add(j)

        if len(group) > 1:
            merged_line = merge_line_group(lines[group])
            if merged_line is not None:
                merged.append(merged_line)
        else:
            merged.append(lines[i])

    return np.array(merged, dtype=np.float32) if merged else lines


def line_to_line_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """Compute minimum distance between two line segments."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    def dist_to_segment(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-10)
        t = max(0, min(1, t))
        cx, cy = ax + t * abx, ay + t * aby
        return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    d1 = dist_to_segment(x3, y3, x1, y1, x2, y2)
    d2 = dist_to_segment(x4, y4, x1, y1, x2, y2)
    d3 = dist_to_segment(x1, y1, x3, y3, x4, y4)
    d4 = dist_to_segment(x2, y2, x3, y3, x4, y4)
    return min(d1, d2, d3, d4)


def merge_line_group(lines: np.ndarray) -> np.ndarray:
    if len(lines) == 0:
        return None
    if len(lines) == 1:
        return lines[0]

    all_pts = []
    for line in lines:
        all_pts.append((line[0], line[1]))
        all_pts.append((line[2], line[3]))

    all_pts = np.array(all_pts, dtype=np.float32)
    distances = np.linalg.norm(all_pts - all_pts.mean(axis=0), axis=1)
    far_indices = np.argsort(distances)[-2:]

    return np.array([*all_pts[far_indices[0]], *all_pts[far_indices[1]]], dtype=np.float32)


def find_quad_from_lines(lines: np.ndarray, img_shape: tuple,
                         order_quad_points_fn=None,
                         min_area_threshold_quad: float = 1680) -> Optional[np.ndarray]:
    h, w = img_shape[:2]
    if lines is None or len(lines) < 4:
        return None

    angles = np.degrees(np.arctan2(
        lines[:, 3] - lines[:, 1],
        lines[:, 2] - lines[:, 0]
    )) % 180

    hist, bin_edges = np.histogram(angles, bins=36, range=(0, 180))
    peak_indices = np.argsort(hist)[-4:]
    peak_angles = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2

    best_pair = None
    best_score = 0

    for i in range(len(peak_angles)):
        for j in range(i + 1, len(peak_angles)):
            diff = abs(peak_angles[i] - peak_angles[j])
            diff = min(diff, 180 - diff)
            if 70 < diff < 110:
                score = hist[peak_indices[i]] + hist[peak_indices[j]]
                if score > best_score:
                    best_score = score
                    best_pair = (peak_angles[i], peak_angles[j])

    if best_pair is None:
        return None

    angle1, angle2 = best_pair

    valid_mask = np.ones(len(lines), dtype=bool)
    valid_lines = lines[valid_mask]

    group1, group2 = [], []
    valid_angles = np.degrees(np.arctan2(
        valid_lines[:, 3] - valid_lines[:, 1],
        valid_lines[:, 2] - valid_lines[:, 0]
    )) % 180

    for i, line in enumerate(valid_lines):
        angle = valid_angles[i]
        diff1 = min(abs(angle - angle1), 180 - abs(angle - angle1))
        diff2 = min(abs(angle - angle2), 180 - abs(angle - angle2))
        if diff1 < diff2:
            group1.append(line)
        else:
            group2.append(line)

    if len(group1) < 2 or len(group2) < 2:
        return None

    def find_extreme_parallel(lines_group, ref_angle):
        if len(lines_group) < 2:
            return None, None
        distances = []
        for line in lines_group:
            x1, y1, x2, y2 = line
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            perp_angle = ref_angle + 90
            dist = mid_x * np.cos(np.radians(perp_angle)) + mid_y * np.sin(np.radians(perp_angle))
            distances.append(dist)
        distances = np.array(distances)
        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        return lines_group[min_idx], lines_group[max_idx]

    line1a, line1b = find_extreme_parallel(group1, angle1)
    line2a, line2b = find_extreme_parallel(group2, angle2)

    if line1a is None or line1b is None or line2a is None or line2b is None:
        return None

    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    corners = []
    for l1 in [line1a, line1b]:
        for l2 in [line2a, line2b]:
            pt = line_intersection(l1, l2)
            if pt is not None:
                corners.append(pt)

    if len(corners) != 4:
        return None

    corners = np.array(corners, dtype=np.float32)

    if np.any(corners[:, 0] < -w * 0.1) or np.any(corners[:, 0] > w * 1.1):
        return None
    if np.any(corners[:, 1] < -h * 0.1) or np.any(corners[:, 1] > h * 1.1):
        return None

    hull = cv2.convexHull(corners.astype(np.float32))
    area = cv2.contourArea(hull)

    min_area_scaled = min_area_threshold_quad * 0.5
    if area < min_area_scaled:
        return None

    rect = cv2.minAreaRect(hull)
    width, height = rect[1]
    if width > 0 and height > 0:
        aspect = max(width, height) / min(width, height)
        if aspect > 5.0:
            return None

    if order_quad_points_fn is not None:
        ordered = order_quad_points_fn(corners)
    else:
        ordered = corners

    return ordered
