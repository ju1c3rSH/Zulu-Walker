# -*- coding: utf-8 -*-
"""
EOD2026 hostage shape classifier (contour geometry, zero-training baseline).

Classifies the frontal silhouette of an upright hostage block into
  SHAPE_CYLINDER(1) / SHAPE_CONE(2, incl. frustum 圆台) / SHAPE_WAIST_DRUM(3).

Method — vertical width profile of the filled contour:
  cylinder   : width ~constant over height, rectangular fill
  cone       : width monotonic from ~0 (or small end) to max
  waist-drum : both ends wide, middle dips (concave waist)

The 2025 provincial-first-prize EOD entry used a 3-class NN detector for this
task (more robust to viewing angle); this geometric classifier is the
training-data-free baseline and doubles as a sanity check. Swap-in path:
MaixCam2AI YOLO11 slot (see docs/eod/method_selection.md).

Pure cv2/numpy — runs on device AND on PC (pytest).
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.zw_uart_module.protocol import (
    SHAPE_CYLINDER, SHAPE_CONE, SHAPE_WAIST_DRUM,
)

# profile sampling rows (fractions of bbox height)
_ROW_TOP, _ROW_MID, _ROW_BOTTOM = 0.15, 0.50, 0.85
_WAIST_BAND = (0.30, 0.70)   # waist search band


def _width_profile(contour: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """Per-row width (max_x - min_x + 1) of the filled contour inside its bbox."""
    x, y, w, h = cv2.boundingRect(contour)
    if h < 4 or w < 4:
        return None, (x, y, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.astype(np.int32) - np.array([x, y])
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    widths = np.zeros(h, dtype=np.float32)
    for row in range(h):
        cols = np.flatnonzero(mask[row])
        if cols.size:
            widths[row] = cols[-1] - cols[0] + 1
    if widths.max() <= 0:
        return None, (x, y, w, h)
    k = max(3, h // 20) | 1
    widths = np.convolve(widths, np.ones(k, dtype=np.float32) / k, mode="same")
    return widths / widths.max(), (x, y, w, h)


def _row_width(profile: np.ndarray, frac: float) -> float:
    idx = int(frac * (len(profile) - 1))
    return float(profile[idx])


def classify_contour(
    contour: np.ndarray,
    waist_ratio_max: float = 0.88,
    cone_small_end_max: float = 0.70,
    cone_slope_min: float = 0.30,
    cylinder_end_min: float = 0.82,
    min_confidence: float = 0.25,
) -> Tuple[int, float, dict]:
    """Classify one contour.

    Returns (shape_id, confidence 0..1, features dict); shape_id 0 = unknown.
    """
    features: dict = {}
    profile, bbox = _width_profile(contour)
    if profile is None:
        return 0, 0.0, features
    x, y, w, h = bbox

    top = _row_width(profile, _ROW_TOP)
    bottom = _row_width(profile, _ROW_BOTTOM)
    band = profile[int(_WAIST_BAND[0] * len(profile)):int(_WAIST_BAND[1] * len(profile))]
    mid = float(band.min()) if band.size else min(top, bottom)
    maxw = 1.0
    top_r, bottom_r, mid_r = top / maxw, bottom / maxw, mid / maxw
    ends_min = min(top_r, bottom_r)

    # taper = least-squares slope of the width profile over rows 10%..90%
    # (robust for both full triangles and frustums, unlike end-point sampling)
    prof_band = profile[int(0.1 * len(profile)):int(0.9 * len(profile))]
    if prof_band.size >= 4:
        xs = np.linspace(0.0, 1.0, prof_band.size).astype(np.float32)
        taper = abs(float(np.polyfit(xs, prof_band, 1)[0]))
    else:
        taper = abs(bottom_r - top_r)

    area = cv2.contourArea(contour)
    rectangularity = area / float(w * h) if w * h > 0 else 0.0

    features = {
        "top_ratio": round(top_r, 3),
        "bottom_ratio": round(bottom_r, 3),
        "mid_ratio": round(mid_r, 3),
        "slope_ratio": round(taper, 3),
        "rectangularity": round(rectangularity, 3),
        "bbox": (x, y, w, h),
    }

    # hard gates first (a class must qualify at all), then scores decide
    dip = ends_min - mid_r          # positive = middle narrower than both ends
    qualifies_cyl = ends_min >= cylinder_end_min
    qualifies_cone = taper >= cone_slope_min and ends_min <= cone_small_end_max
    qualifies_waist = mid_r <= waist_ratio_max and dip > 0.08 and ends_min >= 0.70

    # score_cylinder: both ends wide, no middle dip, rectangle-like fill
    no_dip = max(0.0, mid_r - 0.10)  # tolerate small measurement noise
    score_cyl = min(ends_min, no_dip) * min(1.0, rectangularity / 0.80) if qualifies_cyl else 0.0
    # score_cone: strong monotonic taper with a small end
    small_end = 1.0 - ends_min
    score_cone = taper * (0.5 + 0.5 * small_end) if qualifies_cone else 0.0
    # score_waist: ends wide, middle clearly narrower (concave waist)
    score_waist = dip * ends_min if qualifies_waist else 0.0

    candidates = [
        (SHAPE_CYLINDER, score_cyl),
        (SHAPE_CONE, score_cone),
        (SHAPE_WAIST_DRUM, score_waist),
    ]
    shape_id, best = max(candidates, key=lambda c: c[1])
    if best < min_confidence:
        return 0, best, features
    features["scores"] = {
        "cylinder": round(score_cyl, 3),
        "cone": round(score_cone, 3),
        "waist_drum": round(score_waist, 3),
    }
    return shape_id, min(1.0, best), features


def classify_id(contour: np.ndarray, **kwargs) -> int:
    """Convenience wrapper returning only the shape id."""
    shape_id, _, _ = classify_contour(contour, **kwargs)
    return shape_id
