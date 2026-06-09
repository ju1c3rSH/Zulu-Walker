from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np


def compute_v_primary_ranges(
    roi_hsv: np.ndarray, roi_mask: np.ndarray,
    uv_config: dict,
    ema_v_min: Optional[float] = None,
    uv_miss_counter: int = 0,
) -> Tuple[List, Optional[float], int]:
    s_channel = roi_hsv[:, :, 1]
    v_channel = roi_hsv[:, :, 2]
    sat_gate = (roi_mask > 0) & (s_channel >= uv_config["uv_s_gate"])
    v_sampled = v_channel[sat_gate]

    if v_sampled.size == 0:
        v_sampled = v_channel[roi_mask > 0]
    if v_sampled.size == 0:
        h_lo, h_hi = uv_config["uv_h_range"]
        ranges = [(np.array([h_lo, uv_config["uv_s_min"], uv_config["uv_v_floor"]]),
                   np.array([h_hi, 255, 255]))]
        return ranges, ema_v_min, uv_miss_counter

    raw_v_min = int(np.percentile(v_sampled, uv_config["uv_v_percentile"]))
    raw_v_min = max(raw_v_min, uv_config["uv_v_floor"])

    if ema_v_min is None:
        ema_v_min = float(raw_v_min)
    else:
        ema_v_min = (uv_config["ema_alpha"] * raw_v_min +
                     (1 - uv_config["ema_alpha"]) * ema_v_min)
    smoothed_v_min = int(ema_v_min)

    if uv_miss_counter >= uv_config["uv_miss_threshold"]:
        ema_v_min = None
        uv_miss_counter = 0
        h_lo, h_hi = uv_config["uv_h_range"]
        ranges = [(np.array([h_lo, uv_config["uv_s_min"], uv_config["uv_v_floor"]]),
                   np.array([h_hi, 255, 255]))]
        return ranges, ema_v_min, uv_miss_counter

    h_lo, h_hi = uv_config["uv_h_range"]
    ranges = [(np.array([h_lo, uv_config["uv_s_min"], smoothed_v_min]),
               np.array([h_hi, 255, 255]))]
    return ranges, ema_v_min, uv_miss_counter


def detect_uv_spot_with_search_contour(
    frame: np.ndarray, uv_config: dict,
    search_contour: np.ndarray = None,
    hsv: np.ndarray = None, gray: np.ndarray = None,
    ema_v_min: Optional[float] = None,
    uv_miss_counter: int = 0,
) -> Tuple[Optional[Tuple[int, int]], Optional[float], int]:
    uv_ranges = uv_config["color_ranges"].get("UV", [])
    if not uv_ranges:
        return None, ema_v_min, uv_miss_counter

    if search_contour is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in uv_ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        if mask is None:
            return None, ema_v_min, uv_miss_counter

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < uv_config["uv_min_area"]:
                return None, ema_v_min, uv_miss_counter
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            M = cv2.moments(masked_gray, binaryImage=False)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"] + 0.5)
                cy = int(M["m01"] / M["m00"] + 0.5)
                return (cx, cy), ema_v_min, uv_miss_counter
        return None, ema_v_min, uv_miss_counter

    x, y, w, h = cv2.boundingRect(search_contour)
    if w <= 0 or h <= 0:
        return None, ema_v_min, uv_miss_counter

    roi_frame = frame[y:y + h, x:x + w]
    roi_hsv = hsv[y:y + h, x:x + w] if hsv is not None else cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = search_contour - [x, y]
    cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)

    if uv_config.get("uv_adaptive_enabled", False):
        effective_ranges, ema_v_min, uv_miss_counter = compute_v_primary_ranges(
            roi_hsv, roi_mask, uv_config, ema_v_min, uv_miss_counter
        )
    else:
        effective_ranges = uv_ranges

    mask = None
    for lower, upper in effective_ranges:
        color_mask = cv2.inRange(roi_hsv, lower, upper)
        mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

    if mask is None:
        return None, ema_v_min, uv_miss_counter

    contour_mask = cv2.bitwise_and(mask, roi_mask)

    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < uv_config["uv_min_area"]:
            if uv_config.get("uv_adaptive_enabled", False):
                uv_miss_counter += 1
            return None, ema_v_min, uv_miss_counter

        if uv_config.get("uv_adaptive_enabled", False):
            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inner_mask, [largest_contour], -1, 255, -1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (uv_config["uv_contrast_dilate"] * 2 + 1, uv_config["uv_contrast_dilate"] * 2 + 1)
            )
            outer_mask = cv2.dilate(inner_mask, kernel)
            annulus_mask = cv2.subtract(outer_mask, inner_mask)
            if cv2.countNonZero(annulus_mask) >= 10:
                roi_v = roi_hsv[:, :, 2]
                mean_inner = cv2.mean(roi_v, mask=inner_mask)[0]
                mean_annulus = cv2.mean(roi_v, mask=annulus_mask)[0]
                if mean_annulus < 1:
                    mean_annulus = 1.0
                if mean_inner / mean_annulus < uv_config["uv_contrast_ratio_min"]:
                    uv_miss_counter += 1
                    return None, ema_v_min, uv_miss_counter

        gray_roi = gray[y:y + h, x:x + w] if gray is not None else cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray_roi, gray_roi, mask=contour_mask)
        M = cv2.moments(masked_gray, binaryImage=False)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"] + 0.5) + x
            cy = int(M["m01"] / M["m00"] + 0.5) + y
            if uv_config.get("uv_adaptive_enabled", False):
                uv_miss_counter = 0
            return (cx, cy), ema_v_min, uv_miss_counter

    if uv_config.get("uv_adaptive_enabled", False):
        uv_miss_counter += 1
    return None, ema_v_min, uv_miss_counter
