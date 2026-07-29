from typing import Optional, Dict, List, Tuple
import cv2
import numpy as np
from utils.log_util import log_print



def detect_contour_color(
    hsv: np.ndarray, contour, img_shape: tuple,
    color_h_ranges: Dict[str, Tuple[float, float, float, float]],
    color_s_min: float = 60,
    debug_color: bool = False,
) -> Optional[str]:
    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_shape[1] - x)
    h = min(h, img_shape[0] - y)
    if w <= 0 or h <= 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

    roi_hsv = hsv[y:y + h, x:x + w]
    mean_val = cv2.mean(roi_hsv, mask=mask)[:3]

    if debug_color:
        log_print(f"[ColorDebug] HSV: H={mean_val[0]:.1f}, S={mean_val[1]:.1f}, V={mean_val[2]:.1f}")

    if mean_val[1] < color_s_min:
        return "Black"

    h_low, h_high, h_low2, h_high2 = color_h_ranges["Red"]
    if (h_low <= mean_val[0] < h_high) or (h_low2 <= mean_val[0] <= h_high2):
        return "Red"

    h_low, h_high, _, _ = color_h_ranges["Green"]
    if h_low <= mean_val[0] < h_high:
        return "Green"

    h_low, h_high, _, _ = color_h_ranges["Blue"]
    if h_low <= mean_val[0] < h_high:
        return "Blue"

    return None


def get_color_mask(hsv: np.ndarray, color_name: str,
                   color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    ranges = color_ranges[color_name]
    mask = None
    for lower, upper in ranges:
        color_mask = cv2.inRange(hsv, lower, upper)
        mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)
    return mask
