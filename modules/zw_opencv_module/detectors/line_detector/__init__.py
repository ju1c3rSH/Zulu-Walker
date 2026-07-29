"""
Line detector for black line following.
Detects a black line on a light background using thresholding.
Output: percent_error_x in [-5000, 5000], target_found bool
"""
import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LineDetectResult:
    target_found: bool = False
    percent_error_x: int = 0
    center_x: int = -1
    center_y: int = -1
    contour_area: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


class LineDetector:
    def __init__(self):
        self._debug_frame = None
    
    def detect(self, frame: np.ndarray) -> LineDetectResult:
        result = LineDetectResult()
        result.frame_width = frame.shape[1]
        result.frame_height = frame.shape[0]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 500:
                moments = cv2.moments(largest)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    w = frame.shape[1]
                    half = w / 2.0
                    pe_x = int(((cx - half) / half) * 5000.0)
                    pe_x = max(-5000, min(5000, pe_x))
                    
                    result.target_found = True
                    result.percent_error_x = pe_x
                    result.center_x = cx
                    result.center_y = cy
                    result.contour_area = area
        
        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if result.target_found:
            cv2.circle(debug, (result.center_x, result.center_y), 8, (0, 0, 255), -1)
            cv2.line(debug, (result.frame_width // 2, 0), (result.frame_width // 2, result.frame_height), (0, 255, 0), 1)
            cv2.line(debug, (result.center_x, 0), (result.center_x, result.frame_height), (0, 0, 255), 1)
        self._debug_frame = debug
        
        return result
    
    def get_debug_frame(self) -> np.ndarray:
        if self._debug_frame is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return self._debug_frame
