from typing import Optional
from .circle import CircleTargets
import cv2
import numpy as np
from .processors.base import Processor, VisionResult
from skimage import transform

class CircleTargetDetector:
    def __init__(self, name: str):
        self.name = name
        self.color_ranges = {
            "Red": [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255])),
            ],
            "Green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
        }
        self.circle_target = CircleTargets()
        self.min_area_percentage_rectangle_threshold = 30  #矩形在视界中占比的最小百分比阈值

    def detect_circle_targets(self, frame: np.ndarray, target_color: Optional[str] = None) -> CircleTargets:
        """用于检测图像中指定颜色的圆形目标，并返回它们的坐标和半径等信息"""
        self.circle_target.clear()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        colors_to_detect = [target_color] if target_color else list(self.color_ranges.keys())
        for color_name in colors_to_detect:
            if color_name not in self.color_ranges:
                continue

            ranges = self.color_ranges[color_name]
            mask = None
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

            ellipses = transform.hough_ellipse(masked_edges, accuracy=20, threshold=250, min_size=30, max_size=300)

            # circles = cv2.HoughCircles(masked_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            #                     param1=50, param2=30, minRadius=5, maxRadius=50)


            if ellipses is not None and len(ellipses) > 0:
                ellipses = sorted(ellipses, key=lambda e: e[-1], reverse=True)
                for e in ellipses:
                    cy, cx, a, b, angle, acc = e
                    
                    cx, cy = int(round(cx)), int(round(cy))
                    a, b = int(round(a)), int(round(b))
                    angle_deg = float(angle)  # 弧度转度数（如果需要）
                    cy, cx, a, b = [int(round(x)) for x in best[1:5]]
                    # selected_ellipse = ((cx, cy), (2*a, 2*b), orientation)
                    radius = int(np.sqrt(a * b))
                    
                    target = self.circle_target.build_target(color=color_name, center=(cx, cy), radius=radius)
                    self.circle_target.add_target(target)

        return self.circle_target
