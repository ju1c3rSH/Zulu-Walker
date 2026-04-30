# -*- coding: utf-8 -*-
"""UV 点检测模块"""

from typing import Optional, Tuple, List
import cv2
import numpy as np


class UVDetector:
    """UV 点检测器"""

    def __init__(self):
        # 参数
        self.min_area: int = 5

        # UV 颜色范围
        self.uv_ranges: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.array([120, 10, 80]), np.array([160, 255, 255])),
            (np.array([90, 5, 90]), np.array([130, 220, 255])),
        ]

        # 检测状态
        self.is_detected: bool = False
        self.spot_center: Optional[Tuple[int, int]] = None

    def detect(
        self,
        frame: np.ndarray,
        search_contour: np.ndarray = None,
    ) -> Optional[Tuple[int, int]]:
        """
        检测 UV 点（使用亮度加权质心提高精度）

        Args:
            frame: BGR 格式的输入图像
            search_contour: 搜索轮廓，只在此轮廓内检测 UV 点

        Returns:
            UV 点坐标 (x, y) 或 None
        """
        if search_contour is None:
            return self._detect_full_frame(frame)
        else:
            return self._detect_in_contour(frame, search_contour)

    def _detect_full_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """在全图检测 UV 点"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._create_uv_mask(hsv)

        if mask is None:
            return None

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_area:
            return None

        # 使用亮度加权质心
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        M = cv2.moments(masked_gray, binaryImage=False)

        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'] + 0.5)
            cy = int(M['m01'] / M['m00'] + 0.5)
            return (cx, cy)

        return None

    def _detect_in_contour(
        self, frame: np.ndarray, search_contour: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """在指定轮廓内检测 UV 点"""
        x, y, w, h = cv2.boundingRect(search_contour)
        if w <= 0 or h <= 0:
            return None

        roi_frame = frame[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # 创建轮廓掩码
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = search_contour - [x, y]
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)

        # UV 颜色掩码
        uv_mask = self._create_uv_mask(roi_hsv)
        if uv_mask is None:
            return None

        contour_mask = cv2.bitwise_and(uv_mask, roi_mask)

        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_area:
            return None

        # 使用亮度加权质心
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray_roi, gray_roi, mask=contour_mask)
        M = cv2.moments(masked_gray, binaryImage=False)

        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'] + 0.5) + x
            cy = int(M['m01'] / M['m00'] + 0.5) + y
            return (cx, cy)

        return None

    def _create_uv_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """创建 UV 颜色掩码"""
        mask = None
        for lower, upper in self.uv_ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)
        return mask

    def set_min_area(self, min_area: int) -> None:
        """设置最小面积阈值"""
        self.min_area = min_area

    def reset(self) -> None:
        """重置检测状态"""
        self.is_detected = False
        self.spot_center = None
