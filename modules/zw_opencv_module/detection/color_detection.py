# -*- coding: utf-8 -*-
"""颜色检测模块"""

from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np


class ColorDetector:
    """颜色检测器"""

    def __init__(self):
        # 颜色范围定义（HSV）
        self.color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "Red": [
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            "Green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            "UV": [
                (np.array([120, 10, 80]), np.array([160, 255, 255])),
                (np.array([90, 5, 90]), np.array([130, 220, 255])),
            ],
        }

        # H 范围（用于轮廓颜色检测）
        self.color_h_ranges = {
            "Red": (0, 15, 165, 180),
            "Green": (40, 80, 0, 0),
            "Blue": (100, 130, 0, 0),
        }

        # 颜色阈值
        self.color_s_min: int = 60
        self.color_v_min: int = 50
        self.debug_color: bool = False

    def get_color_mask(self, hsv: np.ndarray, color_name: str) -> Optional[np.ndarray]:
        """
        获取指定颜色的掩码

        Args:
            hsv: HSV 格式图像
            color_name: 颜色名称

        Returns:
            颜色掩码或 None
        """
        if color_name not in self.color_ranges:
            return None

        ranges = self.color_ranges[color_name]
        mask = None

        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        return mask

    def detect_contour_color(
        self, hsv: np.ndarray, contour: np.ndarray, img_shape: tuple
    ) -> Optional[str]:
        """
        检测轮廓内部的主要颜色

        Args:
            hsv: HSV 格式图像
            contour: 轮廓
            img_shape: 图像形状 (h, w)

        Returns:
            颜色名称或 None
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mean_val = cv2.mean(hsv, mask=mask)[:3]
        h, s, v = mean_val

        if self.debug_color:
            print(f"[ColorDebug] HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")

        # 低饱和度：黑色
        if s < self.color_s_min:
            return "Black"

        # 红色判断（跨越 0 度）
        h_low, h_high, h_low2, h_high2 = self.color_h_ranges["Red"]
        if (h_low <= h < h_high) or (h_low2 <= h <= h_high2):
            return "Red"

        # 绿色判断
        h_low, h_high, _, _ = self.color_h_ranges["Green"]
        if h_low <= h < h_high:
            return "Green"

        # 蓝色判断
        h_low, h_high, _, _ = self.color_h_ranges["Blue"]
        if h_low <= h < h_high:
            return "Blue"

        return None

    def get_uv_ranges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取 UV 颜色范围"""
        return self.color_ranges.get("UV", [])

    def set_debug(self, enabled: bool) -> None:
        """设置调试模式"""
        self.debug_color = enabled