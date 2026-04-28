# -*- coding: utf-8 -*-
"""
相机焦距与距离计算工具

核心公式：
- 像素焦距: fx = focal_length_mm * image_width / sensor_width_mm
- 距离计算: distance = (real_size * focal_length_px) / pixel_size
- 相机坐标: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
"""

from dataclasses import dataclass
from typing import Tuple

# 目标实际尺寸 (mm)
reference_size_dict = {
    "quad": (290.0, 200.0, 245.0),  # mm (宽, 高, 平均)
}

@dataclass
class CameraIntrinsics:
    """相机内参"""
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width: int
    image_height: int

    # 计算得到的像素焦距
    fx: float = 0.0  # x方向焦距(像素)
    fy: float = 0.0  # y方向焦距(像素)
    cx: float = 0.0  # 主点x坐标
    cy: float = 0.0  # 主点y坐标

    def __post_init__(self):
        self.fx = self.focal_length_mm * self.image_width / self.sensor_width_mm
        self.fy = self.focal_length_mm * self.image_height / self.sensor_height_mm
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0


class FocalDistanceCalculator:
    """焦距距离计算器"""

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def calculate_distance(self, real_size_mm: float, pixel_size: float) -> float:
        """
        根据目标实际尺寸和像素尺寸计算距离

        Args:
            real_size_mm: 目标实际尺寸 (mm)
            pixel_size: 目标在图像中的像素尺寸

        Returns:
            目标到相机的距离 (mm)
        """
        if pixel_size <= 0:
            return 0.0
        return (real_size_mm * self.intrinsics.fx) / pixel_size

    def pixel_to_camera_coords(self, pixel_x: float, pixel_y: float,
                                distance_mm: float) -> Tuple[float, float, float]:
        """
        将像素坐标转换为相机坐标系下的坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            distance_mm: 目标距离 (mm)

        Returns:
            (X, Y, Z) 相机坐标系下的坐标 (mm)
            X: 相机右侧为正
            Y: 相机下方为正
            Z: 相机前方为正（深度）
        """
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        X = (pixel_x - cx) * distance_mm / fx
        Y = (pixel_y - cy) * distance_mm / fy
        Z = distance_mm

        return (X, Y, Z)
