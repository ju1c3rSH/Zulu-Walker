# -*- coding: utf-8 -*-
"""
焦距距离计算工具

根据相机内参和目标在图像中的像素尺寸，计算目标到相机的实际距离。
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class CameraIntrinsics:
    """相机内参"""
    focal_length_mm: float      # 焦距 (mm)
    sensor_width_mm: float      # 传感器宽度 (mm)
    sensor_height_mm: float     # 传感器高度 (mm)
    image_width: int            # 图像宽度 (像素)
    image_height: int           # 图像高度 (像素)

    @property
    def fx(self) -> float:
        """x 方向焦距 (像素)"""
        return self.focal_length_mm * self.image_width / self.sensor_width_mm

    @property
    def fy(self) -> float:
        """y 方向焦距 (像素)"""
        return self.focal_length_mm * self.image_height / self.sensor_height_mm

    @property
    def cx(self) -> float:
        """主点 x 坐标 (像素)"""
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        """主点 y 坐标 (像素)"""
        return self.image_height / 2.0


class FocalDistanceCalculator:
    """
    焦距距离计算器

    使用相似三角形原理计算目标到相机的距离。

    公式: distance = (focal_length_pixel * real_size) / pixel_size

    其中:
    - focal_length_pixel: 焦距对应的像素数
    - real_size: 目标的实际尺寸 (mm)
    - pixel_size: 目标在图像中的像素尺寸
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        初始化计算器

        Args:
            intrinsics: 相机内参
        """
        self.intrinsics = intrinsics

    def calculate_distance(
        self,
        real_size_mm: float,
        pixel_size: float,
        use_average_focal: bool = True
    ) -> float:
        """
        计算到目标的距离

        Args:
            real_size_mm: 目标的实际尺寸 (mm)
            pixel_size: 目标在图像中的像素尺寸
            use_average_focal: 是否使用平均焦距（默认 True）

        Returns:
            目标到相机的距离 (mm)
        """
        if pixel_size <= 0:
            return 0.0

        if use_average_focal:
            focal_pixel = (self.intrinsics.fx + self.intrinsics.fy) / 2.0
        else:
            focal_pixel = self.intrinsics.fx

        distance = (focal_pixel * real_size_mm) / pixel_size
        return distance

    def pixel_to_camera_coords(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_mm: float
    ) -> Tuple[float, float, float]:
        """
        将像素坐标转换为相机坐标系下的坐标

        相机坐标系定义:
        - 原点在相机光心
        - X 轴指向右
        - Y 轴指向下
        - Z 轴指向前（光轴方向）

        Args:
            pixel_x: 像素 x 坐标
            pixel_y: 像素 y 坐标
            distance_mm: 目标距离 (mm)

        Returns:
            (X, Y, Z) 相机坐标系下的坐标 (mm)
        """
        # 计算相对于主点的偏移
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy

        # 使用相似三角形计算 X, Y
        Z = distance_mm
        X = dx * Z / self.intrinsics.fx
        Y = dy * Z / self.intrinsics.fy

        return (X, Y, Z)

    def camera_coords_to_pixel(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[float, float]:
        """
        将相机坐标系下的坐标转换为像素坐标

        Args:
            x: 相机坐标系 X (mm)
            y: 相机坐标系 Y (mm)
            z: 相机坐标系 Z (mm)

        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        if z <= 0:
            return (self.intrinsics.cx, self.intrinsics.cy)

        pixel_x = x * self.intrinsics.fx / z + self.intrinsics.cx
        pixel_y = y * self.intrinsics.fy / z + self.intrinsics.cy

        return (pixel_x, pixel_y)

    def calculate_angular_offset(
        self,
        pixel_x: float,
        pixel_y: float
    ) -> Tuple[float, float]:
        """
        计算像素点相对于光轴的角度偏移

        Args:
            pixel_x: 像素 x 坐标
            pixel_y: 像素 y 坐标

        Returns:
            (angle_x, angle_y) 角度偏移 (弧度)
        """
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy

        angle_x = math.atan(dx / self.intrinsics.fx)
        angle_y = math.atan(dy / self.intrinsics.fy)

        return (angle_x, angle_y)
