# -*- coding: utf-8 -*-
"""
几何点工具类

提供二维点和三维点的常用操作。
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class Point2D:
    """二维点"""
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    def distance_to(self, other: 'Point2D') -> float:
        """计算到另一点的欧氏距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_origin(self) -> float:
        """计算到原点的距离"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def as_tuple(self) -> Tuple[float, float]:
        """转换为元组"""
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        """转换为整数元组"""
        return (int(round(self.x)), int(round(self.y)))

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> 'Point2D':
        """从元组创建"""
        return cls(t[0], t[1])

    def midpoint_to(self, other: 'Point2D') -> 'Point2D':
        """计算到另一点的中心点"""
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

    def angle_to(self, other: 'Point2D') -> float:
        """
        计算到另一点的角度

        Returns:
            角度（弧度），范围 [-π, π]
        """
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class Point3D:
    """三维点"""
    x: float
    y: float
    z: float

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def distance_to(self, other: 'Point3D') -> float:
        """计算到另一点的欧氏距离"""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def distance_to_origin(self) -> float:
        """计算到原点的距离"""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def as_tuple(self) -> Tuple[float, float, float]:
        """转换为元组"""
        return (self.x, self.y, self.z)

    def as_int_tuple(self) -> Tuple[int, int, int]:
        """转换为整数元组"""
        return (int(round(self.x)), int(round(self.y)), int(round(self.z)))

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Point3D':
        """从元组创建"""
        return cls(t[0], t[1], t[2])

    def to_2d(self, drop_z: bool = True) -> Point2D:
        """
        转换为二维点

        Args:
            drop_z: True 则丢弃 z，False 则投影到 xy 平面
        """
        return Point2D(self.x, self.y)

    def normalize(self) -> 'Point3D':
        """返回单位向量"""
        length = self.distance_to_origin()
        if length > 0:
            return self / length
        return Point3D(0, 0, 0)

    def dot(self, other: 'Point3D') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Point3D') -> 'Point3D':
        """叉积"""
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
