import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ShapeType(Enum):
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    QUAD = "quad"
    UNKNOWN = "unknown"

@dataclass
class CircleTargetItem:
    index: int
    center_coordinates: Tuple[int, int]
    radius: float
    area: float
    shape_type: ShapeType = ShapeType.UNKNOWN
    contour_points: Optional[np.ndarray] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    color: Optional[str] = None
    major_axis: Optional[float] = None
    minor_axis: Optional[float] = None
    quad_points: Optional[np.ndarray] = None  # 四边形顶点，用于调试绘制
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "center_coordinates": self.center_coordinates,
            "radius": self.radius,
            "area": self.area,
            "bounding_box": self.bounding_box,
            "color": self.color
        }
    def __eq__(self, other):
        """定义相等性比较（基于index）"""
        if not isinstance(other, CircleTargetItem):
            return False
        return self.index == other.index
    
    def __hash__(self):
        """使对象可哈希（用于集合）"""
        return hash(self.index)   
@dataclass
class CircleTargets:
    targets: List[CircleTargetItem] = field(default_factory=list)
    number: int = 0

    def build_target(self, color: str, center: Tuple[int, int], radius: float, area: float = 0.0, contour_points: Optional[np.ndarray] = None, bounding_box: Optional[Tuple[int, int, int, int]] = None) -> None:
        """添加一个圆形目标

        Args:
            color: 目标颜色
            center: 圆心坐标 (x, y)
            radius: 圆半径
            area: 圆面积
            contour_points: 圆的轮廓点（可选）
            bounding_box: 圆的边界框 (x, y, w, h)（可选）
        """
        return CircleTargetItem(
            index=self.number,
            center_coordinates=center,
            radius=radius,
            area=area,
            contour_points=contour_points,
            bounding_box=bounding_box,
            color=color
        )


    def add_target(self,target: CircleTargetItem) -> None:
        """添加一个圆形目标

        Args:
            target: CircleTargetItem对象
        """
        target.index = len(self.targets) #unique index for every target
        self.targets.append(target)
        self.number = len(self.targets)

    def clear(self) -> None:
        """清空所有目标"""
        self.targets.clear()
        self.number = 0

    def get_target_by_index(self, index: int) -> Optional[CircleTargetItem]:
        """根据索引获取目标

        Args:
            index: 目标索引

        Returns:
            CircleTargetItem对象或None
        """
        for target in self.targets:
            if target.index == index:
                return target
        return None

    def get_all_centers(self) -> List[Tuple[int, int]]:
        """获取所有目标的中心点坐标

        Returns:
            List[Tuple[int, int]]: 所有目标的中心点坐标列表
        """
        return [target.center_coordinates for target in self.targets]

    def to_dict(self) -> dict:
        """将CircleTargets对象转换为字典格式"""
        return {
            "number": self.number,
            "targets": [target.to_dict() for target in self.targets]
        }
