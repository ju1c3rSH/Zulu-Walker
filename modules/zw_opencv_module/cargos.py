import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ShapeType(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    UNKNOWN = "unknown"
    
class ShapeMapping(Enum):
    RECTANGLE = 4
    TRIANGLE = 3
    
@dataclass
class CargoItem:
    index: int
    center_coordinates: Tuple[int, int]
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    area: Optional[float] = None
    contour_points: Optional[np.ndarray] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    aspect_ratio: Optional[float] = None
    shape_type: ShapeType = ShapeType.UNKNOWN
    color: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "center_coordinates": self.center_coordinates,
            "width": self.width,
            "height": self.height,
            "radius": self.radius,
            "area": self.area,
            "bounding_box": self.bounding_box,
            "aspect_ratio": self.aspect_ratio,
            "shape_type": self.shape_type.value,
            "color": self.color
        }
    
@dataclass
class Cargos:
    shape: ShapeType = ShapeType.UNKNOWN
    number: int = 0 
    payload: List[CargoItem] = field(default_factory=list)
    
    def add_cargo(self, cargo: CargoItem) -> None:
        """添加单个货物"""
        self.payload.append(cargo)
        self.number = len(self.payload)
    
    def add_cargos(self, cargos: List[CargoItem]) -> None:
        """添加多个货物"""
        self.payload.extend(cargos)
        self.number = len(self.payload)
    
    def clear(self) -> None:
        """清空所有数据"""
        self.shape = ShapeType.UNKNOWN
        self.number = 0
        self.payload.clear()
    
    def get_cargo_by_index(self, index: int) -> Optional[CargoItem]:
        """根据索引获取货物"""
        for cargo in self.payload:
            if cargo.index == index:
                return cargo
        return None
    
    def get_all_centers(self) -> List[Tuple[int, int]]:
        """获取所有货物的中心点坐标"""
        return [cargo.center_coordinates for cargo in self.payload]
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "shape": self.shape.value,
            "number": self.number,
            "payload": [cargo.to_dict() for cargo in self.payload]
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)    