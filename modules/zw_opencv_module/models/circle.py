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
    quad_points: Optional[np.ndarray] = None

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
        if not isinstance(other, CircleTargetItem):
            return False
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)


@dataclass
class CircleTargets:
    targets: List[CircleTargetItem] = field(default_factory=list)
    number: int = 0

    def build_target(self, color: str, center: Tuple[int, int], radius: float,
                     area: float = 0.0,
                     contour_points: Optional[np.ndarray] = None,
                     bounding_box: Optional[Tuple[int, int, int, int]] = None) -> CircleTargetItem:
        return CircleTargetItem(
            index=self.number,
            center_coordinates=center,
            radius=radius,
            area=area,
            contour_points=contour_points,
            bounding_box=bounding_box,
            color=color
        )

    def add_target(self, target: CircleTargetItem) -> None:
        target.index = len(self.targets)
        self.targets.append(target)
        self.number = len(self.targets)

    def clear(self) -> None:
        self.targets.clear()
        self.number = 0

    def get_target_by_index(self, index: int) -> Optional[CircleTargetItem]:
        for target in self.targets:
            if target.index == index:
                return target
        return None

    def get_all_centers(self) -> List[Tuple[int, int]]:
        return [target.center_coordinates for target in self.targets]

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "targets": [target.to_dict() for target in self.targets]
        }
