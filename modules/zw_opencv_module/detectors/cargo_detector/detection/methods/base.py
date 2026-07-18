from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from .....models.color import Color
from .....models.cargo import CargoItem


class BaseDetectionMethod(ABC):
    def __init__(self, name: str = "base_detection", detector=None):
        self.name = name
        self.detector = detector

    @abstractmethod
    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[CargoItem]:
        pass

    def _store_cargo_meta(self, target_color: Color,
                          center: Tuple[float, float], radius: float):
        area = np.pi * radius * radius
        if hasattr(self.detector, '_last_cargo_meta'):
            self.detector._last_cargo_meta[target_color] = {
                'center': center,
                'outer_radius': radius,
                'area': area,
            }