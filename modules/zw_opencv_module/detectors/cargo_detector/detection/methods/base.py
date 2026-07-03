from abc import ABC, abstractmethod
from typing import Optional
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