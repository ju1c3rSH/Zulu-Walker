from abc import ABC, abstractmethod
import numpy as np

from .....models.circle import CircleTargetItem


class DetectionMethod(ABC):
    def __init__(self, detector):
        self.detector = detector

    @abstractmethod
    def detect(self, frame: np.ndarray, target_color: str) -> None:
        pass
