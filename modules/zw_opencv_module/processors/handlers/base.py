from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from framework.hal.interface import AIInference, Detection


class AbstractModelHandler(ABC):
    def __init__(self, ai: AIInference) -> None:
        self._ai = ai

    @abstractmethod
    def draw(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        ...
