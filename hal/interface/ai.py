from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class Keypoint:
    x: float
    y: float
    score: float = 0.0
    id: int = -1
    visibility: float = 1.0
    name: str = ""


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    score: float
    label: str = ""
    angle: Optional[float] = None
    keypoints: list[Keypoint] = field(default_factory=list)
    mask_index: int = -1


@runtime_checkable
class AIInference(Protocol):
    @property
    def models(self) -> list[str]:
        ...

    @property
    def active_model(self) -> str:
        ...

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        ...

    def remove(self, nick_name: str) -> None:
        ...

    def switch(self, nick_name: str) -> bool:
        ...

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        ...

    def unload(self) -> None:
        ...

    @property
    def loaded(self) -> bool:
        ...

    @property
    def input_width(self) -> int:
        ...

    @property
    def input_height(self) -> int:
        ...

    @property
    def labels(self) -> list[str]:
        ...

    @property
    def model_path(self) -> str:
        ...

    def detect(
        self, frame: np.ndarray, conf_th: float = 0.5, iou_th: float = 0.45
    ) -> list[Detection]:
        ...

    def classify(self, frame: np.ndarray, top_k: int = 1) -> list[tuple[int, float]]:
        ...

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        ...

    def __enter__(self):
        ...

    def __exit__(self, *args):
        ...
