from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, TypedDict, Unpack, runtime_checkable

import numpy as np


class DetectKwargs(TypedDict, total=False):
    conf_th: float
    iou_th: float


class ClassifyKwargs(TypedDict, total=False):
    top_k: int


@dataclass
class Keypoint:
    x: float
    y: float
    score: float = 0.0
    id: int = -1
    visibility: float = 1.0
    name: str = ""


@dataclass
class MaskStats:
    center_x: float = 0.0
    center_y: float = 0.0
    area_px: int = 0


@dataclass
class SegmentResult:
    class_id: int
    center_x: float
    center_y: float
    area_px: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    score: float
    detection: Detection = field(default=None)  # type: ignore[assignment]


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
    seg_mask: Optional[np.ndarray] = None
    mask_stats: Optional[MaskStats] = None


@runtime_checkable
class AIInference(Protocol):
    @property
    def models(self) -> list[str]:
        ...

    @property
    def active_model(self) -> str:
        ...

    @property
    def model_type(self) -> str:
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
        self, frame: np.ndarray,
        **kwargs: Unpack[DetectKwargs]
    ) -> list[Detection]:
        ...

    def segment(
        self, frame: np.ndarray,
        **kwargs: Unpack[DetectKwargs]
    ) -> list[SegmentResult]:
        ...

    def classify(
        self, frame: np.ndarray,
        **kwargs: Unpack[ClassifyKwargs]
    ) -> list[tuple[int, float]]:
        ...

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        ...

    def __enter__(self):
        ...

    def __exit__(self, *args):
        ...
