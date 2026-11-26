"""AI inference contracts shared by every platform backend.

Coordinate and unit conventions used by `Detection` and friends:

* `x`/`y` are the top-left corner of the box in **source frame pixels**,
  `w`/`h` its width and height.  A backend must undo its own letterbox
  scaling before returning.
* `score` is a probability in `[0.0, 1.0]`.
* `class_id` indexes the model label list.  The label order is part of the
  model contract and must be identical across backends.
* `seg_mask` is a single channel `uint8` mask in **box local coordinates**
  (same size as the box).  `MaskStats.center_x`/`center_y` are reported in
  source frame coordinates.
* `keypoints` are in source frame coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, TypedDict, Unpack, runtime_checkable


import numpy as np

# Detection filter thresholds, in source frame pixels.  They are
# absolute pixels and must be re-tuned whenever the frame resolution
# changes.
MIN_AREA = 100
MAX_AREA = 20000
MAX_RATIO = 2.0
MARGIN = 4

# Stable failure identifiers.  Callers and tests branch on these instead of
# matching on message text.
E_MODEL_NOT_LOADED = "E_MODEL_NOT_LOADED"
E_MODEL_TYPE_UNSUPPORTED = "E_MODEL_TYPE_UNSUPPORTED"
E_MODEL_LOAD_FAILED = "E_MODEL_LOAD_FAILED"
E_OUTPUT_LAYOUT_MISMATCH = "E_OUTPUT_LAYOUT_MISMATCH"
E_INFER_FAILED = "E_INFER_FAILED"
E_INVALID_INPUT = "E_INVALID_INPUT"
E_ENGINE_UNAVAILABLE = "E_ENGINE_UNAVAILABLE"
E_CONTRACT_MISSING = "E_CONTRACT_MISSING"
E_DIALECT_UNSUPPORTED = "E_DIALECT_UNSUPPORTED"
E_SWITCH_TIMEOUT = "E_SWITCH_TIMEOUT"


class AIError(Exception):
    """Base class for AI layer failures.

    `code` is one of the `E_*` constants above so that callers can branch
    on the failure kind without parsing message text.
    """

    code = E_INFER_FAILED

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class ModelConfigError(AIError):
    """The requested model/backend combination cannot be honoured.

    Configuration errors are never downgraded to an empty result.  Running
    the wrong post-processing on a model output does not raise, it silently
    produces wrong boxes, which is more dangerous than a visible failure.
    """

    code = E_MODEL_TYPE_UNSUPPORTED


class ModelLoadError(AIError):
    """The model file could not be loaded by the selected runtime class."""

    code = E_MODEL_LOAD_FAILED


class ContractError(AIError):
    """Model output does not match the declared output layout."""

    code = E_OUTPUT_LAYOUT_MISMATCH


class ManifestError(AIError):
    """The model manifest does not satisfy the engine contract.

    Raised when a required manifest field is absent, when the manifest
    declares no artifact for the selected engine, or when the artifact
    file name disagrees with the configured name.  Kept apart from
    `ContractError`, which reports a mismatch between the declared
    output layout and the produced output.
    """

    code = E_CONTRACT_MISSING

    def __init__(self, message: str = "", code: str = "") -> None:
        super().__init__(message, code)


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


def filter_detections(
    detections: list[Detection],
    *,
    min_area: int = MIN_AREA,
    max_area: int = MAX_AREA,
    max_ratio: float = MAX_RATIO,
    margin: int = MARGIN,
    image_width: int = 640,
    image_height: int = 640,
    on_discard: Optional[Callable[[Detection, str], None]] = None,
) -> list[Detection]:
    """Drop detections that are too small, too large, too distorted or too
    close to the frame border.

    `on_discard` receives every removed detection together with the reason
    (`area_too_small`, `area_too_large`, `aspect_ratio`, `edge_proximity`)
    so the caller can report what was filtered instead of only how many.
    """
    filtered: list[Detection] = []
    for obj in detections:
        area = obj.w * obj.h
        if area < min_area:
            if on_discard:
                on_discard(obj, "area_too_small")
            continue
        if area > max_area:
            if on_discard:
                on_discard(obj, "area_too_large")
            continue
        ratio = max(obj.w, obj.h) / max(min(obj.w, obj.h), 1)
        if ratio > max_ratio:
            if on_discard:
                on_discard(obj, "aspect_ratio")
            continue
        if (obj.x < margin or obj.y < margin
                or obj.x + obj.w > image_width - margin
                or obj.y + obj.h > image_height - margin):
            if on_discard:
                on_discard(obj, "edge_proximity")
            continue
        filtered.append(obj)
    return filtered


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

    @property
    def last_error_code(self) -> str:
        """Code of the most recent failure, empty while the backend is healthy."""
        ...

    @property
    def error_count(self) -> int:
        """Number of failures recorded since the last successful inference."""
        ...

    @property
    def degraded(self) -> bool:
        """True while the backend runs but does not produce valid output."""
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
