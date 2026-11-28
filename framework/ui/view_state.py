"""What a layout is given to draw, independent of where the data came from.

The vision module knows about pipelines, cameras and AI results; a layout
knows about a header, boxes and a fault banner.  This type is the seam: the
module fills it, the layout reads it, and neither imports the other.

Keeping the detections as a tuple of HAL ``Detection`` values (not raw
dicts) is what lets a layout be unit-tested with plain objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

__all__ = ["ViewState"]


@dataclass(frozen=True, slots=True)
class ViewState:
    """One pipeline's view for one frame."""

    pipeline_id: str = ""
    fps: float = 0.0

    #: Detections for this frame; empty when the model found nothing or failed.
    detections: Tuple[Any, ...] = ()

    #: Class names; ``detections[i].class_id`` indexes this.
    labels: Tuple[str, ...] = ()

    #: False until a model is loaded; the layout shows a fault banner instead.
    ai_ok: bool = False

    #: Non-empty when inference failed.  A failed run must look different from
    #: an empty scene, otherwise a broken model is indistinguishable from a
    #: picture of nothing.
    error_code: str = ""

    #: True while the camera is reconnecting.
    reconnecting: bool = False

    #: Whether to draw the detection list beside the boxes.
    list_enabled: bool = True

    #: Wall clock, refreshed once per second by the caller: reading it every
    #: frame is the cost, drawing three strings is not.
    clock: str = ""

    def label_of(self, class_id: int) -> str:
        """The class name for a detection, or a stable placeholder."""
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"cls{class_id}"
