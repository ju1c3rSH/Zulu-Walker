from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from framework.inference.facade import DEFAULT_SWITCH_TIMEOUT_MS, AIFacade
from framework.inference.runtime import FrameSpec

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_ID = "onnxruntime"
DEFAULT_FRAME_SPEC = FrameSpec(color="bgr", dtype="uint8", layout="hwc")

__all__ = ["DEFAULT_ENGINE_ID", "LinuxAI"]


class LinuxAI(AIFacade):
    """Desktop AI backend: the inference kernel owns the engine and the contract.

    The platform contributes exactly one thing - the shape of the frames the
    camera delivers.  Everything else (the engine, the artifact, the
    preprocessing and the task dialect) comes from the manifest and the
    inference kernel, so the desktop and the device share one contract.

    An engine that cannot be brought up is a startup failure, not a silent
    downgrade to a stub: a system that runs an order of magnitude slower
    without a visible cause is worse than one that refuses to start.
    """

    def __init__(
        self,
        engine_id: str = DEFAULT_ENGINE_ID,
        *,
        switch_timeout_ms: int = DEFAULT_SWITCH_TIMEOUT_MS,
        frame: Optional[Any] = None,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(
            engine_id,
            switch_timeout_ms=switch_timeout_ms,
            frame_spec=_frame_spec(frame),
            engine_factory=engine_factory,
        )


def _frame_spec(frame: Any) -> FrameSpec:
    """Build the platform frame contract from the configuration."""
    if isinstance(frame, FrameSpec):
        return frame
    if isinstance(frame, dict):
        return FrameSpec(
            color=str(frame.get("color", DEFAULT_FRAME_SPEC.color)),
            dtype=str(frame.get("dtype", DEFAULT_FRAME_SPEC.dtype)),
            layout=str(frame.get("layout", DEFAULT_FRAME_SPEC.layout)),
        )
    return DEFAULT_FRAME_SPEC
