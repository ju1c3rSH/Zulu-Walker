from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from framework.hal.platforms.frame_spec import frame_spec_from_config
from framework.inference.facade import DEFAULT_SWITCH_TIMEOUT_MS, AIFacade

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_ID = "onnxruntime"

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
        runtime_options: Optional[dict] = None,
    ) -> None:
        super().__init__(
            engine_id,
            switch_timeout_ms=switch_timeout_ms,
            frame_spec=frame_spec_from_config(frame),
            engine_factory=engine_factory,
            runtime_options=runtime_options,
        )
