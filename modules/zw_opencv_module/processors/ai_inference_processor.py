from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from framework.hal.interface import AIInference, Detection

from .registry import register_processor
from .base import Processor, VisionResult
from .handlers.registry import ModelHandlerRegistry

logger = logging.getLogger(__name__)


@register_processor("AIInferenceProcessor")
class AIInferenceProcessor(Processor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._ai: Optional[AIInference] = None
        self._handler = None

    def set_ai(self, ai: AIInference) -> None:
        self._ai = ai

    def _ensure_handler(self):
        if self._ai is None:
            return None
        mt = self._ai.model_type
        if self._handler is None or getattr(self._handler, "_model_type", None) != mt:
            self._handler = ModelHandlerRegistry.get(mt, self._ai)
            self._handler._model_type = mt
        return self._handler

    def process(
        self, frame: np.ndarray, context: dict = None
    ) -> VisionResult:
        if self._ai is None:
            return VisionResult(
                self.name, success=False,
                error_message="AI not available",
            )
        if not self._ai.loaded:
            return VisionResult(
                self.name, success=False,
                error_message="AI model not loaded",
            )

        try:
            input_img = frame
            detections = self._ai.detect(input_img)
        except Exception as e:
            logger.error("AIInferenceProcessor detect failed: %s", e)
            return VisionResult(
                self.name, success=False, error_message=str(e),
            )

        segment_dicts = []
        for d in detections:
            if d.mask_stats is not None and d.mask_stats.area_px > 0:
                segment_dicts.append({
                    "class_id": d.class_id,
                    "center_x": d.mask_stats.center_x,
                    "center_y": d.mask_stats.center_y,
                    "area_px": d.mask_stats.area_px,
                })

        return VisionResult(
            self.name,
            result_data={
                "detections": detections,
                "segments": segment_dicts,
            },
            success=True,
        )

    def draw_result(
        self, frame: np.ndarray, result: VisionResult
    ) -> np.ndarray:
        if not result.success:
            return frame

        detections = result.result_data.get("detections", [])
        if not detections:
            return frame

        handler = self._ensure_handler()
        if handler is None:
            return frame

        return handler.draw(frame, detections)

    def release(self) -> None:
        self._handler = None
