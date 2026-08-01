from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from framework.hal.interface import AIInference, Detection
from framework.hal.interface.ai import filter_detections
from utils.log_util import log_print

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
        # Last wall-clock time the filter aggregate was logged (once/second).
        self._last_discard_log_ts: float = 0.0

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
        if frame is None:
            return VisionResult(
                self.name, success=False,
                error_message="frame is None",
            )

        try:
            camera = (context or {}).get("camera")
            raw = getattr(camera, "last_raw", None)
            if raw is not None:
                detections = self._ai.detect(raw, _raw=True)
                h, w = raw.height(), raw.width()
            else:
                detections = self._ai.detect(frame)
                h, w = frame.shape[:2]
        except Exception as e:
            logger.error("AIInferenceProcessor detect failed: %s", e)
            return VisionResult(
                self.name, success=False, error_message=str(e),
            )

        # Filter detections before any downstream consumption
        pre_count = len(detections)
        discard_log: list[tuple[Detection, str]] = []
        detections = filter_detections(
            detections,
            image_width=w,
            image_height=h,
            on_discard=lambda obj, reason: discard_log.append((obj, reason)),
        )
        # Log only a once-per-second aggregate instead of one line per discarded
        # object every frame (the per-item spam was pure log noise).
        if discard_log:
            now = time.monotonic()
            if now - self._last_discard_log_ts >= 1.0:
                self._last_discard_log_ts = now
                log_print(
                    f"filter_detections: {pre_count} -> {len(detections)} "
                    f"(removed {len(discard_log)})"
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
                "infer_fps": getattr(self._ai, "infer_fps", 0.0),
                "infer_avg_ms": getattr(self._ai, "infer_avg_ms", 0.0),
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
