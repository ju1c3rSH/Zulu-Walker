from typing import Optional

import cv2
import numpy as np
from .base import Processor, VisionResult
from ..detectors.cargo_detector import CargoDetector
from ..models import Color


class TrackCargoProcessor(Processor):

    def __init__(self, name: str = "cargo_detect"):
        super().__init__(name)
        self.detector = CargoDetector(name)
        self._log_interval = 300
        self.target_color: Optional[Color] = None

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        ctx_snapshot = self.get("mission_ctx")
        if ctx_snapshot is None:
            return VisionResult(task_name=self.name, success=False,
                                error_message="mission_ctx not registered")

        cargo_set, current_batch, target_color = ctx_snapshot
        if target_color is None:
            return VisionResult(task_name=self.name, success=False,
                                error_message="target_color is None")

        item = self.detector.detect_cargo(frame, target_color)
        if item is None or item.coordinate is None:
            return VisionResult(task_name=self.name, success=False)

        cx, cy = item.coordinate
        if cargo_set:
            matched = cargo_set.get_available_by_color_batch(target_color, current_batch)
            if matched:
                matched.update_position((cx, cy))

        frame_h, frame_w = frame.shape[:2]
        pe_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
        pe_y = (cy - frame_h / 2.0) / (frame_h / 2.0)

        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": True,
                "percent_error_x": pe_x,
                "percent_error_y": pe_y,
                "coordinate": (cx, cy),
            },
        )

    def draw_result(self, frame, result):
        return super().draw_result(frame, result)

    def set_target_color(self, color: Optional[Color]) -> None:
        self.target_color = color
