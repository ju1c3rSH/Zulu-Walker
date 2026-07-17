from typing import Optional

import cv2
import numpy as np
from .base import Processor, VisionResult
from ..detectors.cargo_detector import CargoDetector
from ..models import Color


_COLOR_BGR = {
    Color.RED: (0, 0, 255),
    Color.GREEN: (0, 255, 0),
    Color.BLUE: (255, 0, 0),
}
_COLOR_NAMES = {
    Color.RED: "RED",
    Color.GREEN: "GREEN",
    Color.BLUE: "BLUE",
}


class TrackCargoProcessor(Processor):
    """物料跟踪处理器。

    输出 result_data:
      percent_error_x: int, [-5000, 5000], 0=画面中心, 负=偏左, 正=偏右
      percent_error_y: int, [-5000, 5000], 0=画面中心, 负=偏上, 正=偏下
      coordinate: (cx, cy) 像素坐标
      radius: float, 检测到的圆半径（像素）
    """

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
        pe_x = int(((cx - frame_w / 2.0) / (frame_w / 2.0)) * 5000.0)
        pe_y = int(((cy - frame_h / 2.0) / (frame_h / 2.0)) * 5000.0)

        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": True,
                "percent_error_x": pe_x,
                "percent_error_y": pe_y,
                "coordinate": (cx, cy),
                "radius": item.radius,
                "target_color": target_color,
                "is_predicted": item.is_predicted,
                "confidence": item.confidence,
            },
        )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame

        data = result.result_data
        target_found = data.get("target_found", False)
        if not target_found:
            return frame

        is_predicted = data.get("is_predicted", False)
        if is_predicted:
            return frame

        confidence = data.get("confidence", 100.0)

        coordinate = data.get("coordinate")
        radius = data.get("radius")
        target_color = data.get("target_color")
        pe_x = data.get("percent_error_x", 0)
        pe_y = data.get("percent_error_y", 0)

        if coordinate is None:
            return frame

        cx, cy = coordinate
        color_bgr = _COLOR_BGR.get(target_color, (0, 255, 255))

        if radius is not None and radius > 0:
            cv2.circle(frame, (int(cx), int(cy)), int(radius), color_bgr, 2)

        cv2.circle(frame, (int(cx), int(cy)), 4, color_bgr, -1)

        label = _COLOR_NAMES.get(target_color, "CARGO")
        if confidence < 100.0:
            label = f"{label}(H)"
        cv2.putText(frame, label, (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        cv2.putText(frame, f"({int(cx)},{int(cy)})", (int(cx) + 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w // 2, frame_h // 2
        cv2.line(frame, (center_x, center_y), (int(cx), int(cy)), (0, 255, 255), 1,
                 cv2.LINE_AA)

        info = f"ERR x:{pe_x:+d} y:{pe_y:+d} conf:{confidence:.0f}"
        cv2.putText(frame, info, (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame

    def set_target_color(self, color: Optional[Color]) -> None:
        self.target_color = color
