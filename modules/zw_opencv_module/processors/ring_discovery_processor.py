import cv2
import numpy as np
from typing import Optional
from .base import Processor, VisionResult, ColorTrackable
from ..models.color import Color
from ..detectors.ring_detector import RingDetector


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


class RingDiscoveryProcessor(Processor):
    """色环发现处理器。

    输出 result_data:
      percent_error_x: int, [-5000, 5000], 0=画面中心, 负=偏左, 正=偏右
      percent_error_y: int, [-5000, 5000], 0=画面中心, 负=偏上, 正=偏下
      coordinate: 色环中心像素坐标
      confidence: float 置信度
      target_color: Color 当前搜索的颜色
    """

    def __init__(self, name: str = "ring_discovery"):
        super().__init__(name)
        self.detector = RingDetector()
        self.target_color: Optional[Color] = None

    def set_target_color(self, color: Optional[Color]):
        self.target_color = color

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        if self.target_color is None:
            return self._no_target()

        ring = self.detector.detect_ring(frame, self.target_color)

        if ring is None:
            return self._no_target()

        cx, cy = ring.coordinate
        h, w = frame.shape[:2]
        half = max(w, h) / 2.0
        pe_x = int(((cx - w / 2.0) / half) * 5000.0)
        pe_y = int(((cy - h / 2.0) / half) * 5000.0)

        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": True,
                "percent_error_x": pe_x,
                "percent_error_y": pe_y,
                "coordinate": ring.coordinate,
                "confidence": ring.confidence,
                "target_color": self.target_color,
            },
        )

    def _no_target(self) -> VisionResult:
        return VisionResult(
            task_name=self.name,
            success=True,
            result_data={
                "target_found": False,
                "percent_error_x": 0,
                "percent_error_y": 0,
                "coordinate": None,
                "confidence": 0.0,
            },
        )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame

        data = result.result_data
        target_found = data.get("target_found", False)
        if not target_found:
            return frame

        coordinate = data.get("coordinate")
        target_color = data.get("target_color")
        confidence = data.get("confidence", 0)
        pe_x = data.get("percent_error_x", 0)
        pe_y = data.get("percent_error_y", 0)

        if coordinate is None:
            return frame

        cx, cy = coordinate
        color_bgr = _COLOR_BGR.get(target_color, (0, 255, 255))

        cv2.circle(frame, (int(cx), int(cy)), 4, color_bgr, -1)

        label = _COLOR_NAMES.get(target_color, "RING")
        cv2.putText(frame, f"RING {label} {confidence:.0f}%",
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        cv2.putText(frame, f"({int(cx)},{int(cy)})", (int(cx) + 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w // 2, frame_h // 2
        cv2.line(frame, (center_x, center_y), (int(cx), int(cy)), (0, 255, 255), 1,
                 cv2.LINE_AA)

        info = f"ERR x:{pe_x:+d} y:{pe_y:+d}"
        cv2.putText(frame, info, (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame
