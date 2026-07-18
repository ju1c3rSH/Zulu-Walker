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

        self._draw_ring_meta_overlay(frame, target_color, coordinate)

        return frame

    def _draw_ring_meta_overlay(self, frame: np.ndarray, target_color, coordinate):
        meta = self.detector._last_ring_meta.get(target_color)
        if meta is None:
            return

        cx, cy = coordinate
        outer_r = meta.get('outer_radius', 0) or 0
        axes = meta.get('axes')
        angle = meta.get('angle', 0) or 0
        area = meta.get('area') or 0
        color_bgr = _COLOR_BGR.get(target_color, (0, 255, 255))

        if axes is not None and axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(frame, (int(cx), int(cy)),
                        (int(axes[0]), int(axes[1])), angle,
                        0, 360, color_bgr, 2)
        elif outer_r > 0:
            cv2.circle(frame, (int(cx), int(cy)),
                       int(outer_r), color_bgr, 2)

        hsv_mean = self._compute_ring_hsv(frame, (cx, cy), outer_r)
        text_x = int(cx) + 10
        line_h = 18

        if hsv_mean is not None:
            cv2.putText(frame,
                        f"H:{hsv_mean[0]:.0f} S:{hsv_mean[1]:.0f} V:{hsv_mean[2]:.0f}",
                        (text_x, int(cy) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if area > 0:
            cv2.putText(frame, f"Area: {area:.0f}",
                        (text_x, int(cy) + 22 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        method = meta.get('method', '')
        if method:
            cv2.putText(frame, f"Method: {method}",
                        (text_x, int(cy) + 22 + line_h * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    @staticmethod
    def _compute_ring_hsv(frame: np.ndarray, center, outer_r):
        if outer_r <= 0:
            return None
        cx, cy = int(center[0]), int(center[1])
        r = int(outer_r)
        pad = 4
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(frame.shape[1], cx + r + pad)
        y2 = min(frame.shape[0], cy + r + pad)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        local_cx = cx - x1
        local_cy = cy - y1
        band_w = max(4, r // 6)
        r_outer = min(int(r + band_w), min(roi.shape[0], roi.shape[1]) // 2 - 1)
        r_inner = max(int(r - band_w), 1)
        if r_inner >= r_outer:
            return None

        annulus = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(annulus, (local_cx, local_cy), r_outer, 255, -1)
        cv2.circle(annulus, (local_cx, local_cy), r_inner, 0, -1)

        mask = annulus > 0
        if not mask.any():
            return None
        hsv_values = roi_hsv[mask]
        if len(hsv_values) == 0:
            return None
        h_mean = float(np.mean(hsv_values[:, 0]))
        s_mean = float(np.mean(hsv_values[:, 1]))
        v_mean = float(np.mean(hsv_values[:, 2]))
        return (h_mean, s_mean, v_mean)
