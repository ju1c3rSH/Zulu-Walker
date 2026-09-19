import cv2
import numpy as np

from .base import DetectionMethod
from .....models.circle import CircleTargetItem, ShapeType


class ContourEllipseMethod(DetectionMethod):
    def detect(self, frame: np.ndarray, target_color: str) -> None:
        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        colors_to_detect = (
            [target_color] if target_color else list(self.detector.color_ranges.keys())
        )

        min_area_scaled = self.detector.min_area_threshold_ellipse * (scale * scale)

        for color_name in colors_to_detect:
            if color_name not in self.detector.color_ranges:
                continue

            ranges = self.detector.color_ranges[color_name]
            mask = None
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area_scaled:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

                if len(approx) < self.detector.min_contour_points:
                    continue

                try:
                    ellipse = cv2.fitEllipse(approx)
                except cv2.error:
                    continue

                center_orig = (int(ellipse[0][0] / scale), int(ellipse[0][1] / scale))
                axes_orig = (ellipse[1][0] / scale, ellipse[1][1] / scale)
                radius = max(axes_orig) / 2
                area = 3.14159 * (axes_orig[0] / 2) * (axes_orig[1] / 2)

                if scale < 1.0:
                    contour_orig = (contour * (1.0 / scale)).astype(np.int32)
                else:
                    contour_orig = contour

                target_item = CircleTargetItem(
                    index=0,
                    center_coordinates=center_orig,
                    radius=radius,
                    area=area,
                    shape_type=ShapeType.ELLIPSE,
                    contour_points=contour_orig,
                    bounding_box=(
                        int((center_orig[0] - axes_orig[0] / 2)),
                        int((center_orig[1] - axes_orig[1] / 2)),
                        int(axes_orig[0]),
                        int(axes_orig[1]),
                    ),
                    color=color_name,
                    major_axis=axes_orig[0],
                    minor_axis=axes_orig[1],
                )
                self.detector.circle_target.add_target(target_item)
