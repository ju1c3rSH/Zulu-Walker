from __future__ import annotations

import cv2
import numpy as np

from hal.interface import Detection

from .base import AbstractModelHandler
from .registry import ModelHandlerRegistry


_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_HAND_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}


@ModelHandlerRegistry.register("hand_landmarks")
class HandLandmarksHandler(AbstractModelHandler):
    def draw(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        for det in detections:
            color = _HAND_COLORS.get(det.class_id, (0, 255, 255))
            kps = det.keypoints

            for i, j in _HAND_CONNECTIONS:
                if i >= len(kps) or j >= len(kps):
                    continue
                p1 = (int(kps[i].x), int(kps[i].y))
                p2 = (int(kps[j].x), int(kps[j].y))
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

            for kp in kps[:21]:
                pt = (int(kp.x), int(kp.y))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                cv2.circle(frame, pt, 4, color, 1, cv2.LINE_AA)

            label = (
                f"{'Left' if det.class_id == 0 else 'Right'}:{det.score:.2f}"
            )
            cv2.putText(
                frame, label, (det.x, det.y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        return frame
