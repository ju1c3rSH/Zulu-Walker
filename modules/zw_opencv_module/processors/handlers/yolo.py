from __future__ import annotations

import cv2
import numpy as np

from framework.hal.interface import Detection

from .base import AbstractModelHandler
from .registry import ModelHandlerRegistry


@ModelHandlerRegistry.register("yolo")
class YoloHandler(AbstractModelHandler):
    def draw(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        labels = self._ai.labels if hasattr(self._ai, "labels") else []

        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.w, det.y + det.h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = (
                labels[det.class_id]
                if det.class_id < len(labels)
                else str(det.class_id)
            )
            cv2.putText(
                frame, f"{label}:{det.score:.2f}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        return frame


@ModelHandlerRegistry.register("default")
class DefaultHandler(AbstractModelHandler):
    def draw(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.w, det.y + det.h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{det.class_id}:{det.score:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        return frame
