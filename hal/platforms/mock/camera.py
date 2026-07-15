from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MockCamera:
    def __init__(self, camera_id: str = "mock") -> None:
        self._camera_id = camera_id
        self._frame_index = 0
        self._pattern_size = (8, 6)

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return 30.0

    def read(self) -> Optional[np.ndarray]:
        self._frame_index += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (32, 32, 32)
        cv2.putText(
            frame,
            f"MOCK {self._camera_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame

    def set(self, prop_id: int, value) -> bool:
        return True

    def release(self) -> None:
        logger.info("MockCamera %s released", self._camera_id)
