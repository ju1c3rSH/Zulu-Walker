from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LinuxCamera:
    def __init__(
        self,
        camera_id: str,
        source,
        width: int = 640,
        height: int = 480,
        fps: float = 120,
        queue_size: int = 2,
    ) -> None:
        self._camera_id = camera_id
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_size = queue_size
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._capture_thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps = fps

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return self._actual_fps

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            logger.warning("V4L2 open failed for %s, trying default backend", self._source)
            self._cap = cv2.VideoCapture(self._source)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        self._bind_little_cores()
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                logger.error("Camera %s: read failed", self._camera_id)

    def _bind_little_cores(self) -> None:
        try:
            import os
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except Exception:
            logger.error("Failed to bind camera thread to little cores")

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def set(self, prop_id: int, value) -> bool:
        if self._cap is None:
            return False
        return self._cap.set(prop_id, value)

    def release(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1)
        if self._cap:
            self._cap.release()
            self._cap = None
