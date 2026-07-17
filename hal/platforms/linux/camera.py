from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
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
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        sensor_height_mm: Optional[float] = None,
    ) -> None:
        self._camera_id = camera_id
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_size = queue_size
        self._focal_length_mm = focal_length_mm
        self._sensor_width_mm = sensor_width_mm
        self._sensor_height_mm = sensor_height_mm
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._capture_thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps = fps
        self._actual_width = width
        self._actual_height = height

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return self._actual_fps

    @property
    def focal_length_mm(self) -> Optional[float]:
        return self._focal_length_mm

    @property
    def sensor_width_mm(self) -> Optional[float]:
        return self._sensor_width_mm

    @property
    def sensor_height_mm(self) -> Optional[float]:
        return self._sensor_height_mm

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @property
    def actual_height(self) -> int:
        return self._actual_height

    def _open_once(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.warning("V4L2 open failed for %s, trying default backend", self._source)
            cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {self._camera_id}: device not accessible")
        return cap

    def start(self) -> None:
        timeout = 5.0
        for attempt in (1, 2):
            result: list[cv2.VideoCapture] = []
            exc: list[BaseException] = []

            def worker():
                try:
                    result.append(self._open_once())
                except KeyboardInterrupt:
                    raise
                except SystemExit:
                    raise
                except BaseException as e:
                    exc.append(e)

            t = threading.Thread(
                target=worker,
                daemon=True,
                name=f"cam-open-{self._camera_id}",
            )
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                logger.warning(
                    "Camera %s: open timed out after %.1fs (attempt %d/2)",
                    self._camera_id, timeout, attempt,
                )
                if attempt == 2:
                    raise concurrent.futures.TimeoutError(
                        f"Camera {self._camera_id}: V4L2 open timed out"
                    )
                time.sleep(1)
            elif exc:
                if isinstance(exc[0], (KeyboardInterrupt, SystemExit)):
                    raise exc[0]
                if attempt == 2:
                    raise RuntimeError(
                        f"Camera {self._camera_id}: open failed"
                    ) from exc[0]
                time.sleep(1)
            else:
                self._cap = result[0]
                break
        else:
            raise RuntimeError(f"Camera {self._camera_id}: all open attempts failed")

        if not self._cap.isOpened():
            raise RuntimeError(f"Camera {self._camera_id}: opened but not accessible")

        if not self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')):
            logger.debug("Camera %s: failed to set FOURCC=MJPG", self._camera_id)

        for prop, name in [
            (cv2.CAP_PROP_FRAME_WIDTH, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
            (cv2.CAP_PROP_FPS, "fps"),
        ]:
            value = {"width": self._width, "height": self._height, "fps": self._fps}[name]
            if not self._cap.set(prop, value):
                logger.debug("Camera %s: failed to set %s=%s", self._camera_id, name, value)

        actual = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = actual if actual > 0 else self._fps
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (self._actual_width, self._actual_height) != (self._width, self._height):
            logger.warning(
                "Camera %s: requested resolution %dx%d, got %dx%d",
                self._camera_id, self._width, self._height,
                self._actual_width, self._actual_height,
            )
        if abs(self._actual_fps - self._fps) > 1:
            logger.warning(
                "Camera %s: requested fps %.1f, got %.1f",
                self._camera_id, self._fps, self._actual_fps,
            )

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-capture-{self._camera_id}",
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        self._bind_capture_cores()
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

    def _bind_capture_cores(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("camera_capture")

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
