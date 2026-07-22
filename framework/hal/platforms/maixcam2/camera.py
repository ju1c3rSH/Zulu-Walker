from __future__ import annotations

from typing import Optional

import numpy as np

import maix.camera
import maix.image

_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5
_CAP_PROP_GAIN = 14
_CAP_PROP_EXPOSURE = 15


class MaixCam2Camera:
    def __init__(
        self,
        source,
        width: int = 640,
        height: int = 480,
        fps: float = -1,
        camera_id: str = "maixcam2",
        buff_num: int = 3,
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        sensor_height_mm: Optional[float] = None,
    ) -> None:
        self._camera_id = camera_id
        self._focal_length_mm = focal_length_mm
        self._sensor_width_mm = sensor_width_mm
        self._sensor_height_mm = sensor_height_mm
        self._opened = False
        self._cam: Optional[maix.camera.Camera] = None
        try:
            self._cam = maix.camera.Camera(
                width=width,
                height=height,
                format=maix.image.Format.FMT_RGB888,
                device=str(source) if source is not None else None,
                fps=fps if fps > 0 else -1,
                buff_num=buff_num,
                open=True,
            )
            self._opened = True
        except Exception:
            self._opened = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        if self._cam is None:
            return 0.0
        return self._cam.fps()

    @property
    def width(self) -> int:
        if self._cam is None:
            return 0
        return self._cam.width()

    @property
    def height(self) -> int:
        if self._cam is None:
            return 0
        return self._cam.height()

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
    def is_opened(self) -> bool:
        if self._cam is None:
            return False
        return self._cam.is_opened()

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        try:
            img = self._cam.read(block=False)
            if img is None:
                return None
            return maix.image.image2cv(img, ensure_bgr=True, copy=True)
        except Exception:
            return None

    def set(self, prop_id: int, value) -> bool:
        if self._cam is None:
            return False
        try:
            if prop_id == _CAP_PROP_FRAME_WIDTH:
                self._cam.set_resolution(int(value), self._cam.height())
                return True
            elif prop_id == _CAP_PROP_FRAME_HEIGHT:
                self._cam.set_resolution(self._cam.width(), int(value))
                return True
            elif prop_id == _CAP_PROP_FPS:
                self._cam.set_fps(float(value))
                return True
            elif prop_id == _CAP_PROP_EXPOSURE:
                self._cam.exposure(int(value))
                return True
            elif prop_id == _CAP_PROP_GAIN:
                self._cam.gain(int(value))
                return True
            else:
                return False
        except Exception:
            return False

    def release(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        self._opened = False
