from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

import maix.camera
import maix.image
logger = logging.getLogger(__name__)

_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5
_CAP_PROP_GAIN = 14
_CAP_PROP_EXPOSURE = 15


class MaixCam2Camera:
    def __init__(
        self,
        source: Optional[Union[str, int]],
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
        camera_id: str = "maixcam2",
        buff_num: int = 3,
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        sensor_height_mm: Optional[float] = None,
        exposure_us: Optional[int] = None,
        gain: Optional[int] = None,
        aec: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._camera_id = camera_id
        self._focal_length_mm = focal_length_mm
        self._sensor_width_mm = sensor_width_mm
        self._sensor_height_mm = sensor_height_mm
        self._opened = False
        self._cam: Optional[maix.camera.Camera] = None
        self._last_raw: Optional[maix.image.Image] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_exposure: Optional[int] = None
        self._last_gain: Optional[int] = None
        self._aec_cfg: Optional[Dict[str, Any]] = aec if aec else None
        try:
            self._cam = maix.camera.Camera(
                width=width,
                height=height,
                format=maix.image.Format.FMT_RGB888,
                device=str(source) if source is not None else None,
                fps=fps,
                buff_num=buff_num,
                open=True,
            )
            self._opened = True
        except Exception as e:
            logger.warning("Camera init failed: %s", e)
            self._opened = False
            return
        # exposure/gain set outside the Camera() try block so a failure here
        # does not leave a partially-initialized camera handle dangling
        try:
            if exposure_us is not None:
                self._cam.exposure(int(exposure_us))
                self._last_exposure = int(exposure_us)
            if gain is not None:
                self._cam.gain(int(gain))
                self._last_gain = int(gain)
        except Exception as e:
            logger.warning("Camera post-init (exposure/gain) failed: %s", e)

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

    @property
    def last_raw(self):
        """The most recent raw maix Image frame (RGB888)."""
        return self._last_raw

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """The most recent BGR ndarray frame (cached by read_raw)."""
        return self._last_frame

    @property
    def last_gain(self) -> Optional[int]:
        return self._last_gain

    @property
    def last_exposure(self) -> Optional[int]:
        return self._last_exposure

    @property
    def aec_config(self) -> Optional[Dict[str, Any]]:
        return self._aec_cfg

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        try:
            img = self._cam.read(block=False)
            if img is None:
                return None
            self._last_raw = img
            result = maix.image.image2cv(img, ensure_bgr=False, copy=True)[:, :, ::-1]
            self._last_frame = result
            return result
        except Exception as e:
            logger.warning("Camera read failed: %s", e)
            return None

    def read_raw(self):
        if self._cam is None:
            return None
        try:
            raw = self._cam.read(block=False)
            if raw is not None:
                self._last_raw = raw
                np_img = maix.image.image2cv(raw, ensure_bgr=False, copy=True)
                self._last_frame = np_img[:, :, ::-1]
            return self._last_frame
        except Exception as e:
            logger.warning("Camera read_raw failed: %s", e)
            return None

    @property
    def raw_camera(self):
        return self._cam

    def set_gain(self, val: int) -> bool:
        if self._cam is None:
            return False
        try:
            self._cam.gain(int(val))
            self._last_gain = int(val)
            return True
        except Exception:
            return False

    def set_exposure(self, val: int) -> bool:
        if self._cam is None:
            return False
        try:
            self._cam.exposure(int(val))
            self._last_exposure = int(val)
            return True
        except Exception:
            return False

    def set(self, prop_id: int, value: int) -> bool:
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
                return False
            elif prop_id == _CAP_PROP_EXPOSURE:
                return self.set_exposure(value)
            elif prop_id == _CAP_PROP_GAIN:
                return self.set_gain(value)
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
