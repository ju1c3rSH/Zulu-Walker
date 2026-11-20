from __future__ import annotations

import logging
from typing import Optional

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
        source,
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
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
        self._last_raw: Optional[maix.image.Image] = None
        self._last_frame: Optional[np.ndarray] = None
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

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        try:
            img = self._cam.read(block=False)
            if img is None:
                return None
            self._last_raw = img
            return maix.image.image2cv(img, ensure_bgr=False, copy=True)[:, :, ::-1]
        except Exception as e:
            logger.warning("Camera read failed: %s", e)
            return None

    def _ingest_raw(self, raw) -> Optional[np.ndarray]:
        """新 raw 帧 -> 更新 last_raw/_last_frame 并返回 BGR numpy; None = 停帧。

        停帧必须返回 None（不得回退 _last_frame）: 接口约定见
        framework/hal/interface/camera.py::read_raw，mock/linux 两个平台都按此实现。
        这里曾经在空队列时返回上一帧 numpy，后果是 pipeline_camera 的"停帧不复用
        旧帧"早退在设备上失效 —— 视觉线程会把同一帧反复当新帧算（any_fresh 恒真、
        只 sleep(0) 忙转），推流侧据此认为"一直有新结果"，1s 陈旧归零永远不触发，
        相机真的挂死时会以冻结误差持续喂主站。2026-09-28 修正为跟约定一致。

        延迟转换（2026-10-09, `_lazy_convert` 为真时）: 同 CAM1 —— 不在这里
        做 image2cv, 交 LazyFrame 代理, 只有真用像素时才转（见 maixcam1 同款实现）。
        """
        if raw is None:
            return None
        self._last_raw = raw
        if self.__dict__.get("_lazy_convert", False):
            return self._ingest_lazy(raw)
        np_img = maix.image.image2cv(raw, ensure_bgr=False, copy=True)
        self._last_frame = np_img[:, :, ::-1]
        return self._last_frame

    def _ingest_lazy(self, raw):
        """延迟转换路径（跟 CAM1 一样，见 maixcam1/camera.py::_ingest_lazy）。"""
        from framework.hal.platforms.lazy_frame import LazyFrame

        def _convert(img):
            np_img = maix.image.image2cv(img, ensure_bgr=False, copy=True)
            if np_img is None:
                # 转换失败: 交回 None 由 LazyFrame 退化为空数组, 不在此抛异常
                # （见 lazy_frame.py::numpy 的回退处理说明, 与 CAM1 同款处理）。
                return None
            return np_img[:, :, ::-1]

        shape_hw = None
        try:
            shape_hw = (int(raw.height()), int(raw.width()))
        except Exception:
            pass
        frame = LazyFrame(raw, _convert, shape_hw=shape_hw)
        self._last_frame = frame
        return frame

    def set_lazy_convert(self, enabled: bool) -> None:
        """开关延迟转换（默认关 = 保持 `read_raw()` 急切转 numpy 的老约定）。"""
        self._lazy_convert = bool(enabled)

    def read_raw(self):
        if self._cam is None:
            return None
        try:
            return self._ingest_raw(self._cam.read(block=False))
        except Exception as e:
            logger.warning("Camera read_raw failed: %s", e)
            return None

    @property
    def raw_camera(self):
        return self._cam

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
                return False
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
