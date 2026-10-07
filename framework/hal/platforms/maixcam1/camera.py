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


class MaixCam1Camera:
    """MaixCAM / MaixCAM-Pro 相机（CVI cv181x平台）。

    与 MaixCam2Camera 的差异（2026-10-07回移植）:

    1. **取图延时**: MaixPy 文档明确"对于 maixcam, 由于内部软件框架的限制,
       即使设置 buff_num=1, 实际至少还是会存在一个双缓存, 测试取图延时最低在
       30+ms 左右"。因此 CAM1 上 `fps=60` 拿不到 60fps（帧率上限约 30），且
       `buff_num` 再小也压不到 CAM2 的水平。**推流率不得按相机标称帧率验收**。

    2. **镜头**: 默认 M12 手动调焦镜头（CAM2 是固定焦距），靶标/爆炸物
       标定几何（坑位 ROI、误差中心）**不能直接沿用 CAM2 的值**，需现场重标。

    3. **constructor 签名**: CAM1 走位置参数 `Camera(w, h, format)`，
       `device=`/`buff_num=` 等关键字在老固件上可能不认，故全部走 try/except
       降级构造（见 __init__），保证新固件能用、老固件不崩。
    """

    def __init__(
        self,
        source,
        width: int = 640,
        height: int = 480,
        fps: Optional[float] = None,
        camera_id: str = "maixcam1",
        buff_num: int = 2,
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
        # 非阻塞读帧是否可用：CAM1 老固件的 read() 可能不认 block kwarg。
        # 探测一次后缓存，read_raw 据此决定走非阻塞（保停帧检测）还是阻塞读。
        self._block_kwarg_ok: Optional[bool] = None
        self._cam = self._open(source, width, height, fps, buff_num)
        self._opened = self._cam is not None

    # ------------------------------------------------------------------ #
    #  构造（CAM1 固件版本兼容）
    # ------------------------------------------------------------------ #

    def _open(self, source, width, height, fps, buff_num):
        """按"能力从多到少"逐级尝试构造相机，任一级成功即返回。

        CAM1 官方示例是 `camera.Camera(640, 480)`（位置参数、无 buff_num），
        老固件对 `device=` / `buff_num=` / `open=` 的支持程度不一，这里不假设
        固件版本，直接降级到最小可用签名。
        """
        device = str(source) if source is not None else None

        # 1) 全参数（新版固件）
        try:
            return maix.camera.Camera(
                width=width,
                height=height,
                format=maix.image.Format.FMT_RGB888,
                device=device,
                fps=fps,
                buff_num=buff_num,
                open=True,
            )
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Camera open(full kwargs) failed: %s", e)

        # 2) 位置参数 w/h/format + buff_num
        try:
            return maix.camera.Camera(
                width, height, maix.image.Format.FMT_RGB888, device, fps, buff_num
            )
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Camera open(positional+buff) failed: %s", e)

        # 3) 位置参数 w/h/format（CAM1 官方示例签名）
        try:
            return maix.camera.Camera(width, height, maix.image.Format.FMT_RGB888)
        except Exception as e:
            logger.warning("Camera open(positional) failed: %s", e)

        # 4) 最小签名
        try:
            return maix.camera.Camera(width, height)
        except Exception as e:
            logger.error("Camera open(minimal) failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        if self._cam is None:
            return 0.0
        try:
            return self._cam.fps()
        except Exception:
            return 0.0

    @property
    def width(self) -> int:
        if self._cam is None:
            return 0
        try:
            return self._cam.width()
        except Exception:
            return 0

    @property
    def height(self) -> int:
        if self._cam is None:
            return 0
        try:
            return self._cam.height()
        except Exception:
            return 0

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
        try:
            return self._cam.is_opened()
        except Exception:
            return False

    @property
    def last_raw(self):
        """The most recent raw maix Image frame (RGB888)."""
        return self._last_raw

    # ------------------------------------------------------------------ #
    #  取帧
    # ------------------------------------------------------------------ #

    def _read_nonblocking(self):
        """非阻塞取一帧；固件不支持 block kwarg 时返回 (None, False)。"""
        try:
            return self._cam.read(block=False), True
        except TypeError:
            self._block_kwarg_ok = False
            logger.warning("camera.read(block=...) unsupported on this firmware; "
                           "falling back to blocking read (停帧检测降级, 见 read_raw)")
            return None, False
        except Exception as e:
            logger.warning("Camera read failed: %s", e)
            return None, True

    def read(self) -> Optional[np.ndarray]:
        """调参页独占读帧用（可接受阻塞语义）。"""
        if self._cam is None:
            return None
        img, _ok = self._read_nonblocking()
        if img is None:
            try:
                img = self._cam.read()
            except Exception as e:
                logger.warning("Camera read(blocking) failed: %s", e)
                return None
        if img is None:
            return None
        return self._ingest_raw(img)

    def _ingest_raw(self, raw) -> Optional[np.ndarray]:
        """新 raw 帧 -> 更新 last_raw/_last_frame 并返回 BGR numpy; None = 停帧。

        **不得回退 _last_frame**（契约见 framework/hal/interface/camera.py::
        read_raw）。CAM2 曾在这里回退过旧帧，导致 pipeline_camera 的"停帧不复用
        旧帧"在设备上失效：视觉线程把同一帧反复当新帧算、结果时间戳持续刷新，
        1s 陈旧归零永不触发，相机真的挂死时会以冻结误差持续喂主站
        （2026-09-28 修正）。CAM1 实现同款语义。
        """
        if raw is None:
            return None
        self._last_raw = raw
        np_rgb = maix.image.image2cv(raw, ensure_bgr=False, copy=True)
        self._last_frame = np_rgb[:, :, ::-1]
        return self._last_frame

    def read_raw(self) -> Optional[np.ndarray]:
        """主流水线取帧：非阻塞，空队列 = None（停帧）。

        CAM1 特有风险：若固件不支持 `read(block=False)`（见 _read_nonblocking），
        退化为阻塞读 —— 此时相机真挂死会让调用线程卡在 read() 里，**停帧
        归零保护不会触发**。这是 CAM1 上的已知降级，日志已明示；台架验收时
        需确认固件走的是非阻塞分支（启动日志无 "block=... unsupported"）。

        降级探测发生在**首次**调用：那一拍必须补一次阻塞读把帧拿回来，否则
        首帧直接丢成 None，等于凭空多一帧"停帧"（视觉侧 any_fresh 首拍为假，
        虽不至于错乱，但会推迟第一帧结果，也让降级看起来像"相机没画面"）。
        """
        if self._cam is None:
            return None
        if self._block_kwarg_ok is False:
            return self._read_blocking()
        raw, _ok = self._read_nonblocking()
        if raw is None and self._block_kwarg_ok is False:
            # 本拍刚探测出固件不支持 block kwarg -> 立刻走阻塞读补回这一帧
            return self._read_blocking()
        return self._ingest_raw(raw)

    def _read_blocking(self) -> Optional[np.ndarray]:
        try:
            raw = self._cam.read()
        except Exception as e:
            logger.warning("Camera read_raw(blocking) failed: %s", e)
            return None
        return self._ingest_raw(raw)

    @property
    def raw_camera(self):
        return self._cam

    # ------------------------------------------------------------------ #
    #  曝光/增益
    # ------------------------------------------------------------------ #

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
                # CAM1 fps 由构造参数决定，运行期改分辨率不重设帧率
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
