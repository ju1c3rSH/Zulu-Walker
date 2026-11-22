from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

import maix.camera
import maix.image

from utils.boot_timing import boot_stage
from utils.log_util import perf_detail_enabled, perf_note

logger = logging.getLogger(__name__)

_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5
_CAP_PROP_GAIN = 14
_CAP_PROP_EXPOSURE = 15


try:
    # Build the FMT_* lookup once: walking dir() every frame would consume the
    # very overhead the probe is trying to measure.
    _FMT_NAMES = {getattr(maix.image.Format, a): a[4:]
                  for a in dir(maix.image.Format) if a.startswith("FMT_")}
except Exception:
    _FMT_NAMES = {}

def _fmt_tag(img) -> str:
    """Map a maix Image format to a name like "RGB888" for conv probing."""
    if img is None:
        return "none"
    try:
        return _FMT_NAMES.get(img.format(), "?")
    except Exception:
        return "?"


def _log_attempts(n: int) -> None:
    """Report how many construction levels were actually attempted.

    Use log_print rather than logger: the bench reads stdout, while
    basicConfig sends all logger output to logs/app.log, invisible on the
    console. A normal path attempts exactly once; more than one means a
    fallback was wasted. If an earlier failure was a generic Exception raised
    after sensor/ISP initialization had partly run, retrying repeats that
    expensive initialization.
    """
    if n <= 1:
        return
    from utils.log_util import log_print
    log_print(f"[CAM1] camera.attempts={n}  <-- 降级重试，前面的构造被丢弃"
              f"（若 sensor_cfg.ini 打印了多组，说明初始化真做了多遍）")


class MaixCam1Camera:
    """Camera for MaixCAM / MaixCAM-Pro (CVI cv181x platform).

    Differences from MaixCam2Camera:

    1. Capture latency: MaixPy documents that CAM1 double-buffers internally,
       giving a minimum frame age of 30+ ms. This bounds latency, not frame
       rate (hardware tests reach ~60 fps on the [FPS] pipeline). Lowering
       buff_num does not reduce frame age; streaming rate is verified by the
       [FPS] pipeline measurement.
    2. Lens: the default is an M12 manual-focus lens (CAM2 uses a fixed focal
       length), so calibration geometry (ROI and error center) cannot be
       copied from CAM2 and must be recalibrated on site.
    3. Constructor: CAM1 uses positional ``Camera(w, h, format)``. Older
       firmware may reject ``device=``/``buff_num=`` keywords, so __init__
       tries several signatures under try/except and falls back to simpler
       ones, working on both new and old firmware.
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
        # Per-frame timings in ms: read = camera capture, convert = image2cv
        # conversion to BGR numpy. Keeping them separate distinguishes causes
        # that total time hides: on the bench read is ~0 ms (frames always
        # ready) while on hardware it is ~13.6 ms, a threefold gap that cannot
        # otherwise tell "waiting for a frame" from "copying 921 KB".
        #   - waiting -> camera cannot keep up / buffer drained: check fps and
        #     buff_num;
        #   - slow copy -> image2cv competing for cores or memory bandwidth,
        #     which is pure overhead on TARGET (C++ find_blobs consumes
        #     last_raw directly; this numpy array only serves the cv2 fallback
        #     and the tuning page).
        self.stage_ms: dict = {"read": 0.0, "convert": 0.0}
        # Whether nonblocking reads are available: older CAM1 firmware may
        # reject the block kwarg. Probe once and cache the result; read_raw
        # then chooses nonblocking (preserving stalled-frame detection) or
        # blocking reads.
        self._block_kwarg_ok: Optional[bool] = None
        self._cam = self._open(source, width, height, fps, buff_num)
        self._opened = self._cam is not None

    # ------------------------------------------------------------------ #
    #  Construction (CAM1 firmware compatibility)
    # ------------------------------------------------------------------ #

    def _open(self, source, width, height, fps, buff_num):
        """Try camera constructors from richest to simplest; return on first success.

        CAM1's official example is ``camera.Camera(640, 480)`` (positional,
        no buff_num), and support for ``device=``/``buff_num=``/``open=``
        varies by firmware. Rather than assume a version, fall back to the
        smallest usable signature.

        Falling back drops parameters: at levels 3/4, fps and buff_num are
        never passed. Per docs/doc/zh/vision/camera.md:163, MaixPy sets 80 fps
        for resolutions <=1280x720, so project_config's ``fps: 60`` is silently
        ignored and the camera runs at 80 fps. Log the level reached and any
        dropped parameters at each success, otherwise the bench cannot tell
        the real state.

        Time each attempt separately: camera open is the most expensive boot
        step, and fallback can construct repeatedly. If level 1 raises a
        generic Exception instead of TypeError, the handler only warns and
        tries level 2, repeating sensor/ISP initialization. Per-level timing
        plus a total attempt count make the number of opens visible; it should
        normally be exactly one.
        """
        device = str(source) if source is not None else None
        attempts = 0
        from utils.log_util import log_print

        # On fallback, only drop keywords; never guess positional argument
        # order. docs/doc/zh/vision/camera.md verifies at most three
        # positional args (640,480) / (640,480,FORMAT); device/fps/buff_num/
        # open always appear as keywords. Passing six positional args when the
        # real signature is (w,h,format,buff_num,fps,..) would silently swap
        # buff_num and fps: no error, entirely wrong behavior.

        def _level1():
            return maix.camera.Camera(
                width=width,
                height=height,
                format=maix.image.Format.FMT_RGB888,
                device=device,
                fps=fps,
                buff_num=buff_num,
                open=True,
            )

        def _level2():
            return maix.camera.Camera(
                width=width,
                height=height,
                format=maix.image.Format.FMT_RGB888,
                device=device,
                fps=fps,
                buff_num=buff_num,
            )

        # 1) All kwargs (new firmware): on success every parameter takes effect.
        attempts += 1
        try:
            with boot_stage("cam.open:L1(full-kwargs)"):
                cam = _level1()
            # Use warning rather than info: basicConfig pins the root logger
            # at WARNING, and the bench relies on this line to see which level
            # was reached (info would not be printed).
            logger.warning("[CAM1] camera open level=1 (full kwargs, fps=%s buff_num=%s)",
                        fps, buff_num)
            log_print(f"[CAM1] camera open level=1 fps={fps} buff_num={buff_num}")
            _log_attempts(attempts)
            return cam
        except TypeError:
            pass
        except Exception as e:
            # A non-TypeError failure does not mean the firmware rejects this
            # signature. The bench once saw ``RuntimeError: sensor timeout``,
            # a transient power-up issue rather than a missing capability.
            # Falling back directly lands on the positional L3 signature,
            # silently dropping fps/buff_num and leaving latency and buffer
            # depth uncontrolled. Retry L1 unchanged first; fall back only if
            # that also fails.
            logger.warning("Camera open(level1 full kwargs) failed: %s; 原样重试一次", e)
            log_print(f"[CAM1] camera L1 构造失败({type(e).__name__}: {e}), "
                      f"原样重试一次（避免降级丢掉 fps/buff_num）")
            try:
                time.sleep(0.3)
                with boot_stage("cam.open:L1-retry(full-kwargs)"):
                    cam = _level1()
                logger.warning("[CAM1] camera open level=1 (retry ok, fps=%s buff_num=%s)",
                               fps, buff_num)
                log_print(f"[CAM1] camera open level=1（重试成功）"
                          f"fps={fps} buff_num={buff_num}")
                _log_attempts(attempts)
                return cam
            except TypeError:
                pass
            except Exception as e2:
                logger.warning("Camera open(level1 retry) failed: %s", e2)
                log_print(f"[CAM1] camera L1 重试仍失败({type(e2).__name__}: {e2}) -> 降级")

        # 2) Drop open=.
        attempts += 1
        try:
            with boot_stage("cam.open:L2(no-open)"):
                cam = _level2()
            logger.warning("[CAM1] camera open level=2 (no open=, fps=%s buff_num=%s)",
                        fps, buff_num)
            log_print(f"[CAM1] camera open level=2 fps={fps} buff_num={buff_num}")
            _log_attempts(attempts)
            return cam
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Camera open(level2 no-open) failed: %s", e)

        # 3) Official positional signature (w, h, format): fps/buff_num dropped.
        attempts += 1
        try:
            with boot_stage("cam.open:L3(positional)"):
                cam = maix.camera.Camera(width, height, maix.image.Format.FMT_RGB888)
            logger.warning("[CAM1] camera open DEGRADED level=3: fps=%s buff_num=%s "
                           "未生效（MaixPy 对 <=1280x720 默认 80fps），实际帧率以实测为准",
                           fps, buff_num)
            log_print(f"[CAM1] camera 降级到 level=3(位置参数): fps={fps} "
                      f"buff_num={buff_num} **未生效**, 实际帧率以实测为准")
            _log_attempts(attempts)
            return cam
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Camera open(level3 positional) failed: %s", e)

        # 4) Minimal signature (w, h): same as above, fps/buff_num all dropped.
        attempts += 1
        try:
            with boot_stage("cam.open:L4(minimal)"):
                cam = maix.camera.Camera(width, height)
            logger.warning("[CAM1] camera open DEGRADED level=4: fps=%s buff_num=%s "
                           "未生效（MaixPy 对 <=1280x720 默认 80fps），实际帧率以实测为准",
                           fps, buff_num)
            log_print(f"[CAM1] camera 降级到 level=4(最小签名): fps={fps} "
                      f"buff_num={buff_num} **未生效**, 实际帧率以实测为准")
            _log_attempts(attempts)
            return cam
        except Exception as e:
            logger.error("Camera open(level4 minimal) failed: %s", e)
            log_print(f"[CAM1] camera 四级构造全部失败: {type(e).__name__}: {e}")
            _log_attempts(attempts)
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
    #  Frame capture
    # ------------------------------------------------------------------ #

    def _read_nonblocking(self):
        """Read a frame nonblocking; return (None, False) when block is unsupported."""
        try:
            return self._cam.read(block=False), True
        except TypeError:
            self._block_kwarg_ok = False
            logger.warning("camera.read(block=...) unsupported on this firmware; "
                           "falling back to blocking read (停帧检测降级, 见 read_raw)")
            from utils.log_util import log_print
            log_print("[CAM1] camera.read(block=False) 不被本固件接受 -> 已降级为"
                      "阻塞读: 停帧保护失效, 且每帧要等满一个取帧间隔（如需非阻塞,"
                      " 升级 MaixPy）")
            return None, False
        except Exception as e:
            logger.warning("Camera read failed: %s", e)
            return None, True

    def read(self) -> Optional[np.ndarray]:
        """Read a frame for the tuning page (blocking is allowed here).

        Always return a real numpy array; see the conversion at the end.
        """
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
        frame = self._ingest_raw(img)
        if frame is not None and not isinstance(frame, np.ndarray):
            # In lazy mode _ingest_raw yields a LazyFrame proxy, but this
            # method promises an ndarray (as signed, and the tuning page uses
            # pixels every frame). OpenCV's Python bindings for resize/
            # cvtColor accept only real ndarrays and ignore the
            # __array__/__array_interface__ protocols, so without conversion
            # here the tuning page dies on resize at the first frame. numpy()
            # converts once and caches on first access, which is this call.
            frame = frame.numpy()
        return frame

    def _ingest_raw(self, raw) -> Optional[np.ndarray]:
        """Ingest a new raw frame, update caches, and return it; None = stalled.

        Never fall back to _last_frame (contract in framework/hal/interface/
        camera.py::read_raw). CAM2 once did, breaking the "no reuse on stall"
        rule: the vision thread kept treating the same frame as new, refreshed
        timestamps continuously so the 1 s staleness counter never fired, and
        fed frozen errors to the main station when the camera really hung.
        CAM1 follows the same rule.

        With deferred conversion (``_lazy_convert`` true), skip image2cv here
        (a 921 KB copy, measured at 9-16 ms) and return a ``LazyFrame`` proxy
        instead. Conversion happens once only when pixels are actually used
        (cv2 fallback, tuning page, numpy handlers); C++ find_blobs handlers
        cause no copy at all. ``_last_frame`` still holds the converted numpy
        array or the proxy.
        """
        if raw is None:
            return None
        self._last_raw = raw
        if self.__dict__.get("_lazy_convert", False):
            return self._ingest_lazy(raw)
        _t0 = time.perf_counter()
        np_rgb = maix.image.image2cv(raw, ensure_bgr=False, copy=True)
        self._last_frame = np_rgb[:, :, ::-1]
        self._mark_stage("convert", _t0)
        self._note_conv(_t0)
        return self._last_frame

    def _ingest_lazy(self, raw):
        """Return a LazyFrame without converting immediately.

        Conversion happens on first access and is still charged to the
        ``convert`` stage, so the meaning of the ``[PERF]`` conv reading is
        unchanged (only its timing shifts from every frame to actual pixel
        use). On the pure C++ TARGET path, conv should drop to zero, which is
        the acceptance signal for this optimization.
        """
        from framework.hal.platforms.lazy_frame import LazyFrame

        def _convert(img):
            _t0 = time.perf_counter()
            np_rgb = maix.image.image2cv(img, ensure_bgr=False, copy=True)
            self._mark_stage("convert", _t0)
            self._note_conv(_t0, img)
            if np_rgb is None:
                # On conversion failure, return None so LazyFrame falls back
                # to an empty array. Do not raise inside the closure, or the
                # exception would break the whole frame (see the fallback
                # handling in lazy_frame.py::numpy).
                return None
            return np_rgb[:, :, ::-1]

        shape_hw = None
        try:
            shape_hw = (int(raw.height()), int(raw.width()))
        except Exception:
            pass
        frame = LazyFrame(raw, _convert, shape_hw=shape_hw)
        self._last_frame = frame
        return frame

    def set_lazy_convert(self, enabled: bool) -> None:
        """Toggle deferred conversion (off by default, keeping read_raw's
        eager numpy conversion).

        PipelineCamera decides based on whether the task needs numpy.
        """
        self._lazy_convert = bool(enabled)

    def _mark_stage(self, key: str, t0: float) -> None:
        """Record one stage duration in milliseconds.

        If the container is missing, create it lazily in the instance dict
        (``self.__dict__``) rather than as a class attribute: a mutable class
        default would be shared by all instances, and one instance writing to
        it would corrupt another, a bug that is hard to trace on the bench.
        Using ``__dict__.get`` avoids sharing and also lets instances built
        without __init__ (test stubs) work without gaining an extra field.
        """
        st = self.__dict__.get("stage_ms")
        if st is None:
            st = self.__dict__["stage_ms"] = {"read": 0.0, "convert": 0.0}
        st[key] = (time.perf_counter() - t0) * 1000.0

    def _note_conv(self, t0: float, raw=None) -> None:
        """``[PERF] conv`` probe: per-frame image2cv cost plus source format.

        Cache the format for 1 s: ``format()`` queries C++, but caching it for
        the whole process would leave src= permanently stale.
        """
        if not perf_detail_enabled():
            return
        now = time.monotonic()
        tag = self.__dict__.get("_conv_fmt")
        if tag is None or now - self.__dict__.get("_conv_fmt_ts", 0.0) >= 1.0:
            tag = _fmt_tag(raw if raw is not None
                           else self.__dict__.get("_last_raw"))
            self.__dict__["_conv_fmt"] = tag
            self.__dict__["_conv_fmt_ts"] = now
        perf_note("conv", (time.perf_counter() - t0) * 1000.0, f"src={tag} out=BGR")

    def read_raw(self) -> Optional[np.ndarray]:
        """Read a frame for the main pipeline; nonblocking, empty = None (stall).

        CAM1 risk: if the firmware rejects ``read(block=False)`` (see
        _read_nonblocking), reads fall back to blocking, so a truly hung
        camera wedges the calling thread and the stalled-frame counter never
        fires. This fallback is known and logged; bench acceptance must
        confirm the nonblocking branch is used (no "block=... unsupported" in
        the startup log).

        The fallback is detected on first call; that tick must perform one
        blocking read to recover the frame, otherwise the first frame is lost
        as None and invents a spurious stall.
        """
        if self._cam is None:
            return None
        if self._block_kwarg_ok is False:
            return self._read_blocking()
        _t0 = time.perf_counter()
        raw, _ok = self._read_nonblocking()
        self._mark_stage("read", _t0)
        if raw is None and self._block_kwarg_ok is False:
            # This tick just detected that block is unsupported: read
            # blocking now to recover the frame.
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
    #  Exposure / gain
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
                # CAM1 fps is set by the constructor; changing resolution at
                # runtime does not reset the frame rate.
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
        # Clear frame caches too: _last_raw holds an Image reference from the
        # closed camera, and leaving it lets the display keep drawing the HUD
        # on a released frame.
        self._last_raw = None
        self._last_frame = None
        self._opened = False
