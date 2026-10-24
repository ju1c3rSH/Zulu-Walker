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
    # FMT_* 查表只建一次: `dir()` 每帧遍历会直接吃掉探针想量的那点开销。
    _FMT_NAMES = {getattr(maix.image.Format, a): a[4:]
                  for a in dir(maix.image.Format) if a.startswith("FMT_")}
except Exception:
    _FMT_NAMES = {}


def _fmt_tag(img) -> str:
    """maix Image 格式 -> "RGB888"/"BGR888"/… —— conv 探针的归因依据。"""
    if img is None:
        return "none"
    try:
        return _FMT_NAMES.get(img.format(), "?")
    except Exception:
        return "?"


def _log_attempts(n: int) -> None:
    """报告本次 camera 构造实际试了几级（2026-10-08 启动打点）。

    走 log_print 而非 logger: 台架看的是 stdout，而 basicConfig 把logger
    全部导向 logs/app.log，控制台看不到。
    正常路径恰好 1 次；>1 说明降级阶梯在白跑 —— 前面的失败若是"已经把
    sensor/ISP 初始化做了一半"的通用 Exception，重试等于把那段昂贵初始化
    重复做一遍（台架日志里 sensor_cfg.ini 出现两组，与此吻合）。
    """
    if n <= 1:
        return
    from utils.log_util import log_print
    log_print(f"[CAM1] camera.attempts={n}  <-- 降级重试，前面的构造被丢弃"
              f"（若 sensor_cfg.ini 打印了多组，说明初始化真做了多遍）")


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
        # 最近一帧的分项耗时(ms): read=相机取帧本身, convert=image2cv 转 BGR numpy。
        # 为什么必须拆开: 2026-10-08 台架真机 `[PERF] vision read=13.6ms`, 而隔离
        # 台架里 read≈0ms（帧总是就绪）+ image2cv 4.73ms —— 差 3 倍, 但**无法判断
        # 那 13.6ms 是"在等一帧"还是"在拷 921KB"**:
        #   · 等一帧   -> 相机没跟上 / 缓冲被打空, 要查 fps 与 buff_num;
        #   · 拷贝慢   -> image2cv 在抢核/抢内存带宽, 而它在 TARGET 下是**纯白做**
        #                 （TARGET 走 C++ find_blobs, 直接吃 last_raw, 这份 numpy
        #                  只服务 cv2 回退路径与调参页）。
        # 拆开后一眼定方向, 别再靠推测。
        self.stage_ms: dict = {"read": 0.0, "convert": 0.0}
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
        固件版本，降级到最小可用签名。

        ⚠ **降级会丢参数**：命中 level3/4 时 fps/buff_num 根本没传进去。按
        docs/doc/zh/vision/camera.md:163，分辨率<=1280x720 时 MaixPy 会把帧率
        设成 **80fps**，于是 project_config 的 `fps: 60` 被静默忽略、实际跑 80fps。
        每级成功都打日志记录命中级别与被丢弃的参数，否则台架无法判断实际状态。

        ⚠ **每次尝试都单独计时（2026-10-08）**: 相机是启动期最贵的一步，而
        降级阶梯意味着**可能白构造多次** —— 若level-1 抛的是通用 Exception
        而非 TypeError，会走 except Exception 只打warning 然后继续试 level-2,
        等于把sensor/ISP 初始化做了两遍（台架日志里 sensor_cfg.ini 被完整解析
        两组,与此吻合）。逐级打点 + 累计"尝试次数"日志, 让"开了几次"直接可见:
        正常应恰好 1次; 若看到 camera.attempts=N(N>1), 说明降级在白跑,
        此时应把命中的那一级固件签名固定下来、或给 level-1 补TypeError 之外
        的精确判定。
        """
        device = str(source) if source is not None else None
        attempts = 0
        from utils.log_util import log_print

        # 降级只做"**关键字逐步减少**"，不猜位置参数顺序。
        # 依据：docs/doc/zh/vision/camera.md 只验证到 3 位位置参数
        # (640,480) / (640,480,FORMAT)；device/fps/buff_num/open 一律以关键字
        # 出现。若这里写 6 位位置参数而真实签名是 (w,h,format,buff_num,fps,..)，
        # 会**静默把 buff_num 当 fps、fps 当 buff_num** —— 不报错、行为全错。
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

        # 1) 全参数（新版固件）—— 命中即所有参数生效
        attempts += 1
        try:
            with boot_stage("cam.open:L1(full-kwargs)"):
                cam = _level1()
            # warning 而非 info: root logger 被basicConfig 钉在 WARNING，
            # 台架要靠这行判断相机实际命中哪级构造（info 不会被打印）
            logger.warning("[CAM1] camera open level=1 (full kwargs, fps=%s buff_num=%s)",
                        fps, buff_num)
            log_print(f"[CAM1] camera open level=1 fps={fps} buff_num={buff_num}")
            _log_attempts(attempts)
            return cam
        except TypeError:
            pass
        except Exception as e:
            # ⚠ **非 TypeError 的失败不等于"本固件不支持这套签名"**。
            # 台架实测过一次 `RuntimeError: sensor timeout`（前一个进程/ISP 尚未
            # 释放, 或 sensor 上电未就绪）—— 那是**瞬时故障**, 不是能力缺失。
            # 此时直接往下降级会落到位置参数签名 L3, 后果是 fps / buff_num
            # **静默丢失**（MaixPy 对 <=1280x720 默认给 80fps, buff_num 用默认值）,
            # 取图延时与缓冲深度全部不可控 —— 而这正是我们在查的帧率问题。
            # 故先原样重试一次 L1, 仍失败才降级。
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

        # 2) 去掉 open=
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

        # 3) 官方示例已验证的位置参数签名 (w, h, format) —— **fps/buff_num 被丢弃**
        attempts += 1
        try:
            with boot_stage("cam.open:L3(positional)"):
                cam = maix.camera.Camera(width, height, maix.image.Format.FMT_RGB888)
            logger.warning("[CAM1] camera open DEGRADED level=3: fps=%s buff_num=%s "
                           "未生效（MaixPy 对 <=1280x720 默认 80fps），实际帧率以实测为准",
                           fps, buff_num)
            log_print(f"[CAM1] ⚠ camera 降级到 level=3(位置参数): fps={fps} "
                      f"buff_num={buff_num} **未生效**, 实际帧率以实测为准")
            _log_attempts(attempts)
            return cam
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Camera open(level3 positional) failed: %s", e)

        # 4) 最小签名 (w, h) —— 同上，fps/buff_num 全丢
        attempts += 1
        try:
            with boot_stage("cam.open:L4(minimal)"):
                cam = maix.camera.Camera(width, height)
            logger.warning("[CAM1] camera open DEGRADED level=4: fps=%s buff_num=%s "
                           "未生效（MaixPy 对 <=1280x720 默认 80fps），实际帧率以实测为准",
                           fps, buff_num)
            log_print(f"[CAM1] ⚠ camera 降级到 level=4(最小签名): fps={fps} "
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
            from utils.log_util import log_print
            log_print("[CAM1] ⚠ camera.read(block=False) 不被本固件接受 -> 已降级为"
                      "阻塞读: 停帧保护失效, 且每帧要等满一个取帧间隔（如需非阻塞,"
                      " 升级 MaixPy）")
            return None, False
        except Exception as e:
            logger.warning("Camera read failed: %s", e)
            return None, True

    def read(self) -> Optional[np.ndarray]:
        """调参页独占读帧用（可接受阻塞语义）。**恒返回真 numpy**（见尾部收敛）。"""
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
            # 懒转换模式下 _ingest_raw 交出 LazyFrame 代理 —— 但本方法的**返回
            # 契约就是 np.ndarray**（签名如此, 调参页全链路每帧必用像素）。
            # ⚠ 不能指望 cv2 自己转: OpenCV 的 Python 绑定（设备 4.9 / PC 5.0
            # 实测同报错）对 cv2.resize/cvtColor **只收真 ndarray**,
            # `__array__`/`__array_interface__` 协议一概不认 —— 不在这里收敛,
            # 调参页第一帧就死在 resize 上（2026-10-09 真机
            # "[PAGE] EXPLOSIVE step FAIL: ... src is not a numpy array"）。
            # numpy() 只在首次访问时转一次并缓存, 这里就是那次访问。
            frame = frame.numpy()
        return frame

    def _ingest_raw(self, raw) -> Optional[np.ndarray]:
        """新 raw 帧 -> 更新 last_raw/_last_frame 并返回帧; None = 停帧。

        **不得回退 _last_frame**（契约见 framework/hal/interface/camera.py::
        read_raw）。CAM2 曾在这里回退过旧帧，导致 pipeline_camera 的"停帧不复用
        旧帧"在设备上失效：视觉线程把同一帧反复当新帧算、结果时间戳持续刷新，
        1s 陈旧归零永不触发，相机真的挂死时会以冻结误差持续喂主站
        （2026-09-28 修正）。CAM1 实现同款语义。

        **延迟转换（2026-10-09, `_lazy_convert` 为真时）**: 不在这里做
        `image2cv`（921KB 拷贝, 实测 9~16ms）, 改为交出 `LazyFrame` 代理 ——
        只有真正用到像素（cv2 回退 / 调参页 / RESCUE 传 numpy）时才转一次。
        TARGET/EXPLOSIVE 走 C++ find_blobs 时**一次拷贝都不发生**。
        `_last_frame` 仍置为转换后的 numpy **或** 代理（依模式）; 显示侧只读
        `last_raw`（maix Image）, 与本改动无关。
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
        """延迟转换路径: 返回 LazyFrame, 不立刻 image2cv。

        转换**首次被访问时**才发生, 并计入 `convert` 分项 —— 这样 `[PERF] vision`
        的 conv 读数语义不变（只是从"每帧固定"变成"只在真用像素时才出现"）:
        TARGET 纯 C++ 路径下 conv 应降到 0, 这正是本优化的验收信号。
        """
        from framework.hal.platforms.lazy_frame import LazyFrame

        def _convert(img):
            _t0 = time.perf_counter()
            np_rgb = maix.image.image2cv(img, ensure_bgr=False, copy=True)
            self._mark_stage("convert", _t0)
            self._note_conv(_t0, img)
            if np_rgb is None:
                # 转换失败: 交回 None 由 LazyFrame 退化为空数组（不在闭包里抛,
                # 否则异常会打断整帧 —— 见 lazy_frame.py::numpy 的降级说明）。
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
        """开关延迟转换（默认关 = 保持 `read_raw()` 急切转 numpy 的旧契约）。

        由 PipelineCamera 按"任务是否需要 numpy"决定; 台架/测试若依赖
        `read_raw()` 直接返回 numpy, 保持默认即可。
        """
        self._lazy_convert = bool(enabled)

    def _mark_stage(self, key: str, t0: float) -> None:
        """记一笔分项耗时（毫秒）。

        容器缺失时**懒建到实例字典**（`self.__dict__`）而不是挂在类属性上:
        类级可变默认值会被所有实例共享, 一个实例写进去会污染另一个 —— 那种
        串味 bug 在台架上极难查。这里用 `__dict__.get` 既避开共享, 又让
        "绕过 __init__ 构造的实例"（测试桩）不会因为多一个计时字段就崩。
        """
        st = self.__dict__.get("stage_ms")
        if st is None:
            st = self.__dict__["stage_ms"] = {"read": 0.0, "convert": 0.0}
        st[key] = (time.perf_counter() - t0) * 1000.0

    def _note_conv(self, t0: float, raw=None) -> None:
        """`[PERF] conv` 探针: image2cv 每帧代价 + 源格式。

        ⚠ 格式缓存 1s 过期: `format()` 要问 C++, 但缓存到进程结束会让 src= 永久失真。
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
        _t0 = time.perf_counter()
        raw, _ok = self._read_nonblocking()
        self._mark_stage("read", _t0)
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
        # 一并清掉帧缓存: _last_raw 持有已释放相机的 Image 引用，
        # vision_manager._update_display_frame 只判raw_img is None，
        # 残留会让显示侧继续拿释放后的帧画 HUD。
        self._last_raw = None
        self._last_frame = None
        self._opened = False
