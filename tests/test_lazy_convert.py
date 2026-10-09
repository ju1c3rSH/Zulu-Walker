# -*- coding: utf-8 -*-
"""按需 image2cv（LazyFrame）回归测试 —— "干掉 TARGET 下白做的 image2cv"。

背景（2026-10-09, 用户要求）:
  相机过去每帧都做 `maix.image.image2cv(raw, copy=True)` —— 640x480x3 一次拷
  921KB, 真机 [PERF] vision conv= 实测 9~16ms。但色块任务(EXPLOSIVE/TARGET)走
  C++ find_blobs 直接吃 `camera.last_raw`, 人质任务走 `_raw=True` 也直接吃 maix
  Image —— **真机上这份 BGR numpy 根本没人用**, 属于白做。

本文件守住四件事:
  1. 默认行为**不变**: `read_raw()` 仍急切返回 BGR numpy（不许悄悄改契约, 否则
     调参页/cv2 回退/所有既有测试全崩）。
  2. 开 `set_lazy_convert(True)` 后, `read_raw()` 返回代理, **不触发 image2cv**。
  3. 只读 `.shape` 也**不触发**转换（这是 TARGET 路径最常见的白做来源）。
  4. 一旦真用像素（切片/`__array__`）才转换一次, 且**同一帧只转一次**（缓存）。
"""
import os

import numpy as np
import pytest


_MAIXCAM1_CAMERA_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "framework", "hal", "platforms", "maixcam1", "camera.py")


class _FakeRaw:
    """maix Image 桩: 有 width()/height() 元数据 (零拷贝拿 shape 用)。"""

    def __init__(self, w=640, h=480):
        self._w, self._h = w, h

    def width(self):
        return self._w

    def height(self):
        return self._h


class _FakeCam1:
    def __init__(self, frame):
        self._frame = frame

    def read(self, block=True):        # noqa: A002 - 对齐固件签名
        return self._frame


def _load_cam1(monkeypatch, image2cv):
    """按文件路径加载 maixcam1 相机模块并注入假 maix（PC 端可测设备实现）。"""
    import importlib.util
    import sys
    import types

    image_mod = types.ModuleType("maix.image")
    image_mod.image2cv = image2cv
    image_mod.Format = types.SimpleNamespace(FMT_RGB888=0)
    camera_mod = types.ModuleType("maix.camera")
    camera_mod.Camera = object
    maix_mod = types.ModuleType("maix")
    maix_mod.image = image_mod
    maix_mod.camera = camera_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.camera", camera_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)

    spec = importlib.util.spec_from_file_location("_maixcam1_camera_lazy",
                                                  _MAIXCAM1_CAMERA_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_cam(mod, raw, monkeypatch, image2cv):
    cam = object.__new__(mod.MaixCam1Camera)
    cam._last_frame = None
    cam._last_raw = None
    cam._cam = _FakeCam1(raw)
    cam._block_kwarg_ok = True
    # _mark_stage 依赖这些字段; 桩成 no-op 以免测转换调用次数时被打扰
    cam._stage_ms = {}
    cam._stage_ts = {}
    monkeypatch.setattr(cam, "_mark_stage", lambda *a, **k: None, raising=False)
    return cam


# ---------- 1. 默认行为不变（急切转换） ----------

def test_default_read_raw_is_eager_numpy(monkeypatch):
    """不改任何开关时, read_raw() 必须是**急切转好的真实 numpy**。

    这是既有契约（调参页/cv2 回退/一票既有测试都依赖）—— 绝不能因为引入
    懒转换而悄悄改变默认行为。"""
    calls = []

    def image2cv(img, **kw):
        calls.append(img)
        return np.zeros((2, 2, 3), np.uint8)

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(2, 2), monkeypatch, image2cv)

    frame = cam.read_raw()
    assert isinstance(frame, np.ndarray)     # 真 numpy, 不是代理
    assert len(calls) == 1                   # 立即发生了一次 image2cv


# ---------- 2. 开懒转换后 read_raw 不触发 image2cv ----------

def test_lazy_read_raw_skips_image2cv(monkeypatch):
    """开 set_lazy_convert(True): read_raw() 返回代理, **一次 image2cv 都不做**。

    这就是本优化的核心收益 —— TARGET/EXPLOSIVE 走 C++ find_blobs 时, 这份拷贝
    完全省掉（真机 conv 应从 9~16ms 掉到 ~0）。"""
    calls = []

    def image2cv(img, **kw):
        calls.append(img)
        return np.zeros((480, 640, 3), np.uint8)

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(640, 480), monkeypatch, image2cv)
    cam.set_lazy_convert(True)

    frame = cam.read_raw()
    assert not isinstance(frame, np.ndarray)   # 代理, 非真数组
    assert calls == []                         # 关键: 没有白做的拷贝
    assert cam.last_raw is not None            # 显示侧照旧能拿到 maix Image


# ---------- 2.5 调参页入口必须拿真 numpy（2026-10-09 真机回归） ----------

def test_lazy_read_returns_real_numpy_for_debug_pages(monkeypatch):
    """懒转换开启时, `read()` 也必须返回**真 ndarray**（返回契约如此）。

    真机回归: 调参页 `_analyze` 的 `cv2.resize(frame, ...)` 在 OpenCV 4.9 上
    对 LazyFrame 代理直接报 "src is not a numpy array, neither a scalar"
    （cv2 绑定不认 `__array__`/`__array_interface__`, PC 5.0 实测同样）——
    表现为切到 EXPLOSIVE/TARGET 页即 "[PAGE] step FAIL" 回 MAIN, 进不了调参页。
    read() 是调参页唯一取帧入口, 在这里收敛为真 numpy。
    """
    def image2cv(img, **kw):
        arr = np.zeros((4, 4, 3), np.uint8)
        arr[:, :, 0] = 10          # RGB 序, R=10
        return arr

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(4, 4), monkeypatch, image2cv)
    cam.set_lazy_convert(True)

    frame = cam.read()
    assert isinstance(frame, np.ndarray)       # 真 numpy, 不是代理
    assert tuple(frame[0, 0]) == (0, 0, 10)    # 已翻到 BGR
    assert cam.last_raw is not None            # last_raw 照旧是 maix Image


# ---------- 3. 只读 .shape 也不触发转换 ----------

def test_lazy_shape_metadata_is_zero_copy(monkeypatch):
    """`.shape` 从 maix 元数据直接给, 零拷贝 —— TARGET 里 `h, w = frame.shape[:2]`
    这种最常见的"非像素"用法不得引发转换。"""
    calls = []

    def image2cv(img, **kw):
        calls.append(img)
        return np.zeros((480, 640, 3), np.uint8)

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(640, 480), monkeypatch, image2cv)
    cam.set_lazy_convert(True)

    frame = cam.read_raw()
    assert frame.shape == (480, 640, 3)
    assert frame.ndim == 3
    assert frame.size == 480 * 640 * 3
    h, w = frame.shape[:2]
    assert (h, w) == (480, 640)
    assert calls == []                         # 全程没转


# ---------- 4. 真用像素时才转, 且只转一次 ----------

def test_lazy_pixel_access_converts_once(monkeypatch):
    """真正切片/取像素才触发转换, 且**同一帧只转一次**（结果缓存）。"""
    calls = []

    def image2cv(img, **kw):
        calls.append(img)
        arr = np.zeros((4, 4, 3), np.uint8)
        arr[:, :, 0] = 10          # RGB 序, R=10
        return arr

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(4, 4), monkeypatch, image2cv)
    cam.set_lazy_convert(True)

    frame = cam.read_raw()
    assert calls == []

    first = frame[0, 0]                        # 第一次真像素访问 -> 转换
    assert len(calls) == 1
    assert tuple(first) == (0, 0, 10)          # 通道已翻到 BGR

    _ = frame.numpy()                          # 再来几次都不该重复转
    _ = np.asarray(frame)
    assert len(calls) == 1                     # 仍是 1


def test_lazy_array_duck_typing(monkeypatch):
    """LazyFrame 必须能冒充 ndarray 通过 cv2 / np.asarray 等库调用。"""
    def image2cv(img, **kw):
        arr = np.zeros((3, 5, 3), np.uint8)
        arr[:, :, 0] = 7
        return arr

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(5, 3), monkeypatch, image2cv)
    cam.set_lazy_convert(True)
    frame = cam.read_raw()

    arr = np.asarray(frame)
    assert arr.shape == (3, 5, 3)
    assert int(arr[0, 0, 0]) == 0 and int(arr[0, 0, 2]) == 7


# ---------- 5.5 .copy() 代理（2026-10-09 调参页冻屏根因回归） ----------

def test_lazy_copy_returns_independent_ndarray(monkeypatch):
    """.copy() 必须像 ndarray.copy() 一样: 触发一次转换并返回**独立副本**。

    背景: threshold_tuner._highlight 第一行就是 frame_bgr.copy(), LazyFrame
    缺这个方法时调参页在主循环 tick 里每帧抛 AttributeError（被吞进 app.log）,
    LCD 冻结在按键前最后一帧 —— 真机表现为"按一下键画面卡死"。"""
    calls = []

    def image2cv(img, **kw):
        calls.append(img)
        arr = np.zeros((4, 4, 3), np.uint8)
        arr[:, :, 0] = 10          # RGB 序, R=10
        return arr

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(4, 4), monkeypatch, image2cv)
    cam.set_lazy_convert(True)
    frame = cam.read_raw()

    copied = frame.copy()
    assert isinstance(copied, np.ndarray)
    assert copied.shape == (4, 4, 3)
    assert tuple(copied[0, 0]) == (0, 0, 10)   # 已翻到 BGR
    assert len(calls) == 1                     # 转换恰好一次
    # 独立副本: 改副本不影响代理内部缓存
    copied[0, 0, 2] = 99
    assert int(frame.numpy()[0, 0, 2]) == 10


def test_threshold_tuner_highlight_accepts_lazy_frame():
    """调参页全链路回归: GksThresholdTuner._highlight( LazyFrame, mask ) 不抛。

    这是 CAM1 真机"按 KEY 切到 EXPLOSIVE 调试页 LCD 冻住"的直接死点
    （app/main.py step_active -> page.step -> _render_step -> _highlight）。
    """
    from framework.hal.platforms.lazy_frame import LazyFrame
    from modules.zw_opencv_module.debug.threshold_tuner import GksThresholdTuner

    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    lazy = LazyFrame(frame, convert=lambda raw: frame[:, :, ::-1],
                     shape_hw=(48, 64))
    mask = np.zeros((48, 64), np.uint8)
    out = GksThresholdTuner._highlight(lazy, mask)   # 修前: AttributeError
    assert out.shape == (48, 64, 3)


# ---------- 5. 转换失败退化为空数组而非抛异常 ----------

def test_lazy_convert_failure_degrades(monkeypatch):
    """image2cv 抛异常/返回 None 时, 应退化为空数组而非把整帧打断。

    处理器对"没有像素"的处理是零值心跳不断流; 抛异常会打断整帧并记 error。"""
    def image2cv(img, **kw):
        return None

    mod = _load_cam1(monkeypatch, image2cv)
    cam = _make_cam(mod, _FakeRaw(4, 4), monkeypatch, image2cv)
    cam.set_lazy_convert(True)
    frame = cam.read_raw()

    arr = frame.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.size == 0                       # 空数组, `.shape[:2]` 安全
    # 元数据仍有效（来自 maix raw）, 所以 shape 保持真实尺寸而非 (0,0) ——
    # 这正是所要的: 处理器能继续拿 `.shape` 算 ROI, 只是"这帧没有像素"。
    assert frame.shape[:2] == (4, 4)
    # 切片走空数组, 不抛、返回空
    assert frame[0:1, 0:1].size == 0


# ---------- 6. 平台无 set_lazy_convert 时 PipelineCamera 静默跳过 ----------

def test_pipeline_enables_lazy_convert_when_supported():
    """PipelineCamera 启动时应对支持该开关的相机开启懒转换。"""
    from modules.zw_opencv_module.pipeline_camera import PipelineCamera

    class _Cam:
        def __init__(self):
            self.lazy = None

        @property
        def camera_id(self):
            return "stub"

        @property
        def fps(self):
            return 30.0

        def set_lazy_convert(self, v):
            self.lazy = v

    # 不走完整 __init__（要 task 配置），只验证方法本身
    p = object.__new__(PipelineCamera)
    p.camera = _Cam()
    p._enable_lazy_convert()
    assert p.camera.lazy is True


def test_pipeline_skips_lazy_when_camera_lacks_switch():
    """相机没有 set_lazy_convert（linux/mock 本就返回真 numpy）时静默跳过, 不抛。"""
    from modules.zw_opencv_module.pipeline_camera import PipelineCamera

    class _Cam:
        @property
        def camera_id(self):
            return "stub"

        @property
        def fps(self):
            return 30.0

    p = object.__new__(PipelineCamera)
    p.camera = _Cam()
    p._enable_lazy_convert()                   # 不抛即通过


def test_pipeline_lazy_switch_failure_is_swallowed():
    """开关抛异常时会话必须能继续（退回急切转换 = 改前行为）。"""
    from modules.zw_opencv_module.pipeline_camera import PipelineCamera

    class _Cam:
        @property
        def camera_id(self):
            return "stub"

        @property
        def fps(self):
            return 30.0

        def set_lazy_convert(self, v):
            raise RuntimeError("boom")

    p = object.__new__(PipelineCamera)
    p.camera = _Cam()
    p._enable_lazy_convert()                   # 不抛即通过
