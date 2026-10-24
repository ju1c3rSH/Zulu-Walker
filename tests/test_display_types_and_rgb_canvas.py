# -*- coding: utf-8 -*-
"""C2: 原生 RGB888 画布路径 + 帧格式/能力类型（PC 端, 无 maix）。

钉住三件事:
  1. 三个 tuner 的 _to_maix 走 cv2image(..., bgr=False) 且交的是**连续 RGB**;
  2. 通道序不互换（纯红像素必须落成 R 高）;
  3. PixelFormat.of() / 四个平台 display 的 caps()/width()/height() 契约。
"""
import importlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from framework.display import types as dt  # noqa: E402
from framework.hal.platforms.linux.display import LinuxDisplay  # noqa: E402
from framework.hal.platforms.mock.display import MockDisplay  # noqa: E402


# ---------------- fake maix ----------------

class _FakeFormat:
    FMT_RGB888 = 11
    FMT_BGR888 = 12
    FMT_GRAYSCALE = 13
    FMT_JPEG = 14
    FMT_YUYV = 15


class _FakeImage:
    """最小 maix Image 替身: 记下像素 + 报告格式。"""

    def __init__(self, pixels, fmt=_FakeFormat.FMT_RGB888):
        self.pixels = np.array(pixels, copy=True)
        self._fmt = fmt

    def format(self):
        return self._fmt

    def width(self):
        return int(self.pixels.shape[1])

    def height(self):
        return int(self.pixels.shape[0])


class _CvCalls(list):
    """记录每次 cv2image 的实参, 顺带断言"交进去的真连续"。"""

    def __call__(self, arr, bgr=True, copy=False):
        self.append({"bgr": bgr, "copy": copy,
                     "contiguous": bool(arr.flags["C_CONTIGUOUS"])})
        return _FakeImage(arr, _FakeFormat.FMT_RGB888)


_FAKE_DISPLAY_SIZE = [552, 368]


class _FakeMaixDisplayDev:
    def __init__(self, *args, **kwargs):
        pass

    def width(self):
        return _FAKE_DISPLAY_SIZE[0]

    def height(self):
        return _FAKE_DISPLAY_SIZE[1]

    def show(self, frame):
        return 0

    def close(self):
        pass


def _install_fake_maix(monkeypatch):
    """只装 display.py / types.py 需要的最小子集, 不拖 camera/ai 依赖。"""
    maix = types.ModuleType("maix")
    display = types.ModuleType("maix.display")
    err = types.ModuleType("maix.err")
    image = types.ModuleType("maix.image")
    display.Display = _FakeMaixDisplayDev
    err.Err = types.SimpleNamespace(ERR_NONE=0)
    image.Format = _FakeFormat
    calls = _CvCalls()
    image.cv2image = calls
    image.image2cv = lambda img, **kw: np.array(img.pixels, copy=True)
    image.Color = types.SimpleNamespace(
        from_rgb=lambda r, g, b: (r << 16) | (g << 8) | b)
    maix.display, maix.err, maix.image = display, err, image
    for name, mod in (("maix", maix), ("maix.display", display),
                      ("maix.err", err), ("maix.image", image)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _load_display_module(name):
    path = os.path.join(_ROOT, "framework", "hal", "platforms", name, "display.py")
    spec = importlib.util.spec_from_file_location("_c2_" + name + "_display", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def fake_maix(monkeypatch):
    return _install_fake_maix(monkeypatch)


# ---------------- _to_maix: fast path + 通道序 ----------------

def _laser():
    from modules.zw_opencv_module.debug.laser_calib_tuner import GksLaserCalibTuner
    return GksLaserCalibTuner(get_processor=lambda: None, send_laser=lambda on: None)


def _rescue():
    from modules.zw_opencv_module.debug.rescue_tuner import GksRescueTuner
    return GksRescueTuner(get_processor=lambda: None, get_ai=lambda: None)


def _threshold():
    from modules.zw_opencv_module.debug.threshold_tuner import GksThresholdTuner
    return GksThresholdTuner(group_indices=[0, 1, 2])


def _bgr_red_negative_stride_view():
    """相机交出来的形态: 底层 RGB, BGR 只是负步长视图。"""
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[:, :, 0] = 255                      # 纯红
    return rgb[:, :, ::-1]                  # -> BGR 字节, 末轴负步长


@pytest.mark.parametrize("make_tuner", [_laser, _rescue, _threshold])
def test_to_maix_uses_native_rgb888_path(fake_maix, make_tuner):
    tuner = make_tuner()
    img = tuner._to_maix(_bgr_red_negative_stride_view())

    assert len(fake_maix) == 1
    assert fake_maix[0]["bgr"] is False
    assert fake_maix[0]["copy"] is True
    # 负步长视图不能直接喂 cv2image: pybind11 会拒收或整帧镜像
    assert fake_maix[0]["contiguous"] is True
    assert img.format() == _FakeFormat.FMT_RGB888


@pytest.mark.parametrize("make_tuner", [_laser, _rescue, _threshold])
def test_to_maix_preserves_channel_order(fake_maix, make_tuner):
    """回归: 纯红 BGR 像素必须落成 R 高, 不能变成蓝。"""
    tuner = make_tuner()
    img = tuner._to_maix(_bgr_red_negative_stride_view())

    r, g, b = (int(v) for v in img.pixels[0, 0])
    assert r == 255 and g == 0 and b == 0


# ---------------- PixelFormat.of ----------------

def test_pixel_format_of_ndarray_is_bgr888():
    assert dt.PixelFormat.of(np.zeros((4, 4, 3), np.uint8)) is dt.PixelFormat.BGR888


@pytest.mark.parametrize("shape", [(480, 640), (480, 640, 4)])
def test_pixel_format_of_non_bgr_ndarray_is_none(shape):
    """灰度/BGRA 不是 BGR888; 误判会让格式闸门放行慢或不支持的格式。"""
    assert dt.PixelFormat.of(np.zeros(shape, np.uint8)) is None


def test_pixel_format_of_maix_image_is_rgb888(fake_maix):
    img = _FakeImage(np.zeros((1, 1, 3), np.uint8))
    assert dt.PixelFormat.of(img) is dt.PixelFormat.RGB888


def test_pixel_format_of_lazy_frame_is_none():
    class _Lazy:
        numpy = staticmethod(lambda: None)

    assert dt.PixelFormat.of(_Lazy()) is None


def test_pixel_format_of_unknown_format_is_none(fake_maix):
    img = _FakeImage(np.zeros((1, 1, 3), np.uint8), _FakeFormat.FMT_YUYV)
    assert dt.PixelFormat.of(img) is None


def test_pixel_format_of_works_when_maix_arrives_after_import(monkeypatch):
    """回归(缺陷B): 表的构建不能只在 import 时试一次, 否则 maix 晚到就一直判 None。"""
    monkeypatch.delitem(sys.modules, "maix.image", raising=False)
    monkeypatch.delitem(sys.modules, "maix", raising=False)
    assert dt.PixelFormat.of(_FakeImage(np.zeros((1, 1, 3), np.uint8))) is None

    image = types.ModuleType("maix.image")
    image.Format = _FakeFormat
    maix = types.ModuleType("maix")
    maix.image = image
    monkeypatch.setitem(sys.modules, "maix", maix)
    monkeypatch.setitem(sys.modules, "maix.image", image)

    # 不碰任何私有表: 公开入口自己补建
    rgb = _FakeImage(np.zeros((1, 1, 3), np.uint8))
    gray = _FakeImage(np.zeros((1, 1, 3), np.uint8), _FakeFormat.FMT_GRAYSCALE)
    assert dt.PixelFormat.of(rgb) is dt.PixelFormat.RGB888
    assert dt.PixelFormat.of(gray) is dt.PixelFormat.GRAY8


# ---------------- Frame.seq 哨兵 ----------------

def test_frame_seq_defaults_to_unset_sentinel():
    """回归(缺陷C): 0 会被 seq 比较读成"与首帧同帧", 于是每帧都被丢。"""
    f = dt.Frame(handle=object(), fmt=dt.PixelFormat.RGB888, w=1, h=1)
    assert f.seq == -1


# ---------------- 四个平台 display 的 caps() / width() / height() ----------------

def test_maixcam1_caps(fake_maix):
    mod = _load_display_module("maixcam1")
    disp = mod.MaixCam1Display()
    caps = disp.caps()
    assert caps.accepted == frozenset({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888})
    assert caps.native_format is dt.PixelFormat.RGB888
    assert caps.encode_in_show is True
    assert (caps.width, caps.height) == (552, 368)
    assert (disp.width(), disp.height()) == (552, 368)


def test_maixcam1_caps_agrees_with_size_on_dead_display(fake_maix):
    """回归(缺陷E): caps() 与 width()/height() 必须说同一件事, 否则调试页 resize(0,0)。"""
    mod = _load_display_module("maixcam1")
    disp = mod.MaixCam1Display()
    disp._disp = None                     # caps() 必须永不抛异常
    caps = disp.caps()
    assert (caps.width, caps.height) == (disp.width(), disp.height()) == (552, 368)
    assert dt.PixelFormat.RGB888 in caps.accepted


def test_maixcam2_caps(fake_maix):
    _FAKE_DISPLAY_SIZE[:] = [640, 480]
    try:
        mod = _load_display_module("maixcam2")
        disp = mod.MaixCam2Display()
        caps = disp.caps()
        assert caps.accepted == frozenset({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888})
        assert caps.native_format is dt.PixelFormat.RGB888
        assert caps.encode_in_show is True
        assert (caps.width, caps.height) == (640, 480)
        assert (disp.width(), disp.height()) == (640, 480)
    finally:
        _FAKE_DISPLAY_SIZE[:] = [552, 368]


def test_maixcam2_caps_agrees_with_size_on_dead_display(fake_maix):
    mod = _load_display_module("maixcam2")
    disp = mod.MaixCam2Display()
    disp._disp = None
    caps = disp.caps()
    assert (caps.width, caps.height) == (disp.width(), disp.height()) == (640, 480)


def test_linux_display_caps():
    disp = LinuxDisplay()
    caps = disp.caps()
    assert caps.accepted == frozenset({dt.PixelFormat.BGR888})
    assert caps.native_format is dt.PixelFormat.BGR888
    assert caps.encode_in_show is False
    assert disp.width() > 0 and disp.height() > 0


def test_mock_display_caps_and_size_follows_frame():
    disp = MockDisplay()
    assert (disp.width(), disp.height()) == (640, 480)
    disp.show(np.zeros((368, 552, 3), np.uint8))
    assert (disp.width(), disp.height()) == (552, 368)
    caps = disp.caps()
    assert caps.accepted == frozenset({dt.PixelFormat.BGR888})
    assert caps.encode_in_show is False


# ---------------- warn_once 格式闸门 ----------------

class _CapsDisp:
    """只声明 accepted 的最小 display。"""

    def __init__(self, accepted):
        self._accepted = frozenset(accepted)

    def caps(self):
        return dt.DisplayCaps(width=552, height=368,
                              native_format=dt.PixelFormat.RGB888,
                              accepted=self._accepted, encode_in_show=True)


@pytest.fixture()
def gate_logs(monkeypatch):
    from framework.display import gate
    logs = []
    monkeypatch.setattr(gate, "log_print", lambda m="": logs.append(m))
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "_ACCEPTED_BY_TYPE", {})
    return logs


def test_warn_slow_format_logs_once(gate_logs):
    from framework.display import gate
    disp = _CapsDisp({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888})
    frame = np.zeros((4, 4, 3), np.uint8)

    assert gate.warn_slow_format(disp, frame) is True
    assert gate.warn_slow_format(disp, frame) is True
    assert len(gate_logs) == 1
    assert "BGR888" in gate_logs[0] and "accepted" in gate_logs[0]


def test_warn_slow_format_silent_on_fast_path(gate_logs, fake_maix):
    from framework.display import gate
    disp = _CapsDisp({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888})
    frame = _FakeImage(np.zeros((1, 1, 3), np.uint8))    # FMT_RGB888

    assert dt.PixelFormat.of(frame) is dt.PixelFormat.RGB888
    assert gate.warn_slow_format(disp, frame) is False
    assert gate_logs == []


@pytest.mark.parametrize("wrap", ["plain", "nested", "falsy_inner"])
def test_warn_slow_format_works_through_any_proxy(gate_logs, wrap):
    """回归(缺陷D): 闸门只认公开 caps(); 代理嵌代理 / 假值 _real 也要判对。"""
    from framework.display import gate
    real = _CapsDisp({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888})

    class _Falsy:
        def __bool__(self):
            return False

        def caps(self):
            return real.caps()

    class _Proxy:
        def __init__(self, inner):
            self._real = inner

        def __getattr__(self, name):        # 与 _ThrottledPageDisplay 同款透传
            return getattr(self._real, name)

    disp = {"plain": lambda: _Proxy(real),
            "nested": lambda: _Proxy(_Proxy(real)),
            "falsy_inner": lambda: _Proxy(_Falsy())}[wrap]()

    # slow path 仍要被判出来, 且只告警一次（缓存键取公开类型, 不再嗅探 _real）
    assert gate.warn_slow_format(disp, np.zeros((4, 4, 3), np.uint8)) is True
    assert gate.warn_slow_format(disp, np.zeros((4, 4, 3), np.uint8)) is True
    assert len(gate_logs) == 1


def test_warn_slow_format_never_raises(monkeypatch):
    from framework.display import gate

    class _Disp:
        def caps(self):
            raise RuntimeError("no caps")

    assert gate.warn_slow_format(_Disp(), np.zeros((2, 2, 3), np.uint8)) is False


def test_warn_slow_format_needs_no_caps_attr():
    from framework.display import gate
    assert gate.warn_slow_format(object(), np.zeros((2, 2, 3), np.uint8)) is False
