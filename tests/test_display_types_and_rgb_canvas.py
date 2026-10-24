# -*- coding: utf-8 -*-
"""C2: 原生 RGB888 画布路径 + 帧格式/能力类型（PC 端, 无 maix）。

钉住三件事:
  1. 三个 tuner 的 _to_maix 走 cv2image(..., bgr=False) 且交的是**连续 RGB**;
  2. 通道序不互换（相机交来的 BGR 是负步长视图, 最容易在这里红蓝互换）;
  3. PixelFormat.of() 与四个平台 display 的 caps()/width()/height() 契约。
"""
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
        self.append({"bgr": bgr, "copy": copy, "contiguous": arr.flags["C_CONTIGUOUS"]})
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
    """只装 display.py 需要的最小子集, 不拖 camera/ai 的依赖。"""
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
    image.Color = types.SimpleNamespace(from_rgb=lambda r, g, b: (r << 16) | (g << 8) | b)
    maix.display, maix.err, maix.image = display, err, image
    for name, mod in (("maix", maix), ("maix.display", display),
                      ("maix.err", err), ("maix.image", image)):
        monkeypatch.setitem(sys.modules, name, mod)
    # types.py 的表在 import 时建; PC 上当时没有 maix, 这里按同一条构建路径重建。
    monkeypatch.setattr(dt, "_FMT_MAP", dt._build_format_map())
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
    bgr = _bgr_red_negative_stride_view()
    img = tuner._to_maix(bgr)

    assert len(fake_maix) == 1
    assert fake_maix[0]["bgr"] is False
    # 负步长视图不能直接喂 cv2image: bgr=False 只换标记, 通道会红蓝互换
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


def test_maixcam1_caps_survives_dead_display(fake_maix):
    mod = _load_display_module("maixcam1")
    disp = mod.MaixCam1Display()
    disp._disp = None                     # caps() 必须永不抛异常
    caps = disp.caps()
    assert (caps.width, caps.height) == (552, 368)
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

def test_warn_slow_format_logs_once(monkeypatch):
    from framework.display import gate

    logs = []
    monkeypatch.setattr(gate, "log_print", lambda m="": logs.append(m))
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "_ACCEPTED_BY_TYPE", {})

    class _Disp:
        def caps(self):
            return dt.DisplayCaps(
                width=552, height=368, native_format=dt.PixelFormat.RGB888,
                accepted=frozenset({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888}),
                encode_in_show=True)

    disp, frame = _Disp(), np.zeros((4, 4, 3), np.uint8)
    assert gate.warn_slow_format(disp, frame) is True
    assert gate.warn_slow_format(disp, frame) is True
    assert len(logs) == 1
    assert "slow path" in logs[0] and "BGR888" in logs[0]


def test_warn_slow_format_transparent_through_proxy(monkeypatch, fake_maix):
    """调试页拿到的是代理; 闸门必须看真实 display 的 caps, 不能凭空报慢路径。"""
    from framework.display import gate

    logs = []
    monkeypatch.setattr(gate, "log_print", lambda m="": logs.append(m))
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "_ACCEPTED_BY_TYPE", {})

    class _Real:
        def caps(self):
            return dt.DisplayCaps(
                width=552, height=368, native_format=dt.PixelFormat.RGB888,
                accepted=frozenset({dt.PixelFormat.NATIVE, dt.PixelFormat.RGB888}),
                encode_in_show=True)

    class _Proxy:
        def __init__(self, real):
            self._real = real

        def caps(self):
            return self._real.caps()

    fast = _FakeImage(np.zeros((1, 1, 3), np.uint8))   # RGB888 -> 快路径
    assert dt.PixelFormat.of(fast) is dt.PixelFormat.RGB888
    assert gate.warn_slow_format(_Proxy(_Real()), fast) is False
    assert logs == []


def test_warn_slow_format_never_raises(monkeypatch):
    from framework.display import gate

    class _Disp:
        def caps(self):
            raise RuntimeError("no caps")

    assert gate.warn_slow_format(_Disp(), np.zeros((2, 2, 3), np.uint8)) is False
