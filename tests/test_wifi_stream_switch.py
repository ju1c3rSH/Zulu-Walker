# -*- coding: utf-8 -*-
"""HTTP 推流开关与编码路径测试。

背景: MaixCAM1 上 VENC 是单例硬件资源, 与 Display.show() 内的 MaixVision 编码互斥。
推流默认必须关闭 (否则启动即抢 VENC -> CVI_VENC_GetStream failed 0xc0078012);
开启时必须先 to_jpeg() 再 write() (write 不接受原始 RGB), 且按 fps 时间节流。
"""
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.zw_wifi_stream as W


class _Jpg:
    pass


class _FakeImg:
    def __init__(self, w=640, h=480):
        self._w, self._h = w, h
        self.resized = None
        self.jpeg_calls = 0

    def width(self):
        return self._w

    def height(self):
        return self._h

    def resize(self, w, h):
        img = _FakeImg(w, h)
        img.resized = (w, h)
        return img

    def to_jpeg(self, quality=80):
        self.jpeg_calls += 1
        return _Jpg()


class _FakeSrv:
    def __init__(self):
        self.written = []

    def set_html(self, h):
        pass

    def start(self):
        pass

    def host(self):
        return "0.0.0.0"

    def port(self):
        return 8000

    def write(self, img):
        if not isinstance(img, _Jpg):
            raise TypeError("write() requires JPEG, got %r" % type(img).__name__)
        self.written.append(img)


@pytest.fixture()
def fake_maix(monkeypatch):
    img_mod = types.ModuleType("maix.image")
    img_mod.Image = _FakeImg
    http_mod = types.ModuleType("maix.http")
    http_mod.JpegStreamer = _FakeSrv
    maix = types.ModuleType("maix")
    maix.image = img_mod
    maix.http = http_mod
    monkeypatch.setitem(sys.modules, "maix", maix)
    monkeypatch.setitem(sys.modules, "maix.image", img_mod)
    monkeypatch.setitem(sys.modules, "maix.http", http_mod)
    monkeypatch.setattr(W, "_streamer", None, raising=False)
    monkeypatch.setattr(W, "_disabled_reason", "", raising=False)
    return img_mod


def test_disabled_by_default_creates_no_streamer(fake_maix):
    """默认不创建 streamer —— 关闭时不得占用任何 VENC 资源。"""
    W.init(machine=None, config={})
    assert W.get_streamer() is None
    assert "enabled" in W.disabled_reason() or W.disabled_reason() != ""


def test_enabled_creates_streamer(fake_maix):
    W.init(machine=None, config={"stream": {"enabled": True, "fps": 5}})
    assert W.get_streamer() is not None


def test_unsupported_platform_refuses(fake_maix, monkeypatch):
    """没有 to_jpeg 的平台必须拒绝, 而不是运行时黑屏。"""
    class NoJpeg:
        pass
    fake_maix.Image = NoJpeg
    W.init(machine=None, config={"stream": {"enabled": True}})
    assert W.get_streamer() is None


def test_write_requires_jpeg_and_downscales(fake_maix):
    """回归: 曾经把原始 RGB 直接喂 write() -> 必然失败且静默。"""
    W.init(machine=None, config={"stream": {"enabled": True, "fps": 100,
                                            "max_width": 320, "quality": 80}})
    st = W.get_streamer()
    st.start_async()
    try:
        for _ in range(20):
            st.push_frame(_FakeImg())
        deadline = time.time() + 2.0
        while time.time() < deadline and not st._server.written:
            time.sleep(0.02)
        assert st._server.written, "no JPEG written"
        assert st._failed == 0
    finally:
        st.stop()


def test_fps_throttle_limits_writes(fake_maix):
    """fps 节流: 高帧率喂入时写入次数应受 fps 限制, 而不是每帧一次。"""
    W.init(machine=None, config={"stream": {"enabled": True, "fps": 5,
                                            "max_width": 320}})
    st = W.get_streamer()
    st.start_async()
    try:
        t0 = time.time()
        fed = 0
        while time.time() - t0 < 1.0:
            st.push_frame(_FakeImg())
            fed += 1
            time.sleep(1.0 / 60)
        time.sleep(0.3)
        assert fed > 40
        assert len(st._server.written) <= 8, "fps=5 时 1 秒不应写出 %d 次" % len(st._server.written)
    finally:
        st.stop()
