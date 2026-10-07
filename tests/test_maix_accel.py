# -*- coding: utf-8 -*-
"""MaixPy acceleration layer — graceful-degradation contract on PC.

On the device these paths hit MaixCV C++ / NPU; on PC (no maix module) every
helper must return None/False so callers fall back to cv2 transparently.
"""
import numpy as np
import pytest

from modules.zw_opencv_module.detectors.eod_color import maix_accel
from modules.zw_opencv_module.detectors.eod_color import EodColorDetector

pytest.importorskip("cv2")


def test_find_blobs_maix_unavailable_on_pc():
    assert maix_accel.find_blobs_maix(None, [0, 80, 40, 80, 10, 80]) is None
    # even with an img argument, no maix module -> None (caller falls back)
    assert maix_accel.find_blobs_maix(object(), [0, 80, 40, 80, 10, 80]) is None


def test_find_blobs_maix_merge_kwarg_unavailable_on_pc():
    # merge 路径同样优雅降级（v2.1 工科赛分支新增参数）
    assert maix_accel.find_blobs_maix(None, [0, 80, 40, 80, 10, 80], merge=True) is None


def test_qr_helpers_unavailable_on_pc():
    assert maix_accel.qr_detect_npu(object()) is None
    assert maix_accel.qr_detect_zbar(object()) is None


def test_context_probe_without_maix_returns_none():
    assert maix_accel.maix_image_from_context(None) is None
    class _Cam:
        last_raw = object()
    assert maix_accel.maix_image_from_context({"camera": _Cam()}) is None


def test_detector_with_maix_img_none_uses_cv2_path():
    """Explicit maix_img=None must behave exactly like the pure cv2 path."""
    W, H = 320, 320
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[200:260, 130:190] = (60, 60, 200)   # red block
    det = EodColorDetector(
        [{"id": 1, "name": "red", "lab": [0, 80, 40, 80, 10, 80]}],
        min_area_px=100)
    result = det.detect(frame, maix_img=None)
    assert result.target_found and result.dev_id == 1


# ---------- BGR -> RGB888 兜底（imlib 把 FMT_BGR888 当 RGB888 用） ----------

class _Format:
    FMT_RGB888 = 1
    FMT_BGR888 = 2
    FMT_BGRA8888 = 4


class _FakeImg:
    def __init__(self, fmt):
        self._fmt = fmt
        self.conversions = 0

    def format(self):
        return self._fmt

    def to_format(self, fmt):
        self.conversions += 1
        self._fmt = fmt
        return self


@pytest.fixture()
def fake_maix_formats(monkeypatch):
    import types
    monkeypatch.setattr(maix_accel, "_maix_image",
                        types.SimpleNamespace(Format=_Format()))
    monkeypatch.setattr(maix_accel, "_BGR_GUARD_LOGGED", False)


def test_as_rgb888_converts_bgr_marked_image(fake_maix_formats):
    """BGR 标记图必须转成真 RGB888, 否则 find_blobs 里红蓝互换（台架 blue->C1）。"""
    img = _FakeImg(_Format.FMT_BGR888)
    out = maix_accel._as_rgb888(img)
    assert out is img
    assert img.conversions == 1
    assert img.format() == _Format.FMT_RGB888


def test_as_rgb888_converts_bgra_marked_image(fake_maix_formats):
    img = _FakeImg(_Format.FMT_BGRA8888)
    maix_accel._as_rgb888(img)
    assert img.conversions == 1 and img.format() == _Format.FMT_RGB888


def test_as_rgb888_keeps_rgb888_untouched(fake_maix_formats):
    img = _FakeImg(_Format.FMT_RGB888)
    out = maix_accel._as_rgb888(img)
    assert out is img and img.conversions == 0


def test_as_rgb888_without_format_attribute(fake_maix_formats):
    """老固件 / 非 Image 对象: 不抛异常, 原样返回。"""
    assert maix_accel._as_rgb888(object()) is not None

