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
