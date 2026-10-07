# -*- coding: utf-8 -*-
"""find_blobs_maix 的**实参形态**契约（2026-10-07 台架滞后修正）。

背景: CAM1(cv181x) 台架实测订阅后只有 5.3fps(189ms/帧), 明显比去年基准
`工科赛上端/main.py` 滞后。逐行比对两年调用形态后定位到一处差异 ——
本模块此前硬编码 `merge=True, margin=1`, 而去年 task1/task2 都是
**裸 `merge=True`、不传 margin**（main.py:338 / main.py:390）。

margin 在 OpenMV 系 find_blobs 里会触发形态学膨胀 + 连通域重标记,
比 merge 本身重得多; CAM1 算力比 CAM2 弱一个量级, 扛不住。

本文件用假 maix 模块**捕获真实 kwargs** —— PC 上原有的降级测试只看返回值,
看不出"到底传了什么", 挡不住这类实参回归。
"""
import sys
import types

import pytest

from modules.zw_opencv_module.detectors.eod_color import maix_accel

pytest.importorskip("cv2")


class _FakeBlob:
    def __init__(self, x=0, y=0, w=10, h=10, pixels=100):
        self._x, self._y, self._w, self._h, self._pixels = x, y, w, h, pixels

    def x(self):
        return self._x

    def y(self):
        return self._y

    def w(self):
        return self._w

    def h(self):
        return self._h

    def cxf(self):
        return self._x + self._w / 2.0

    def cx(self):
        return int(self._x + self._w / 2)

    def cy(self):
        return int(self._y + self._h / 2)

    def pixels(self):
        return self._pixels


class _RecordingImage:
    """记录 find_blobs 的实参, 供断言。"""

    def __init__(self, recorder, accepts_margin=True):
        self._recorder = recorder
        self._accepts_margin = accepts_margin

    def format(self):
        # 声明为非 BGR, 让 _as_rgb888 原样返回（不触发 to_format 转换）
        return "FMT_RGB888"

    def find_blobs(self, **kwargs):
        if "margin" in kwargs and not self._accepts_margin:
            # 先记再抛: 记录的是"尝试过" —— 被拒的尝试也要计入, 否则断言不了
            # "先试带 margin, 失败后重试不带" 这个降级序列。
            self._recorder.append(dict(kwargs, _rejected=True))
            raise TypeError("find_blobs() got an unexpected keyword argument 'margin'")
        self._recorder.append(kwargs)
        return [_FakeBlob()]


@pytest.fixture()
def recorder(monkeypatch):
    """装上假 maix.image, 返回实参记录列表。"""
    calls = []
    image_mod = types.ModuleType("maix.image")
    maix_mod = types.ModuleType("maix")

    def _make(accepts_margin=True):
        return _RecordingImage(calls, accepts_margin)

    image_mod.cv2image = lambda *a, **k: None
    maix_mod.image = image_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)
    monkeypatch.setattr(maix_accel, "_HAVE_MAIX", True)
    monkeypatch.setattr(maix_accel, "_as_rgb888", lambda img: img)
    monkeypatch.setattr(maix_accel, "_BGR_GUARD_LOGGED", False)
    return calls


# ---- 不变量: 默认不传 margin（与去年基准逐字节一致）----

def test_merge_does_not_pass_margin_by_default(recorder):
    """merge=True 默认**不得**传 margin —— 这一条是本次修正的核心不变量。

    去年基准 main.py:390: `img.find_blobs([...], pixels_threshold=1000,
    area_threshold=1000, merge=True, roi=TARGET_ROI)` —— 没有 margin。
    """
    img = _RecordingImage(recorder)
    maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], merge=True)
    assert len(recorder) == 1
    assert "margin" not in recorder[0], "默认不得传 margin（与去年基准不一致）"
    assert recorder[0]["merge"] is True


def test_no_merge_path_never_passes_margin(recorder):
    """merge=False 时连 merge kwarg 都不该有, 更不会有 margin。"""
    img = _RecordingImage(recorder)
    maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], merge=False)
    assert len(recorder) == 1
    assert "margin" not in recorder[0]
    assert "merge" not in recorder[0]


# ---- 显式传 margin 时才出现, 且固件不认时降级 ----

def test_explicit_margin_is_forwarded(recorder):
    img = _RecordingImage(recorder)
    maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], merge=True, margin=2)
    assert recorder[0]["margin"] == 2


def test_explicit_margin_degrades_when_firmware_rejects(monkeypatch):
    """固件不认 margin 时抛 TypeError -> 自动降级到无 margin 的 merge 调用。

    降级后仍须成功返回 blob, 不能整条路径失败掉回 cv2（帧率差好几倍）。
    """
    calls = []
    image_mod = types.ModuleType("maix.image")
    maix_mod = types.ModuleType("maix")
    image_mod.cv2image = lambda *a, **k: None
    maix_mod.image = image_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)
    monkeypatch.setattr(maix_accel, "_HAVE_MAIX", True)
    monkeypatch.setattr(maix_accel, "_as_rgb888", lambda img: img)

    img = _RecordingImage(calls, accepts_margin=False)   # 拒绝 margin
    out = maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], merge=True, margin=1)
    assert out is not None and len(out) == 1, "应降级成功而不是整条路径失败"
    assert len(calls) == 2, "先试带 margin, 失败后重试不带"
    assert "margin" in calls[0] and calls[0].get("_rejected") is True
    # 关键: 降级后必须保住 merge=True（丢掉就是静默改变语义）
    assert "margin" not in calls[1]
    assert calls[1]["merge"] is True


# ---- 其余实参形态与去年一致 ----

def test_roi_passed_only_when_truthy(recorder):
    """roi=None 必须整体省略（pybind11 拒绝 None 列表）—— 全帧扫描形态。"""
    img = _RecordingImage(recorder)
    maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], roi=None)
    assert "roi" not in recorder[0]

    recorder.clear()
    img = _RecordingImage(recorder)
    maix_accel.find_blobs_maix(img, [0, 80, 40, 80, 10, 80], roi=[0, 169, 640, 197])
    assert recorder[0]["roi"] == [0, 169, 640, 197]


def test_thresholds_are_single_group_wrapped_in_list(recorder):
    """thresholds 必须是 [lab] 包裹形态 —— 去年 main.py 传的是 [threshold]。"""
    img = _RecordingImage(recorder)
    lab = [0, 80, 40, 80, 10, 80]
    maix_accel.find_blobs_maix(img, lab)
    assert recorder[0]["thresholds"] == [lab]
