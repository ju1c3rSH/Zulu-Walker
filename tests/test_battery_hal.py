# -*- coding: utf-8 -*-
"""电量 HAL: 接口约定 + 各平台实现。

平台实现按文件路径加载，不走 `framework.hal.platforms.<平台>` 包 —— 那个
`__init__` 会把 `.ai/.camera/.display/.uart` 一起拉进来，PC 上没有 maix 直接
ImportError（同 tests/test_gks_review_fixes.py:179 的说明）。
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from framework.hal.interface import Battery, BatteryReading, NullBattery


def _load(rel: str):
    path = _ROOT / rel
    name = f"{path.parent.name}_{path.stem}_hal"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------- 约定 ----------------

def test_reading_defaults_mean_unavailable():
    r = BatteryReading()
    assert (r.percent, r.millivolts) == (-1, -1)


def test_null_battery_satisfies_protocol_and_reports_absent():
    """默认实现返回 -1（无电池），且 HUD/采样器只判 percent >= 0。"""
    b = NullBattery()
    assert isinstance(b, Battery)
    assert b.read().percent == -1


# ---------------- maixcam2: sysfs ----------------

def test_maixcam2_maps_capacity_0_90_to_0_100(tmp_path, monkeypatch):
    """官方 pmu 的标准: `capacity` 只标定到 90, 放大到 100。45% 原始值显示 50%。"""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    monkeypatch.setattr(mod, "_DIR", str(tmp_path))

    (tmp_path / "capacity").write_text("90\n")
    (tmp_path / "voltage_now").write_text("4020000\n")
    r = mod.MaixCam2Battery().read()
    assert (r.percent, r.millivolts) == (100, 4020)

    (tmp_path / "capacity").write_text("45\n")
    assert mod.MaixCam2Battery().read().percent == 50

    (tmp_path / "capacity").write_text("0\n")
    assert mod.MaixCam2Battery().read().percent == 0


def test_maixcam2_missing_node_is_absent_not_crash(tmp_path, monkeypatch):
    """没装电池 / 内核没起驱动时节点不存在 —— 按无电池处理。"""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    monkeypatch.setattr(mod, "_DIR", str(tmp_path))
    assert mod.MaixCam2Battery().read().percent == -1


def test_maixcam2_voltage_optional(tmp_path, monkeypatch):
    """voltage_now 读不到不影响百分比。"""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    monkeypatch.setattr(mod, "_DIR", str(tmp_path))
    (tmp_path / "capacity").write_text("72\n")
    r = mod.MaixCam2Battery().read()
    assert (r.percent, r.millivolts) == (80, -1)


def test_maixcam2_declares_kernel_poll_floor():
    """内核 CW2015 驱动 8s 才更新一次，读更快是白读。"""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    assert mod.MaixCam2Battery.min_poll_s == 8.0


# ---------------- maixcam1 (Pro): AXP2101 ----------------

def test_maixcam1_without_pmu_degrades_to_absent():
    """PC / 老固件没有 maix.ext_dev: 构造时就确定没电池，read() 不重试也不抛。"""
    try:
        import maix.ext_dev  # noqa: F401
        pytest.skip("本机装了 maix，测不到降级路径")
    except ImportError:
        pass

    mod = _load("framework/hal/platforms/maixcam1/battery.py")
    b = mod.MaixCam1Battery()
    assert b._pmu is None
    r = b.read()
    assert r.percent == -1
    assert "pmu" in r.reason, "原因要跟着读数走, HAL 不再自己打日志"


class _FakePMU:
    def __init__(self, percent=42, mv=3700, boom=False):
        self._p, self._mv, self._boom = percent, mv, boom

    def get_bat_percent(self):
        if self._boom:
            raise RuntimeError("i2c nack")
        return self._p

    def get_bat_vol(self):
        return self._mv


def _battery_with(fake):
    mod = _load("framework/hal/platforms/maixcam1/battery.py")
    b = mod.MaixCam1Battery()      # PC 上构造必然失败, _pmu 为 None
    b._pmu = fake
    return b


def test_maixcam1_passes_through_reading():
    r = _battery_with(_FakePMU(42, 3700)).read()
    assert (r.percent, r.millivolts) == (42, 3700)


def test_maixcam1_zero_voltage_is_reported_as_unavailable():
    """0mV 不是有效读数, 报 0 会被下游当"电芯已死"。"""
    assert _battery_with(_FakePMU(60, 0)).read().millivolts == -1


def test_maixcam1_pmu_exception_is_swallowed():
    r = _battery_with(_FakePMU(boom=True)).read()
    assert r.percent == -1
    assert "读失败" in r.reason


def test_null_battery_explains_itself():
    """linux/mock 上不是"没电池", 是"这个平台没有电量计" —— -1 要能区分。"""
    r = NullBattery().read()
    assert r.percent == -1 and "无电量计" in r.reason
