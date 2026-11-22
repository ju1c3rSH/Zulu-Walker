# -*- coding: utf-8 -*-
"""Battery HAL: interface contract and per-platform implementations.

Platform modules are loaded by file path instead of via the
``framework.hal.platforms.<platform>`` package, whose ``__init__`` also pulls
in ``.ai``, ``.camera``, ``.display`` and ``.uart`` and raises ImportError on a
PC without maix (see tests/test_gks_review_fixes.py:179).
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


# ---------------- contract ----------------

def test_reading_defaults_mean_unavailable():
    r = BatteryReading()
    assert (r.percent, r.millivolts) == (-1, -1)


def test_null_battery_satisfies_protocol_and_reports_absent():
    """The default implementation returns -1 (absent); HUD and sampler only check percent >= 0."""
    b = NullBattery()
    assert isinstance(b, Battery)
    assert b.read().percent == -1


# ---------------- maixcam2: sysfs ----------------

def test_maixcam2_maps_capacity_0_90_to_0_100(tmp_path, monkeypatch):
    """The official PMU calibrates ``capacity`` only up to 90, so scale to 100: a raw 45 shows as 50."""
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
    """A missing node means no battery is installed or the kernel driver has not started; treat as absent."""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    monkeypatch.setattr(mod, "_DIR", str(tmp_path))
    assert mod.MaixCam2Battery().read().percent == -1


def test_maixcam2_voltage_optional(tmp_path, monkeypatch):
    """An unreadable voltage_now does not affect the percentage."""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    monkeypatch.setattr(mod, "_DIR", str(tmp_path))
    (tmp_path / "capacity").write_text("72\n")
    r = mod.MaixCam2Battery().read()
    assert (r.percent, r.millivolts) == (80, -1)


def test_maixcam2_declares_kernel_poll_floor():
    """The kernel CW2015 driver updates only every 8s, so polling faster is pointless."""
    mod = _load("framework/hal/platforms/maixcam2/battery.py")
    assert mod.MaixCam2Battery.min_poll_s == 8.0


# ---------------- maixcam1 (Pro): AXP2101 ----------------

def test_maixcam1_without_pmu_degrades_to_absent():
    """Without maix.ext_dev (PC or old firmware), no battery is known at
    construction time; read() neither retries nor raises.
    """
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
    b = mod.MaixCam1Battery()      # on PC construction fails, so _pmu is None
    b._pmu = fake
    return b


def test_maixcam1_passes_through_reading():
    r = _battery_with(_FakePMU(42, 3700)).read()
    assert (r.percent, r.millivolts) == (42, 3700)


def test_maixcam1_zero_voltage_is_reported_as_unavailable():
    """0 mV is not a valid reading; reporting 0 would look like a dead cell."""
    assert _battery_with(_FakePMU(60, 0)).read().millivolts == -1


def test_maixcam1_pmu_exception_is_swallowed():
    r = _battery_with(_FakePMU(boom=True)).read()
    assert r.percent == -1
    assert "读失败" in r.reason


def test_null_battery_explains_itself():
    """On linux/mock this means "no fuel gauge", not "no battery"; -1 must
    distinguish the two cases.
    """
    r = NullBattery().read()
    assert r.percent == -1 and "无电量计" in r.reason
