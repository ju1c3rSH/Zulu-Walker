# -*- coding: utf-8 -*-
"""激光校准功能 tests — 0x25 协议见 test_protocol_gks.py, 本文件覆盖:

  1. load_laser_calib_file 解析器（key=value / 注释 / 非法行容错）
  2. GksTaskProcessor: laser_cx 运行时校准 + err 零点平移 + mtime 热重载
  3. GksLaserCalibTuner.detect_dot 分层检测（白核+红环 合成帧 / 高光排除）
  4. LASER 页 SAVE 落盘与 laser_is_on/on_leave 兜底语义
"""
import cv2
import numpy as np
import types

import pytest

import modules.zw_opencv_module.processors.gks_task_processor as gp
from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor,
    MODE_TARGET,
    load_gks_config,
    load_laser_calib_file,
)
from modules.zw_opencv_module.debug import laser_calib_tuner as lt


# ---------- fixtures ----------

def lab_threshold_for(bgr, half=25):
    px = np.uint8([[bgr]])
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
    l_om = lab[0] / 2.55
    a = int(lab[1]) - 128
    b = int(lab[2]) - 128
    return [max(0, int(l_om - half)), min(100, int(l_om + half)),
            max(-128, a - half), min(127, a + half),
            max(-128, b - half), min(127, b + half)]


RED = (0, 0, 255)   # BGR


@pytest.fixture()
def base_cfg():
    c = load_gks_config(config_path="/nonexistent/eod_config.yaml")
    c["thresholds_file"] = ""      # 隔离设备端 /my-threshold 文件
    c["laser"]["calib_file"] = ""  # 隔离激光校准文件
    return c


@pytest.fixture()
def log_capture(monkeypatch):
    msgs = []
    monkeypatch.setattr(gp, "log_print", lambda m="": msgs.append(m))
    return msgs


@pytest.fixture()
def laser_frame():
    """640x480: 红色 120px 方块 cx=480（与主测试文件 target_frame 同构）。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[180:300, 420:540] = RED
    return frame


# ---------- 1. 校准文件解析器 ----------

def test_laser_parser_reads_valid_lines(tmp_path):
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("# comment\nlaser_cx=412\nlum_min=210\nmin_area = 12\n",
                 encoding="utf-8")
    assert load_laser_calib_file(str(f)) == {
        "laser_cx": 412, "lum_min": 210, "min_area": 12}


def test_laser_parser_ignores_bad_lines_and_unknown_keys(tmp_path):
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("laser_cx=abc\nunknown=1\n\n# only comment\n",
                 encoding="utf-8")
    assert load_laser_calib_file(str(f)) == {}


def test_laser_parser_missing_file_is_noop(tmp_path):
    assert load_laser_calib_file(str(tmp_path / "nope.txt")) == {}


# ---------- 2. laser_cx 运行时校准 + err 零点 ----------

def test_laser_cx_default_and_runtime_set(base_cfg):
    p = GksTaskProcessor(config=base_cfg)
    assert p.target_zero_px() == 0                     # 默认未校准
    p.set_runtime_param("target", "laser_cx", 412)
    assert p.target_zero_px() == 412
    p.set_runtime_param("target", "laser_cx", 0)       # RESET 回中心
    assert p.target_zero_px() == 0


def test_laser_cx_unknown_key_and_immutable_calib_file(base_cfg):
    p = GksTaskProcessor(config=base_cfg)
    with pytest.raises(ValueError):
        p.set_runtime_param("target", "laser_cx_not_exist", 1)
    with pytest.raises(ValueError):
        p.set_runtime_param("laser", "calib_file", "/other")


def test_target_err_shifts_zero_keeps_gain(base_cfg, laser_frame):
    """校准只平移零点, 除数 w/2 不变: 同一目标, err 随 laser_cx 平移。"""
    base_cfg["target"]["lab"] = {"red": lab_threshold_for(RED),
                                 "green": [], "blue": []}
    p = GksTaskProcessor(config=base_cfg)
    p.set_task(MODE_TARGET, (0, 1, 0))
    # 未校准: (480-320)/320 = +0.5 -> 5000（出厂口径不变）
    assert p.process(laser_frame).result_data["err_fixed"] == 5000
    # 校准到目标自身 -> err=0
    p.set_runtime_param("target", "laser_cx", 480)
    assert p.process(laser_frame).result_data["err_fixed"] == 0
    # 零点左移 280 -> (480-200)/320 = +0.875
    p.set_runtime_param("target", "laser_cx", 200)
    assert p.process(laser_frame).result_data["err_fixed"] == 8750


def test_laser_cx_survives_reset_latch(base_cfg, laser_frame):
    """CMD_STOP 清锁存不动校准: 校准是几何量, 不是任务态。"""
    base_cfg["target"]["lab"] = {"red": lab_threshold_for(RED),
                                 "green": [], "blue": []}
    p = GksTaskProcessor(config=base_cfg)
    p.set_task(MODE_TARGET, (0, 1, 0))
    p.set_runtime_param("target", "laser_cx", 480)
    p.reset()
    assert p.target_zero_px() == 480


def test_laser_calib_hot_reload(base_cfg, tmp_path, log_capture):
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("laser_cx=380\n", encoding="utf-8")
    base_cfg["laser"]["calib_file"] = str(f)
    p = GksTaskProcessor(config=base_cfg)
    assert p.target_zero_px() == 380
    n = len(log_capture)

    # 文件未变 -> 轮询不重载
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert len(log_capture) == n

    # 改文件 -> 下一轮 2Hz 轮询重载
    f.write_text("laser_cx=412\n", encoding="utf-8")
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert p.target_zero_px() == 412
    assert any("laser calib" in m and "reloaded" in m
               for m in log_capture[n:])


# ---------- 3. 红点分层检测（合成帧） ----------

class _FakeProcessor:
    def __init__(self, laser_cfg):
        self.config = {"laser": dict(laser_cfg)}


def _laser_spot_frame(cx=200, cy=150, ring_r=6, core_r=2, bg=60):
    """合成激光点: 灰底 + 红环(饱和红) + 白核(全饱和) — 调研的成像三层结构。"""
    frame = np.full((480, 640, 3), bg, dtype=np.uint8)
    cv2.circle(frame, (cx, cy), ring_r, (0, 0, 255), -1)        # 红环/整盘
    cv2.circle(frame, (cx, cy), core_r, (255, 255, 255), -1)    # 过曝白核
    return frame


def _tuner(cfg):
    return lt.GksLaserCalibTuner(
        get_processor=lambda: _FakeProcessor(cfg), send_laser=lambda on: True)


def test_detect_dot_finds_spot_and_uses_core_centroid(base_cfg):
    dot = _tuner(base_cfg["laser"]).detect_dot(_laser_spot_frame())
    assert dot is not None
    lx, ly, area = dot
    assert abs(lx - 200) <= 1.0 and abs(ly - 150) <= 1.0   # 质心落在白核
    assert 8 <= area <= 4000


def test_detect_dot_rejects_white_highlight(base_cfg):
    """白墙高光(三通道同饱和, R-G≈0)必须被颜色确认层排除。"""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (300, 240), 10, (255, 255, 255), -1)
    assert _tuner(base_cfg["laser"]).detect_dot(frame) is None


def test_detect_dot_rejects_plain_red_scene_blob(base_cfg):
    """大面积普通红块(非饱和)不满足 R>=lum_min, 不算激光点。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:280, 100:180] = (40, 40, 200)   # BGR: 暗红, R=200 不达 lum_min
    assert _tuner(base_cfg["laser"]).detect_dot(frame) is None


def test_detect_dot_area_gate(base_cfg):
    cfg = dict(base_cfg["laser"], min_area=10_000)   # 面积门卡死 -> 永不检出
    assert _tuner(cfg).detect_dot(_laser_spot_frame()) is None


# ---------- 4. 页面 SAVE / 兜底语义 ----------

def test_tuner_save_writes_calib_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "LASER_CALIB_FILE", str(tmp_path / "saved.txt"))
    proc = types.SimpleNamespace(config={
        "target": {"laser_cx": 412},
        "laser": {"lum_min": 210, "dom_min": 40, "min_area": 8,
                  "max_area": 4000},
    })
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: proc, send_laser=lambda on: True)
    tuner._save()
    text = (tmp_path / "saved.txt").read_text(encoding="utf-8")
    assert "laser_cx=412" in text and "lum_min=210" in text
    assert tuner._toast.startswith("SAVED")


def test_tuner_laser_toggle_and_leave_auto_off():
    sent = []
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None,
        send_laser=lambda on: bool(sent.append(on)) or True)
    assert not tuner.laser_is_on()
    tuner._toggle_laser()
    assert sent == [True] and tuner.laser_is_on()
    tuner.on_leave()                       # 切页兜底: 自动发关帧
    assert sent == [True, False] and not tuner.laser_is_on()
    tuner.on_leave()                       # 幂等: 已关不再发
    assert sent == [True, False]


def test_tuner_toggle_tx_fail_keeps_state_off():
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None, send_laser=lambda on: False)
    tuner._toggle_laser()
    assert not tuner.laser_is_on()
    assert tuner._toast.startswith("LASER TX FAIL")
