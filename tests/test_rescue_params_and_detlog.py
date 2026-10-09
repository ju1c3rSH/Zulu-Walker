# -*- coding: utf-8 -*-
"""人质参数文件（rescue_params_file）+ 检出日志 tests。

覆盖 2026-10-10 新增的三块:
  1. load_rescue_params_file 解析器（key=value / 注释 / 非法行容错）
  2. GksTaskProcessor: 启动加载 + mtime 热重载（语义同 thresholds_file, 不清锁存）
  3. _log_rescue_dets "变化才打"去重（集合变化 / 分数跳变 >0.1 / 稳定静默）
  4. GksRescueTuner._save_params 落盘（RESCUE_PARAMS_FILE 注入 tmp 路径）
"""
import types

import pytest

import modules.zw_opencv_module.processors.gks_task_processor as gp
from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor,
    load_gks_config,
    load_rescue_params_file,
)
from modules.zw_opencv_module.debug import rescue_tuner as rt


@pytest.fixture()
def base_cfg():
    c = load_gks_config(config_path="/nonexistent/eod_config.yaml")
    c["thresholds_file"] = ""     # 隔离设备端 /my-threshold 文件
    return c


@pytest.fixture()
def log_capture(monkeypatch):
    """捕获 gks_task_processor 模块内的 log_print 调用。"""
    msgs = []
    monkeypatch.setattr(gp, "log_print", lambda m="": msgs.append(m))
    return msgs


# ---------- 1. 解析器 ----------

def test_parser_reads_valid_lines(tmp_path):
    f = tmp_path / "my_rescue_params.txt"
    f.write_text("# comment\nconf_th=0.55\niou_th = 0.45\nx_crop_px=240\n",
                 encoding="utf-8")
    assert load_rescue_params_file(str(f)) == {
        "conf_th": 0.55, "iou_th": 0.45, "x_crop_px": 240}


def test_parser_ignores_bad_lines_and_unknown_keys(tmp_path):
    f = tmp_path / "my_rescue_params.txt"
    f.write_text("conf_th=abc\nunknown_key=1\n\n# only comment\nscan_scale=0.5\n",
                 encoding="utf-8")
    assert load_rescue_params_file(str(f)) == {}


def test_parser_missing_file_is_noop(tmp_path):
    assert load_rescue_params_file(str(tmp_path / "nope.txt")) == {}


# ---------- 2. 加载 + 热重载 ----------

def _write_params(path, conf):
    path.write_text(f"conf_th={conf}\n", encoding="utf-8")


def test_init_loads_override_file(base_cfg, tmp_path):
    f = tmp_path / "my_rescue_params.txt"
    _write_params(f, 0.61)
    base_cfg["rescue_params_file"] = str(f)
    p = GksTaskProcessor(config=base_cfg)
    assert p.config["rescue"]["conf_th"] == 0.61
    # 未写的键保持 yaml/默认值
    assert p.config["rescue"]["iou_th"] == 0.45


def test_hot_reload_on_mtime_change(base_cfg, tmp_path, log_capture):
    f = tmp_path / "my_rescue_params.txt"
    _write_params(f, 0.61)
    base_cfg["rescue_params_file"] = str(f)
    p = GksTaskProcessor(config=base_cfg)
    n_loaded = len(log_capture)

    # 越过 2Hz 闸门后触发轮询: 文件未变 -> 不重载
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert p.config["rescue"]["conf_th"] == 0.61
    assert len(log_capture) == n_loaded

    # 改文件 -> 下一轮 2Hz 轮询重载, 只 merge 写过的键, 不清锁存
    _write_params(f, 0.55)
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert p.config["rescue"]["conf_th"] == 0.55
    assert any("reloaded" in m for m in log_capture[n_loaded:])


def test_rescue_params_file_is_immutable_runtime_key(base_cfg, tmp_path):
    base_cfg["rescue_params_file"] = str(tmp_path / "x.txt")
    p = GksTaskProcessor(config=base_cfg)
    with pytest.raises(ValueError):
        p.set_runtime_param("rescue_params_file", "rescue_params_file", "/other")


# ---------- 3. 检出日志去重 ----------

def _det(label, score, cx=100, cy=80):
    return (label, score, cx, cy)


def _make_processor(base_cfg):
    base_cfg.setdefault("rescue_params_file", "")
    return GksTaskProcessor(config=base_cfg)


def test_det_log_first_frame_then_silent_while_stable(base_cfg, log_capture):
    p = _make_processor(base_cfg)
    p._log_rescue_dets([_det("yuan_zhu", 0.82), _det("yao_gu", 0.76)])
    assert len(log_capture) == 1
    assert "yuan_zhu=0.82(100,80)" in log_capture[0]

    # 同集合 + 分数小幅波动 -> 静默
    p._log_rescue_dets([_det("yuan_zhu", 0.85), _det("yao_gu", 0.72)])
    assert len(log_capture) == 1

    # 分数跳变 >0.1 -> 打一行
    p._log_rescue_dets([_det("yuan_zhu", 0.96), _det("yao_gu", 0.72)])
    assert len(log_capture) == 2


def test_det_log_set_change_and_disappear(base_cfg, log_capture):
    p = _make_processor(base_cfg)
    p._log_rescue_dets([_det("yuan_zhu", 0.82)])
    p._log_rescue_dets([_det("yuan_zhu", 0.82), _det("yao_gu", 0.76)])   # 新出现
    p._log_rescue_dets([_det("yuan_zhu", 0.83)])                          # 消失
    assert len(log_capture) == 3
    assert "yao_gu=0.76" in log_capture[1]
    assert "(none)" in log_capture[2] or "yuan_zhu" in log_capture[2]


def test_det_log_same_label_keeps_best_score(base_cfg, log_capture):
    p = _make_processor(base_cfg)
    p._log_rescue_dets([_det("yuan_zhu", 0.60, cx=10), _det("yuan_zhu", 0.90, cx=20)])
    p._log_rescue_dets([_det("yuan_zhu", 0.62, cx=10), _det("yuan_zhu", 0.88, cx=20)])
    assert len(log_capture) == 1
    assert "0.90(20,80)" in log_capture[0]


# ---------- 4. 调参页 SAVE ----------

class _FakeProcessor:
    def __init__(self, rescue_cfg):
        self.config = {"rescue": dict(rescue_cfg)}


def test_tuner_save_writes_file(base_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "RESCUE_PARAMS_FILE", str(tmp_path / "saved_params.txt"))
    proc = _FakeProcessor({"conf_th": 0.55, "iou_th": 0.45, "x_crop_px": 0})
    tuner = rt.GksRescueTuner(get_processor=lambda: proc, get_ai=lambda: None)
    tuner._save_params()
    text = (tmp_path / "saved_params.txt").read_text(encoding="utf-8")
    assert "conf_th=0.55" in text
    assert "iou_th=0.45" in text
    assert "x_crop_px=0" in text
    assert tuner._toast.startswith("SAVED")


def test_tuner_save_failure_does_not_raise(base_cfg, tmp_path, monkeypatch):
    # 目录路径是个已存在的**文件** -> makedirs/open 必失败
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(rt, "RESCUE_PARAMS_FILE", str(blocker / "sub" / "p.txt"))
    proc = _FakeProcessor({"conf_th": 0.55})
    tuner = rt.GksRescueTuner(get_processor=lambda: proc, get_ai=lambda: None)
    tuner._save_params()   # 不应抛异常
    assert tuner._toast.startswith("SAVE FAILED")
