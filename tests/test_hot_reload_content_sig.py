# -*- coding: utf-8 -*-
"""热重载变化判定: 必须用内容签名, 不能用 mtime。

背景 (T1/P0): 三处热重载(阈值/人质参数/激光校准)原用 `mtime == 上次` 判变化。
同一文件系统 tick 内二次写入 mtime 不变(实测连续写 300 次 57% 相同),
调参页连按两次 SAVE 会**静默漏掉第二次**, 现场表现为"SAVE 了但参数没变"。
`(st_mtime_ns, st_size)` 同样有 56% 漏检(等长改值), 只有内容哈希可靠。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor, file_signature, load_gks_config,
)


@pytest.fixture()
def base_cfg():
    c = load_gks_config(config_path="/nonexistent/eod_config.yaml")
    c["thresholds_file"] = ""
    c["laser"]["calib_file"] = ""
    c["rescue_params_file"] = ""
    return c


def _proc(cfg):
    return GksTaskProcessor(config=cfg)


def test_file_signature_detects_same_size_change(tmp_path):
    """等长改值(380->412)必须被检出 —— 这正是 mtime_ns+size 漏掉的场景。"""
    f = tmp_path / "x.txt"
    f.write_text("laser_cx=380\n", encoding="utf-8")
    a = file_signature(str(f))
    f.write_text("laser_cx=412\n", encoding="utf-8")   # 同长度, 同 tick
    b = file_signature(str(f))
    assert a is not None and b is not None
    assert a != b, "同长度改值未被检出"


def test_file_signature_missing_returns_none(tmp_path):
    assert file_signature(str(tmp_path / "nope.txt")) is None


def test_laser_calib_reloads_on_same_tick_save(base_cfg, tmp_path):
    """连按两次 SAVE(同 tick) 第二次也必须生效。"""
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("laser_cx=380\n", encoding="utf-8")
    base_cfg["laser"]["calib_file"] = str(f)
    p = _proc(base_cfg)
    assert p.target_zero_px() == 380

    f.write_text("laser_cx=412\n", encoding="utf-8")   # 同长度, 紧接写入
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert p.target_zero_px() == 412, "同 tick SAVE 未生效(T1 回归)"


def test_rescue_params_reloads_on_same_tick_save(base_cfg, tmp_path):
    f = tmp_path / "my_rescue_params.txt"
    f.write_text("conf_th=0.61\niou_th=0.45\n", encoding="utf-8")
    base_cfg["rescue_params_file"] = str(f)
    p = _proc(base_cfg)
    assert abs(p.config["rescue"]["conf_th"] - 0.61) < 1e-9

    f.write_text("conf_th=0.55\niou_th=0.45\n", encoding="utf-8")  # 同长度
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert abs(p.config["rescue"]["conf_th"] - 0.55) < 1e-9, "同 tick SAVE 未生效"


def test_unchanged_file_does_not_reload(base_cfg, tmp_path, monkeypatch):
    """内容没变就不该重复重载(log 不重复打)。"""
    import modules.zw_opencv_module.processors.gks_task_processor as gks
    msgs = []
    monkeypatch.setattr(gks, "log_print", lambda m="": msgs.append(m))
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("laser_cx=380\n", encoding="utf-8")
    base_cfg["laser"]["calib_file"] = str(f)
    p = _proc(base_cfg)
    n = len(msgs)
    p._thresholds_poll_ts = p._now() - 10.0
    p._maybe_reload_thresholds()
    assert len(msgs) == n, "文件未变却重载了"
