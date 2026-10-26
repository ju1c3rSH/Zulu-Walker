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


# -*- coding: utf-8 -*-
"""追加 亮环境/白背景签名扩展（2026-10-25）的用例。

覆盖 laser_calib_tuner.py 文件头的"白核+淡红晕"签名链：
  入口: 白核(R,G,B>=prom_min)膨胀 halo_pad 作第二候选入口（halo_pad=0 关闭）;
  确认: bbox+halo_pad 窗口内 强红(dom>=dom_min) 或 白核+淡晕(dom>=dom_faint);
  打分/质心沿用; HUD 标定读数 _last_stats。
以及兼容契约: halo_pad=0（签名关闭）时与旧实现逐像素一致。
台架背景: bbcf8e0 在白背景 50cm 处概率性丢点(晕 dom 25~45 在 dom_min=40 两侧抖动),
概率性失效回归见 test_probabilistic_failure_regression。
"""
import cv2
import numpy as np

from modules.zw_opencv_module.debug import laser_calib_tuner as lt


RED = (0, 0, 255)   # BGR


def _tuner(cfg):
    return lt.GksLaserCalibTuner(
        get_processor=lambda: _FakeProcessor(cfg), send_laser=lambda on: True)


def _legacy_detect(tuner, frame_bgr):
    """旧实现的判据链（签名扩展加入前的 detect_dot），用于兼容对拍。

    注意 min_area/max_area/lum_min/dom_min 必须取自 cfg 而不是写死 ——
    base_cfg fixture 里 min_area 是 _DEFAULT_CONFIG 的 8, 写死 3 会让
    "5px 小点"这一档假失败（新旧其实都被面积门挡掉, 行为一致）。
    """
    if frame_bgr is None:
        return None
    _c = tuner._get_processor().config.get("laser", {})
    lum_min = int(_c.get("lum_min", 200))
    dom_min = int(_c.get("dom_min", 40))
    min_area = int(_c.get("min_area", 8))
    max_area = int(_c.get("max_area", 4000))
    b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
    mask = cv2.inRange(r, lum_min, 255)
    gb = cv2.max(g, b)
    mask = cv2.bitwise_and(mask, cv2.inRange(cv2.subtract(r, gb), dom_min, 255))
    if cv2.countNonZero(mask) == 0:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if best is None or area > best[2]:
            best = (i, centroids[i], area)
    if best is None:
        return None
    idx, centroid, area = best
    x, y, w, h = (int(stats[idx, cv2.CC_STAT_LEFT]),
                  int(stats[idx, cv2.CC_STAT_TOP]),
                  int(stats[idx, cv2.CC_STAT_WIDTH]),
                  int(stats[idx, cv2.CC_STAT_HEIGHT]))
    roi = frame_bgr[y:y + h, x:x + w]
    if roi.size:
        core = (roi[:, :, 2] >= 250) & (roi[:, :, 1] >= 250)
        if int(core.sum()) >= 3:
            ys, xs = np.nonzero(core)
            return (float(x + xs.mean()), float(y + ys.mean()), int(area))
    return (float(centroid[0]), float(centroid[1]), int(area))


# ---------- 兼容契约：halo_pad=0（签名关闭）== 旧实现 ----------

@pytest.mark.parametrize("cx,cy,ring,core,bg", [
    (200, 150, 6, 2, 60),
    (480, 240, 4, 1, 180),      # 亮背景
    (100, 400, 9, 3, 30),
    (320, 240, 1, 0, 90),       # 极小点
])
def test_signature_off_matches_legacy(base_cfg, cx, cy, ring, core, bg):
    frame = _laser_spot_frame(cx=cx, cy=cy, ring_r=ring, core_r=core, bg=bg)
    cfg = dict(base_cfg["laser"], halo_pad=0)   # 关签名 -> 掩码/确认与旧实现一致
    got = _tuner(cfg).detect_dot(frame)
    exp = _legacy_detect(_tuner(cfg), frame)
    if exp is None:
        assert got is None
    else:
        assert got is not None
        assert abs(got[0] - exp[0]) <= 0.51 and abs(got[1] - exp[1]) <= 0.51
        assert got[2] == exp[2]


# ---------- 签名入口: 白核+淡红晕 ----------

def _far_spot_frame(wall=205, dom_fringe=12, glow=False):
    """台架实测形态: 亮白墙 + 3x3 白核(全饱和) + Chebyshev 2~3 一圈淡红晕。

    dom_fringe=0 即"纯白反光/泛白无晕"对照。glow=True 时帧内另放一块
    30x30 纯白高光(有白核无晕), 验证签名确认层的判别力。
    """
    frame = np.full((480, 640, 3), wall, dtype=np.uint8)
    cx, cy = 320, 240
    frame[cy - 1:cy + 2, cx - 1:cx + 2] = (255, 255, 255)          # 白核
    if dom_fringe:
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if max(abs(dy), abs(dx)) in (2, 3):
                    frame[cy + dy, cx + dx] = (210, 210, 210 + dom_fringe)
    if glow:
        frame[100:130, 100:130] = (255, 254, 253)
    return frame


def test_signature_recovers_whitewashed_small_dot(base_cfg):
    """默认参数即检出 白核+淡晕(dom=12) 的远距形态, 质心落在点上。"""
    dot = _tuner(base_cfg["laser"]).detect_dot(_far_spot_frame(dom_fringe=12))
    assert dot is not None
    assert abs(dot[0] - 320) <= 2.0 and abs(dot[1] - 240) <= 2.0


def test_fringe_below_dom_faint_rejected(base_cfg):
    """晕弱于 dom_faint -> 白核也不放行（dom_faint 是台架第一旋钮）。"""
    cfg = dict(base_cfg["laser"], dom_faint=20)
    assert _tuner(cfg).detect_dot(_far_spot_frame(dom_fringe=12)) is None


def test_halo_pad_zero_restores_legacy_rejection(base_cfg):
    """halo_pad=0 关闭签名: 同一帧回到旧"绝对红"单入口, 检不出。"""
    cfg = dict(base_cfg["laser"], halo_pad=0)
    assert _tuner(cfg).detect_dot(_far_spot_frame(dom_fringe=12)) is None


def test_wall_brightness_independence(base_cfg):
    """墙亮度 150/205/230 下同一组默认参数都检出 —— 判别靠白核+晕, 不靠墙色。"""
    for wall in (150, 205, 230):
        dot = _tuner(base_cfg["laser"]).detect_dot(
            _far_spot_frame(wall=wall, dom_fringe=12))
        assert dot is not None, f"wall={wall}"
        assert abs(dot[0] - 320) <= 2.0 and abs(dot[1] - 240) <= 2.0


def test_white_glow_without_fringe_rejected(base_cfg):
    """纯白反光块(有白核无晕)被确认层拒掉; 同帧真点(有晕)照常锁定。"""
    frame = _far_spot_frame(dom_fringe=12, glow=True)
    dot = _tuner(base_cfg["laser"]).detect_dot(frame)
    assert dot is not None
    assert abs(dot[0] - 320) <= 2.0 and abs(dot[1] - 240) <= 2.0   # 不是 (115,115)
    glow_only = _far_spot_frame(dom_fringe=0, glow=True)
    assert _tuner(base_cfg["laser"]).detect_dot(glow_only) is None


def test_color_confirm_rejects_white_blob(base_cfg):
    """纯白高光(三通道同饱和, 无晕)必须被签名确认层排除。"""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (300, 240), 10, (255, 255, 255), -1)
    assert _tuner(base_cfg["laser"]).detect_dot(frame) is None


def test_probabilistic_failure_regression(base_cfg):
    """台架失效回归: bbcf8e0 对 晕 dom≈35±抖动 的 50cm 光斑检出率仅 37/100
    (逐像素 dom_min=40 门限被 AE 抖动±8 + 噪声±5 打穿)。签名方案必须全检出。

    帧间模型: 帧级 AE 抖动 ±8 + 逐像素噪声 ±5, 饱和截断(相机口径, 非 uint8 回绕)。
    """
    rng = np.random.default_rng(42)
    hits = 0
    n = 50
    for _ in range(n):
        frame = np.full((480, 640, 3), 205, dtype=np.uint8)
        ae = int(rng.integers(-8, 9))
        cx, cy = 320, 240
        frame[cy - 1:cy + 2, cx - 1:cx + 2] = (255, 255, 255)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if max(abs(dy), abs(dx)) in (2, 3):
                    d = max(0, 35 + ae + int(rng.integers(-5, 6)))
                    frame[cy + dy, cx + dx] = (210, 210, min(255, 210 + d))
        if _tuner(base_cfg["laser"]).detect_dot(frame) is not None:
            hits += 1
    assert hits == n, f"probabilistic miss: {n - hits}/{n} frames lost"


def test_detect_dot_reports_calibration_stats(base_cfg):
    """HUD 标定读数: 检出后 _last_stats 提供 core/dom 实测值, 未检出清空。"""
    tuner = _tuner(base_cfg["laser"])
    dot = tuner.detect_dot(_far_spot_frame(dom_fringe=14))
    assert dot is not None
    st = tuner._last_stats
    assert st is not None and st["area"] >= 8
    assert st["core"] == 255          # 白核三通道全饱和
    assert st["dom"] >= 14            # 窗口内红度量峰值
    assert tuner.detect_dot(np.full((480, 640, 3), 60, dtype=np.uint8)) is None
    assert tuner._last_stats is None


# ---------- C. 打分选块：小而极亮胜过大而平庸 ----------

def test_score_prefers_higher_peak_at_similar_area(base_cfg):
    """两个面积相当的合格候选: 峰值更高的那个赢（score=sqrt(area)*peak_R）。

    语义边界(2026-10-24 实测): sqrt(area) 增长远快于峰值差异, 所以"小块极亮胜过
    大块平庸"只在两者面积同量级时成立; 面积差一个量级时仍是面积主导。
    挡住"大而平庸"的职责在 B 段颜色确认, 不在打分 —— 见
    test_color_confirm_rejects_white_blob。
    """
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (200, 240), 3, (0, 0, 200), -1)   # 峰值 200
    cv2.circle(frame, (480, 240), 3, (0, 0, 255), -1)   # 峰值 255, 面积相当
    cfg = dict(base_cfg["laser"], min_area=4, max_area=60000)
    dot = _tuner(cfg).detect_dot(frame)
    assert dot is not None
    assert abs(dot[0] - 480) <= 3.0 and abs(dot[1] - 240) <= 3.0


def test_area_gate_still_applies_with_scoring(base_cfg):
    """打分选块不该绕过面积门。"""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 2, (0, 0, 255), -1)
    cfg = dict(base_cfg["laser"], min_area=10_000)
    assert _tuner(cfg).detect_dot(frame) is None


# ---------- D. 亮度加权质心 ----------

def test_weighted_centroid_tracks_brightest_pixels(base_cfg):
    """偏心光斑（一侧更亮）：加权质心应偏向亮侧，而非几何中心。"""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (300, 240), 5, (0, 0, 200), -1)     # 左半: 较暗
    cv2.circle(frame, (308, 240), 5, (0, 0, 255), -1)     # 右半: 更亮
    dot = _tuner(base_cfg["laser"]).detect_dot(frame)
    assert dot is not None
    assert dot[0] > 303.0        # 几何中心 304, 加权后被拉向亮侧


def test_core_centroid_returns_tuple_and_falls_back(base_cfg):
    """契约: _core_centroid 返回 tuple; 权重全 0 时 None（调用方退回几何质心）。"""
    tuner = _tuner(base_cfg["laser"])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 4, (0, 0, 255), -1)
    mask = np.zeros((480, 640), np.uint8)
    cv2.circle(mask, (320, 240), 4, 255, -1)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    got = lt.GksLaserCalibTuner._core_centroid(frame, stats, 1)
    assert isinstance(got, tuple) and len(got) == 2
    # 全零帧 -> None
    zero = np.zeros((480, 640, 3), np.uint8)
    assert lt.GksLaserCalibTuner._core_centroid(zero, stats, 1) is None


# ---------- 参数面：SAVE / 热重载 / set_runtime_param ----------

def test_signature_keys_saved_and_reloadable(base_cfg, tmp_path, monkeypatch,
                                             log_capture):
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("dom_faint=10\nhalo_pad=5\nprom_min=235\n", encoding="utf-8")
    base_cfg["laser"]["calib_file"] = str(f)
    p = GksTaskProcessor(config=base_cfg)
    assert p.config["laser"]["dom_faint"] == 10
    assert p.config["laser"]["halo_pad"] == 5
    assert p.config["laser"]["prom_min"] == 235
    assert any("laser calib" in m for m in log_capture)


def test_retired_keys_ignored(base_cfg, tmp_path):
    """已退役键(bg_kernel 等)在设备旧校准文件里残留时必须被安全忽略。"""
    f = tmp_path / "my_laser_calib.txt"
    f.write_text("bg_kernel=41\ndiff_th=25\ndom_halo=20\ndom_local=12\n"
                 "dom_faint=10\n", encoding="utf-8")
    base_cfg["laser"]["calib_file"] = str(f)
    p = GksTaskProcessor(config=base_cfg)
    assert p.config["laser"]["dom_faint"] == 10
    assert "bg_kernel" not in p.config["laser"]


def test_signature_keys_accepted_by_runtime_param_gate(base_cfg):
    """签名键必须可 set_runtime_param; 未定义键仍被拒（防拼错键静默失效）。"""
    p = GksTaskProcessor(config=base_cfg)
    p.set_runtime_param("laser", "dom_faint", 8)
    p.set_runtime_param("laser", "halo_pad", 0)
    assert p.config["laser"]["dom_faint"] == 8
    assert p.config["laser"]["halo_pad"] == 0
    with pytest.raises(ValueError):
        p.set_runtime_param("laser", "dom_fiant", 8)       # 拼错
    with pytest.raises(ValueError):
        p.set_runtime_param("laser", "unknown_new_key", 1)


def test_save_writes_signature_keys(base_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "LASER_CALIB_FILE", str(tmp_path / "saved.txt"))
    proc = types.SimpleNamespace(config={
        "target": {"laser_cx": 412},
        "laser": {"lum_min": 210, "dom_min": 40, "min_area": 8,
                  "max_area": 4000, "dom_faint": 12, "halo_pad": 3,
                  "prom_min": 235},
    })
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: proc, send_laser=lambda on: True)
    tuner._save()
    text = (tmp_path / "saved.txt").read_text(encoding="utf-8")
    for key in ("lum_min=210", "dom_faint=12", "halo_pad=3", "prom_min=235"):
        assert key in text


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


# ---------- 5. 手动校零参考线（Line 按钮, 竖线 @ 画面中心 x） ----------

def test_guide_line_toggle_and_begin_reset():
    """点击翻转, 进页(begin)复位; 与激光开关相互独立。"""
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None, send_laser=lambda on: True)
    assert tuner._guide_on is False
    tuner._toggle_guide()
    assert tuner._guide_on is True
    tuner._toggle_guide()
    assert tuner._guide_on is False
    tuner._toggle_guide()                  # 开着离开, 下次进页应复位
    tuner.begin()
    assert tuner._guide_on is False
    assert tuner.laser_is_on() is False    # 参考线不碰激光状态


def test_guide_toggle_has_no_hardware_side_effect():
    """参考线是纯显示状态: 切换不发任何 0x25 帧。"""
    sent = []
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None,
        send_laser=lambda on: sent.append(on) or True)
    tuner._toggle_guide()
    assert sent == [] and tuner._guide_on is True
    tuner.on_leave()                       # 兜底关激光只由 laser_is_on 决定
    assert sent == []


class _FakeImg:
    """记录 draw 调用的假画布: 锁定按钮/参考线的渲染路径(PC 无 maix.image)。

    存在意义: 曾经只接了点击热区没接渲染, 按钮在设备上隐形 —— 状态级测试
    全绿也发现不了, 必须有渲染路径断言。
    """

    def __init__(self, w=640, h=480):
        self._w, self._h = w, h
        self.rects, self.strings, self.lines = [], [], []

    def width(self):
        return self._w

    def height(self):
        return self._h

    def draw_rect(self, x, y, w, h, color=None, thickness=2):
        self.rects.append((x, y, w, h, thickness))

    def draw_string(self, x, y, s, color=None, scale=1.0, thickness=1):
        self.strings.append((x, y, s))

    def draw_line(self, x1, y1, x2, y2, color=None, thickness=1):
        self.lines.append((x1, y1, x2, y2))


def test_guide_button_rendered_with_label():
    """回归: Line 按钮必须有可见渲染 —— 点击热区不能是隐形的。"""
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None, send_laser=lambda on: True)
    img = _FakeImg()
    tuner._draw_guide_btn(img)
    assert lt._UI_GUIDE_BTN[:4] + (-1,) in img.rects       # 填充矩形
    assert any(s == "Line:OFF" for _, _, s in img.strings)
    tuner._toggle_guide()
    img2 = _FakeImg()
    tuner._draw_guide_btn(img2)
    assert any(s == "Line:ON" for _, _, s in img2.strings)


def test_guide_line_rendered_at_center_when_on():
    """开关控制竖线: 关不画; 开画在画面中心 x, 从顶栏下沿到底。"""
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None, send_laser=lambda on: True)
    img = _FakeImg(w=640, h=480)
    tuner._draw_guide(img)
    assert img.lines == []
    tuner._toggle_guide()
    tuner._draw_guide(img)
    assert img.lines == [(320, 30, 320, 480)]


def test_guide_click_route_toggles():
    """按钮矩形内的点击必须路由到 _toggle_guide。"""
    tuner = lt.GksLaserCalibTuner(
        get_processor=lambda: None, send_laser=lambda on: True)
    img = _FakeImg()
    bx, by, bw, bh = lt._UI_GUIDE_BTN[:4]
    tuner._handle_click(img, bx + 10, by + 10)
    assert tuner._guide_on is True
    tuner._handle_click(img, bx + 10, by + 10)
    assert tuner._guide_on is False
