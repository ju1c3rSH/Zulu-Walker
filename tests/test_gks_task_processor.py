# -*- coding: utf-8 -*-
"""GksTaskProcessor tests — 照搬 main.py task1/2/3 的行为等价断言。

锁存/去抖/镜像/定点化语义以 工科赛上端 main.py 为参照（审查报告 §4.4 第二轮）,
PC 跑 cv2 路径（与设备 maix_accel 快路径共享同一套选块/锁存代码）。
"""
import cv2
import numpy as np
import pytest

from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor,
    MODE_EXPLOSIVE, MODE_TARGET, MODE_RESCUE,
    load_gks_config, load_threshold_file,
)


# ---------- fixtures ----------

def lab_threshold_for(bgr, half=25):
    """由合成色块反推 OpenMV 风格 LAB 阈值（覆盖该色、排除邻近色）。"""
    px = np.uint8([[bgr]])
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
    l_om = lab[0] / 2.55
    a = int(lab[1]) - 128
    b = int(lab[2]) - 128
    return [max(0, int(l_om - half)), min(100, int(l_om + half)),
            max(-128, a - half), min(127, a + half),
            max(-128, b - half), min(127, b + half)]


RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)   # BGR


@pytest.fixture()
def pit_frame():
    """640x480: 左红 中绿 右蓝, 三个 120x120 色块, 中心 y=240。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for bgr, cx in ((RED, 107), (GREEN, 320), (BLUE, 533)):
        x0, x1 = cx - 60, cx + 60
        frame[180:300, x0:x1] = bgr
    return frame


@pytest.fixture()
def cfg():
    c = load_gks_config(config_path="/nonexistent/eod_config.yaml")
    c["thresholds_file"] = ""   # 隔离设备端 /my-threshold 文件
    c["explosive"]["lab"] = {"red": lab_threshold_for(RED),
                             "green": lab_threshold_for(GREEN),
                             "blue": lab_threshold_for(BLUE)}
    c["target"]["lab"] = dict(c["explosive"]["lab"])
    return c


# ---------- task1 爆炸物 ----------

def make_pit_frame(colors):
    """colors: [(bgr, cx), ...] -> 640x480 帧, 120x120 色块, 中心 y=240。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for bgr, cx in colors:
        frame[180:300, cx - 60:cx + 60] = bgr
    return frame


def test_explosive_multi_frame_first_seen_is_pit_order(cfg):
    """跨帧扫过三坑: 首次检出的颜色就是 pit1, 未见者按 2025 的 6-a-b 补到 pit3。

    第 1 帧只看得见画面最右的蓝色 -> 蓝就是 pit1（时间序, 不是画面 x 序）。
    """
    cfg["explosive"]["infer_missing"] = "tail"      # 复刻 2025 方案"两色即锁"
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))        # 目标 = 蓝色
    proc.process(make_pit_frame([(BLUE, 533)]))                  # 第1帧: 只有蓝
    proc.process(make_pit_frame([(BLUE, 533), (GREEN, 320)]))    # 第2帧: 蓝+绿
    r = proc.process(make_pit_frame([(BLUE, 533), (GREEN, 320)]))
    assert r.result_data["latched_order"] == [3, 2, 1]   # 未见红 -> 补 pit3
    assert r.result_data["pit"] == 1
    assert r.result_data["inferred"] is True


def test_explosive_same_frame_order_is_spatial_not_color_index(cfg):
    """同帧三色齐见: 帧内按画面左右定序, 不是按遍历色号(红→绿→蓝)定序。"""
    frame = make_pit_frame([(BLUE, 107), (GREEN, 320), (RED, 533)])
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))
    for _ in range(2):
        r = proc.process(frame)
    assert r.result_data["latched_order"] == [3, 2, 1]   # 画面左起: 蓝绿红
    assert r.result_data["pit"] == 1
    assert r.result_data["inferred"] is False


def test_explosive_default_infers_third_color(cfg):
    """默认 infer_missing=tail(2026-09-24 台架决策, 与 2025 方案一致): 两色即锁。

    ⚠ 补位只推出"是什么"、推不出"在第几坑" —— 日志出现 inferred=True 时必须用实物
    核对 PIT/POS(见 gks_bench_checklist 台架必做项); 要退回保守行为置 none。
    """
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))        # 目标 = 蓝色
    frame = make_pit_frame([(BLUE, 533), (GREEN, 320)])
    for _ in range(3):
        r = proc.process(frame)
    assert r.result_data["found"] and r.result_data["inferred"] is True
    assert r.result_data["latched_order"] == [2, 3, 1]   # 画面左起 绿、蓝 -> 红补尾部
    assert r.result_data["pit"] == 2                     # 蓝排第二坑


def test_explosive_infer_missing_none_never_latches_two_colors(cfg):
    """显式 infer_missing=none: 退回保守行为 —— 只有两色永不锁存(锁不上会超时可见)。"""
    cfg["explosive"]["infer_missing"] = "none"
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))
    frame = make_pit_frame([(BLUE, 533), (GREEN, 320)])
    for _ in range(5):
        r = proc.process(frame)
    assert not r.result_data["found"] and r.result_data["pit"] == 0
    assert r.result_data["inferred"] is False


def test_explosive_scan_scale_half(cfg, pit_frame):
    """scan_scale=0.5: 320x240 扫描 —— 面积门槛按 scale² 折算, cx/框还原到全帧坐标。"""
    cfg["explosive"]["scan_scale"] = 0.5
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))   # 订阅绿色 -> 中坑
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["found"], "降采样后仍应检出(门槛已按 scale² 折算)"
    assert r.result_data["pit"] == 2
    assert 300 < r.result_data["cx"] < 340     # 全帧坐标 ≈320, 不是小图的 160
    blob = [h for h in r.result_data["hits"] if h["text"] == "C2"][0]
    assert blob["w"] > 100 and blob["x"] > 250  # HUD 框同样还原到全帧


class _FakeMaixImage:
    """记录 cv2image 入参的假 maix.image 模块（PC 端验证通道序）。"""

    def __init__(self):
        self.calls = []

    def cv2image(self, array, bgr=True, copy=True):   # noqa: A002 - 对齐固件签名
        self.calls.append((np.array(array, copy=True), bool(bgr), bool(copy)))
        return object()


def _install_fake_maix(monkeypatch):
    import sys
    import types
    import modules.zw_opencv_module.processors.gks_task_processor as gtp
    # gtp 缓存了"maix.image 不可用"的探测结论（PC 端首个 _downscale 调用就会
    # 缓存 False, 免得每帧重抛 ImportError）。本用例要注入假 maix, 必须先把
    # 缓存复位成"未探测"——真机上不存在这个场景, 故提供显式复位而非改实现。
    monkeypatch.setattr(gtp, "_maix_image_ok", None, raising=False)
    fake = _FakeMaixImage()
    maix_mod = types.ModuleType("maix")
    image_mod = types.ModuleType("maix.image")
    image_mod.cv2image = fake.cv2image
    maix_mod.image = image_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)
    return fake


def test_downscale_hands_maix_rgb_not_bgr(cfg, monkeypatch):
    """降采样小图必须交 RGB 字节序（2026-09-26 台架事故回归）。

    机理: MaixCDK convert_to_imlib_image() 对 FMT_BGR888 是 fall-through 到
    PIXFORMAT_RGB888、不做通道交换 —— 交 BGR 就会被当 RGB 算 LAB，红蓝互换：
    蓝色目标命中"红色"阈值盒，HUD 把蓝色框标成 C1、坑位定序连带全错。
    """
    cfg["explosive"]["scan_scale"] = 0.5
    proc = GksTaskProcessor(config=cfg)
    fake = _install_fake_maix(monkeypatch)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = BLUE                                   # 整帧纯蓝: BGR=(255,0,0)

    small, small_maix = proc._downscale(frame, None, 0.5)

    assert small_maix is not None, "假 maix 已就绪, 小图应构造成功"
    arr, bgr_flag, _copy = fake.calls[-1]
    assert bgr_flag is False, "必须 bgr=False（FMT_BGR888 会被 imlib 当 RGB888 用）"
    assert tuple(arr[0, 0]) == (0, 0, 255), "小图应是 RGB 字节序（蓝 = 第三通道）"
    assert small.shape == (240, 320, 3)
    assert tuple(small[0, 0]) == (255, 0, 0), "cv2 回退路径的小图仍必须是 BGR"


def test_downscale_maix_failure_falls_back_to_cv2(cfg, monkeypatch):
    """maix 构造失败时返回 None, 调用方走 cv2（BGR 小图 + COLOR_BGR2LAB）。"""
    import sys
    import types
    import modules.zw_opencv_module.processors.gks_task_processor as gtp
    monkeypatch.setattr(gtp, "_maix_image_ok", None, raising=False)
    cfg["explosive"]["scan_scale"] = 0.5
    proc = GksTaskProcessor(config=cfg)
    image_mod = types.ModuleType("maix.image")

    def _boom(*_a, **_kw):
        raise RuntimeError("no maix")

    image_mod.cv2image = _boom
    maix_mod = types.ModuleType("maix")
    maix_mod.image = image_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = BLUE
    small, small_maix = proc._downscale(frame, None, 0.5)
    assert small_maix is None
    assert tuple(small[0, 0]) == (255, 0, 0)



def _blocks_frame(colors, size):
    """色块边长 size（中心 y=240）的 640x480 帧。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    half = size // 2
    for bgr, cx in colors:
        frame[240 - half:240 + half, cx - half:cx + half] = bgr
    return frame


def test_explosive_area_threshold_4000(cfg):
    """面积门槛 10000 -> 4000: 70x70(=4900px) 的色块现在算检出（旧门槛不够）。

    背景: 10000 是 2025 main.py 原值（≈100x100），对应他们那套机位/距离；我们机位
    目标更小 + 运动模糊先啃边缘 —— 调参页看得见（只要 area>=200）主流程却 found=0。
    """
    import copy

    frame = _blocks_frame([(RED, 107), (GREEN, 320), (BLUE, 533)], 70)
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r = proc.process(frame)
    assert r.result_data["found"], "4900px 的色块应够 4000 门槛"
    assert r.result_data["pit"] == 2

    old = copy.deepcopy(cfg)
    old["explosive"]["pixels_threshold"] = 10000
    old["explosive"]["area_threshold"] = 10000
    proc_old = GksTaskProcessor(config=old)
    proc_old.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r_old = proc_old.process(frame)
    assert not r_old.result_data["found"], "同样的色块在旧门槛 10000 下不够"


def test_explosive_scan_scale_full_is_unchanged(cfg, pit_frame):
    """scan_scale=1.0: 与降采样前行为完全一致(默认口径)。"""
    cfg["explosive"]["scan_scale"] = 1.0
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["found"] and r.result_data["pit"] == 2
    assert 300 < r.result_data["cx"] < 340


def test_explosive_two_frame_latch(cfg, pit_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))   # 订阅绿色 -> 中坑
    r1 = proc.process(pit_frame)
    assert not r1.result_data["found"]          # 单帧不锁存
    assert r1.result_data["pending_count"] == 1
    r2 = proc.process(pit_frame)
    assert r2.result_data["found"]
    assert r2.result_data["pit"] == 2           # 绿色在中间
    assert r2.result_data["cx"] > 0
    assert r2.result_data["latched_order"] == [1, 2, 3]   # 红/绿/蓝


def test_explosive_rightmost_pit3(cfg, pit_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))    # 蓝色在最右
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["pit"] == 3


def test_explosive_mirror_switch(cfg, pit_frame):
    cfg["pit_left_is_1"] = False                # 摄像头背装镜像: 右侧=坑1
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (3, 0, 0))
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["pit"] == 1


def test_explosive_incomplete_colors_never_latch(cfg, pit_frame):
    """infer_missing=none 时: 抹掉一色后永不锁存(保守退路, 锁不上会超时可见)。"""
    cfg["explosive"]["infer_missing"] = "none"
    frame = pit_frame.copy()
    frame[:, 400:] = 0                          # 抹掉蓝色: 只有红绿
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 0, 0))
    for _ in range(5):
        r = proc.process(frame)
    assert not r.result_data["found"]
    assert r.result_data["pit"] == 0


def test_explosive_missing_threshold_zero_heartbeat(cfg):
    cfg["explosive"]["lab"] = {"red": [], "green": [], "blue": []}
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (1, 0, 0))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and r.result_data["cx"] == 0
    assert not r.result_data["thresholds_ok"]


def test_explosive_same_codes_resubscribe_keeps_latch(cfg, pit_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        proc.process(pit_frame)
    assert proc._pos_flag == 1
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))    # Brain 1s 看门狗重订阅(幂等)
    assert proc._pos_flag == 1                  # 锁存保持
    assert proc._position == [1, 2, 3]


def test_explosive_code_change_resets_latch(cfg, pit_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        proc.process(pit_frame)
    assert proc._pos_flag == 1
    proc.set_task(MODE_EXPLOSIVE, (1, 1, 3))    # 换任务码
    assert proc._pos_flag == 0 and proc._position == [0, 0, 0]
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["pit"] == 1            # 红色在左


def test_explosive_reset_clears_latch(cfg, pit_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        proc.process(pit_frame)
    proc.reset()                                # CMD_STOP
    r = proc.process(pit_frame)
    assert not r.result_data["found"]           # 重新计数, 第一帧不锁存


# ---------- task1 排爆 ROI（纯提速视窗）----------

def test_explosive_roi_default_none_is_full_frame(cfg, pit_frame):
    """默认 roi=None: 行为与加 ROI 之前逐字节一致（全帧）。"""
    assert cfg["explosive"].get("roi") is None
    proc = GksTaskProcessor(config=cfg)
    assert proc.explosive_roi() is None
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["found"] and r.result_data["pit"] == 2


def test_explosive_roi_restricts_scan_but_keeps_order(cfg, pit_frame):
    """ROI 只缩小搜索域: 三坑都在 ROI 内时, 定序结果与全帧完全一致。"""
    cfg["explosive"]["roi"] = [0.0, 0.3, 1.0, 1.0]   # 覆盖 y 144..480, 三坑 180..300
    proc = GksTaskProcessor(config=cfg)
    assert proc.explosive_roi() == (0.0, 0.3, 1.0, 1.0)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["found"] and r.result_data["pit"] == 2
    assert r.result_data["latched_order"] == [1, 2, 3]
    assert 300 < r.result_data["cx"] < 340


def test_explosive_roi_excludes_outside(cfg, pit_frame):
    """三坑完全落在 ROI 之外(上方)时: 三色全漏检 -> 不锁存(pit=0)。

    这条钉住"ROI 会真实缩小视野"—— 定序前提(三坑都在视野内)不成立时表现为
    **锁不上**, 而不是错序; 静默错序只在"部分色块被 ROI 切半"时才可能发生。
    """
    cfg["explosive"]["roi"] = [0.0, 0.7, 1.0, 1.0]   # y 336..480, 三坑 180..300 全在上方
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(3):
        r = proc.process(pit_frame)
    assert not r.result_data["found"] and r.result_data["pit"] == 0


def test_explosive_roi_clipped_block_still_latches(cfg, pit_frame):
    """色块被 ROI 切掉一半但剩余像素仍过门槛时照常锁存 —— 证明 ROI 只裁搜索域, 不改判定。

    pit_frame 色块 120x120=14400px; ROI y0=0.5 -> 剩 60x120=7200px >= 4000 门槛。
    """
    cfg["explosive"]["roi"] = [0.0, 0.5, 1.0, 1.0]
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        r = proc.process(pit_frame)
    assert r.result_data["found"] and r.result_data["pit"] == 2
    assert r.result_data["latched_order"] == [1, 2, 3]


def test_explosive_roi_clipped_below_threshold_not_latched(cfg, pit_frame):
    """⚠ 危险区: 色块被 ROI 切剩的像素 < 门槛时该色被判"未检出" -> 补位把它塞到尾部。

    这就是"ROI 切断定序视野"的具体故障形态: 目标色(pit1)漏检 -> infer_missing=tail
    补到 pit3 -> 静默锁错坑。台架含义: ROI 必须完整包住三坑, 不能贴着坑边切。
    """
    cfg["explosive"]["roi"] = [0.0, 0.58, 1.0, 1.0]   # y 278..480, 各色只剩 22x120=2640px
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(3):
        r = proc.process(pit_frame)
    assert not r.result_data["found"] and r.result_data["pit"] == 0


def test_explosive_roi_area_threshold_not_scaled_by_roi(cfg):
    """ROI 不折算面积门槛: 60x60=3600px 完全落在 ROI 内仍过不了 4000 门槛。"""
    cfg["explosive"]["roi"] = [0.0, 0.4, 1.0, 1.0]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[260:320, 290:350] = RED           # 60x60=3600px, 完全在 ROI 内
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (1, 0, 0))
    for _ in range(3):
        r = proc.process(frame)
    assert not r.result_data["found"]


# ---------- task2 靶标 ----------
@pytest.fixture()
def target_frame():
    """红色靶标块, 中心 cx=480 (e=+0.5), y 在默认 ROI 内。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[220:260, 420:540] = RED
    return frame


def test_target_error_fixed_point(cfg, target_frame):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))       # 靶标红色
    r = proc.process(target_frame)
    # main.py: int(((480-320)/320) * 10000)
    assert r.result_data["found"]
    assert r.result_data["err_fixed"] == 5000


def test_target_no_blob_zero(cfg):
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and r.result_data["err_fixed"] == 0


def test_target_roi_excludes_outside(cfg):
    cfg["target"]["roi"] = [0.0, 0.352, 1.0, 0.763]   # y 169..366
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[40:80, 420:540] = RED                       # ROI 之外(上方)
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    r = proc.process(frame)
    assert not r.result_data["found"]


def test_target_missing_threshold_zero_heartbeat(cfg):
    cfg["target"]["lab"] = {"red": [], "green": [], "blue": []}
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 2, 0))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and not r.result_data["thresholds_ok"]


# ---------- task2 防误检（2026-10-08）----------
#
# 选候选原来只有 `max(blobs, key=pixels)` 一个判据, 而 err_fixed 是云台闭环的
# 控制量 —— 任何单帧误检都以满幅值灌进 PID。这组用例钉住连续性选择 + 死区 +
# 连续确认, 以及"默认关时行为与改动前逐字节一致"这条不变量。
#
# 注: 形状先验(roundness/density/solidity)已删除 —— 靶标是**圆环**, 实心球的
# 阈值会把真环全部误杀(k=0.5 时圆环 roundness=0.333 < 方块 0.785)。见
# eod_config.yaml target 段。


def test_target_continuity_picks_nearest_not_largest(cfg):
    """两个候选像素接近: 开continuity 后取"离上帧最近"而非"pixels 最大"。"""
    cfg["target"]["prefer_continuity"] = True
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    proc._target_last_cx = 520.0                     # 上一帧在右边
    picked = proc._pick_target_blob(
        [{"cx": 200.0, "pixels": 10000},            # 左, 像素多
         {"cx": 520.0, "pixels": 9000}], cfg["target"])
    assert picked["cx"] == 520.0                    # 近者胜, 尽管像素少


def test_target_continuity_single_candidate_unchanged(cfg):
    """单候选时 continuity 不改变结果（它与自己的距离=0, 自然选中）。"""
    cfg["target"]["prefer_continuity"] = True
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    proc._target_last_cx = 100.0
    picked = proc._pick_target_blob([{"cx": 300.0, "pixels": 500}], cfg["target"])
    assert picked["cx"] == 300.0


def test_target_baseline_largest_pixels_when_continuity_off(cfg):
    """默认(continuity 关)仍是 pixels 最大 —— 但 tiebreak 确定, 不依赖输出序。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    picked = proc._pick_target_blob(
        [{"cx": 300.0, "pixels": 900}, {"cx": 100.0, "pixels": 900}],
        cfg["target"])
    assert picked["cx"] == 100.0                   # 同像素 -> 取更小 cx, 可复现


def test_target_deadband_off_is_identity(cfg):
    """默认 deadband=0 -> 恒等, 与改动前逐字节一致。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    proc._target_last_err = 5000
    assert proc._apply_target_deadband(5200, cfg["target"]) == 5200


def test_target_deadband_holds_last_value(cfg):
    """死区内保持上次 err（不增加相位滞后, 与滑动平均的本质区别）。"""
    cfg["target"]["err_deadband"] = 400
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    proc._target_last_err = 5000
    assert proc._apply_target_deadband(5200, cfg["target"]) == 5000   # 桶内
    assert proc._apply_target_deadband(5600, cfg["target"]) == 5600   # 出桶


def test_target_deadband_actually_engages_via_confirm(cfg):
    """死区在真实调用路径上真的生效（2026-10-08 回归）。

    此前 `_target_last_err = err_fixed` 只写在 _target_confirm_ok 末尾, 而
    confirm_frames=1(默认)时该函数第一行就 return True -> 基准永不更新,
    死区恒等形同关闭。这条用例走 process() 全路径, 钉住"配了 deadband 就有效"。
    """
    cfg["target"]["err_deadband"] = 5000
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    #第 1 帧: 无历史 -> 直接通过, 并写入基准
    assert proc._target_confirm_ok(1000, cfg["target"]) is True
    assert proc._target_last_err == 1000
    # 第 2 帧: 桶内差值, 死区应保持 1000
    assert proc._apply_target_deadband(2000, cfg["target"]) == 1000


def test_target_last_err_zero_is_valid_not_sentinel(cfg):
    """err_fixed=0 是合法值（靶心正中), 不能被当成"尚无历史"。

    哨兵用 0 时首帧之后死区会一直失效；故用 None。
    """
    cfg["target"]["err_deadband"] = 400
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    assert proc._target_last_err is None           # 初始: None 而非 0
    proc._target_confirm_ok(0, cfg["target"])      # 正中 -> 基准 = 0
    assert proc._target_last_err == 0
    assert proc._apply_target_deadband(100, cfg["target"]) == 0   # 桶内保持 0


def test_target_confirm_off_by_default(cfg):
    """默认 confirm_frames=1 -> 恒 True, 不影响现有行为。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    assert proc._target_confirm_ok(1234, cfg["target"]) is True


def test_target_confirm_needs_deadband_to_bucket(cfg):
    """confirm_frames>1 且 deadband=0: 桶退化为 err 本身, 第 2 帧同值才认。"""
    cfg["target"]["confirm_frames"] = 2
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    assert proc._target_confirm_ok(5000, cfg["target"]) is False   # 第 1 帧
    assert proc._target_confirm_ok(5000, cfg["target"]) is True    # 第 2 帧同值


def test_target_confirm_drops_when_err_moves(cfg):
    """err 跨桶 -> 连续确认重新计数（换桶即视为未达N 帧）。"""
    cfg["target"]["confirm_frames"] = 2
    cfg["target"]["err_deadband"] = 1000
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    assert proc._target_confirm_ok(5000, cfg["target"]) is False
    assert proc._target_confirm_ok(9000, cfg["target"]) is False   # 换桶, 重新计
    assert proc._target_confirm_ok(9000, cfg["target"]) is True


def test_target_reset_clears_temporal_state(cfg):
    """换任务/STOP 后必须清 target 时序状态。

    否则新任务首帧会继承上一段的 continuity 历史 -> 一订阅就报一个来自旧场景
    的 err。这是加时序状态最容易埋的坑。
    """
    cfg["target"]["prefer_continuity"] = True
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    proc._target_last_cx = 520.0
    proc._target_last_err = 5000
    proc._target_last_bucket = 12
    proc._target_confirm_count = 2
    proc._target_drop_count = 7
    proc.reset()
    assert proc._target_last_cx is None
    assert proc._target_last_err is None
    assert proc._target_last_bucket is None
    assert proc._target_confirm_count == 0
    assert proc._target_drop_count == 0



# ---------- task2 降采样（target.scan_scale, 2026-10-07） ----------

@pytest.mark.parametrize("cx", [200, 320, 415, 480, 545])
def test_target_downscale_error_identical_to_full_res(cfg, cx):
    """降采样后 err_fixed 必须与全分辨率**逐位一致**（协议语义不可变）。

    面积门槛按 scale² 折算、cx 按 1/scale 还原到全帧坐标，两条折算路径各自
    都能引入偏差：门槛折错 → 漏检 found=0；坐标折错 → 误差值偏。
    这个用例把两者一起钉住：对多个 cx（含画面中心与两侧）比对全量程结果。
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[220:260, cx - 60:cx + 60] = RED
    # 面积门槛折算：0.5 -> 250。测试块面积 40x120=4800, 降采样后仍有 1200, 足够。
    results = {}
    for scale in (1.0, 0.5):
        cfg["target"]["scan_scale"] = scale
        proc = GksTaskProcessor(config=cfg)
        proc.set_task(MODE_TARGET, (0, 1, 0))
        r = proc.process(frame)
        assert r.result_data["found"], f"scale={scale} 漏检"
        results[scale] = r.result_data["err_fixed"]
    assert results[0.5] == results[1.0]


def test_target_downscale_hits_restored_to_full_frame(cfg, target_frame):
    """命中框坐标须还原到全帧, 否则 HUD 框与实际目标错位。"""
    cfg["target"]["scan_scale"] = 0.5
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    r = proc.process(target_frame)
    hits = r.result_data["hits"]
    assert len(hits) == 1
    h = hits[0]
    # 目标块实为 x 420..540 / y 220..260, 允许 cv2 轮廓拟合的少量偏差
    assert abs(h["x"] - 420) <= 4 and abs(h["y"] - 220) <= 4
    assert abs(h["w"] - 120) <= 8 and abs(h["h"] - 40) <= 8


def test_target_downscale_roi_excludes_outside(cfg):
    """降采样后 ROI 边界必须跟着缩, 否则 ROI 外的目标会漏进扫描区。"""
    cfg["target"]["scan_scale"] = 0.5
    cfg["target"]["roi"] = [0.0, 0.352, 1.0, 0.763]   # y 169..366
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[40:80, 420:540] = RED                       # ROI 之上, 距上边界 89px
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (0, 1, 0))
    r = proc.process(frame)
    assert not r.result_data["found"]


def test_target_downscale_area_threshold_shrinks(cfg):
    """门槛按 scale² 折算: 同一块目标在 0.5 档更容易过门槛（全分辨率下应漏检）。

    面积 40x40=1600 的小块: 全分辨率门槛 1000 -> 过; 0.5 档像素数 400,
    折算门槛 250 -> 也过。改用面积 1000 的边界块验证折算方向。
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[230:260, 300:340] = RED        # 40x30 = 1200 px
    cfg["target"]["scan_scale"] = 1.0
    proc_full = GksTaskProcessor(config=cfg)
    proc_full.set_task(MODE_TARGET, (0, 1, 0))
    full = proc_full.process(frame)

    cfg["target"]["scan_scale"] = 0.5
    proc_half = GksTaskProcessor(config=cfg)
    proc_half.set_task(MODE_TARGET, (0, 1, 0))
    half = proc_half.process(frame)

    # 两者都应检出且误差一致（块中心 cx=320 -> err=0）
    assert full.result_data["found"] and half.result_data["found"]
    assert half.result_data["err_fixed"] == full.result_data["err_fixed"] == 0


def test_target_scan_scale_invalid_falls_back_to_full(cfg):
    """非法值（<=0 / >1 / 非数字）必须退回 1.0, 不得静默变成 0 除法或异常比例。"""
    proc = GksTaskProcessor(config=cfg)
    for bad in (0.0, -0.5, 1.8, "abc", None):
        cfg["target"]["scan_scale"] = bad
        assert proc._scan_scale(cfg["target"]) == 1.0


def test_downscale_probes_maix_import_only_once(cfg, monkeypatch):
    """maix.image 探测结论必须被缓存, 不得每帧重抛 ImportError。

    这不是洁癖: PC 端(无 maix) 逐帧重抛一次 ImportError 实测 0.92ms/帧,
    而 _downscale 的有效工作只有 0.07ms —— 93% 的时间在重复失败上。
    设备端 import 命中 sys.modules 便宜得多, 但同样不该逐帧走一遍。
    """
    import sys
    import types
    import modules.zw_opencv_module.processors.gks_task_processor as gtp
    # 确保 sys.modules 里没有 maix, 并复位缓存到"未探测"
    monkeypatch.setitem(sys.modules, "maix", None)
    monkeypatch.setattr(gtp, "_maix_image_ok", None, raising=False)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    proc = GksTaskProcessor(config=cfg)

    calls = {"n": 0}

    def _counting_import(name, *a, **kw):
        if name == "maix.image":
            calls["n"] += 1
        raise ImportError("no maix (test)")

    # 屏蔽真实的 maix import, 统计被触发的次数
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name.startswith("maix"):
            _counting_import(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    proc._downscale(frame, None, 0.5)
    assert gtp._maix_image_ok is False, "探测不到时必须记住不可用"
    first = calls["n"]
    assert first == 1

    for _ in range(5):
        proc._downscale(frame, None, 0.5)
    assert calls["n"] == first, f"缓存失效: 5 帧内又触发了 {calls['n'] - first} 次 import"


# ---------- task3 人质 ----------

class Det:
    def __init__(self, x, y, w, h, class_id, score=0.9):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.class_id, self.score = class_id, score


class StubAI:
    """MaixCam2AI 检测接口桩; labels = model.mud 顺序 yao_gu, yuan_zhu, yuan_tai。"""
    loaded = True
    labels = ["yao_gu", "yuan_zhu", "yuan_tai"]

    def __init__(self, dets):
        self._dets = dets

    def detect(self, frame, conf_th=0.5, iou_th=0.5, **kwargs):
        return self._dets


@pytest.fixture()
def rescue_ai():
    # 左圆柱(1) 中圆台(2) 右腰鼓(3)
    return StubAI([Det(60, 100, 80, 120, class_id=1),
                   Det(280, 100, 80, 120, class_id=2),
                   Det(500, 100, 80, 120, class_id=0)])


def test_rescue_two_frame_latch(cfg, rescue_ai):
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(rescue_ai)
    proc.set_task(MODE_RESCUE, (0, 0, 3))       # 人质形状=腰鼓(3) -> 右坑
    r1 = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r1.result_data["found"]
    r2 = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r2.result_data["found"] and r2.result_data["pos"] == 3
    assert r2.result_data["latched_order"] == [1, 2, 3]


def test_rescue_model_missing_zero_heartbeat(cfg):
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(None)
    proc.set_task(MODE_RESCUE, (0, 0, 1))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and r.result_data["pos"] == 0


def test_rescue_zero_shape_code_never_self_matches(cfg, rescue_ai):
    """守卫: 形状码未下发(0)时不得与未锁存的 0 自匹配报 pos=1（审查 P2）。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(rescue_ai)
    proc.set_task(MODE_RESCUE, (0, 0, 0))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and r.result_data["pos"] == 0


def test_rescue_x_crop_filter(cfg):
    cfg["rescue"]["x_crop_px"] = 240            # 旧行为复现: obj.x>240 丢弃
    ai = StubAI([Det(60, 100, 80, 120, class_id=1),
                 Det(250, 100, 80, 120, class_id=2),
                 Det(500, 100, 80, 120, class_id=0)])
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(ai)
    proc.set_task(MODE_RESCUE, (0, 0, 1))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        r = proc.process(frame)
    assert not r.result_data["found"]           # 只有 1 个形状可见, 永不齐三


def test_rescue_no_crop_by_default(cfg):
    ai = StubAI([Det(400, 100, 80, 120, class_id=1)])   # 右侧目标不被裁
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(ai)
    proc.set_task(MODE_RESCUE, (0, 0, 1))
    r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["hits"]                # 检测有输出(仅 1 形, 未锁存)


# ---------- task3 未见形状补位 (infer_missing) ----------

def test_rescue_infer_missing_tail(cfg):
    """只见到前两个坑: 未见者(腰鼓)是顺序上的最后一个 -> 补到 pit3。"""
    ai = StubAI([Det(60, 100, 80, 120, class_id=1),      # 左 圆柱(1)
                 Det(280, 100, 80, 120, class_id=2)])    # 中 圆锥(2), 右 腰鼓(3) 未见
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(ai)
    proc.set_task(MODE_RESCUE, (0, 0, 3))                # 目标 = 未见者 -> 坑3
    r1 = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r1.result_data["found"] and r1.result_data["pending_count"] == 1
    r2 = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r2.result_data["found"] and r2.result_data["pos"] == 3
    assert r2.result_data["latched_order"] == [1, 2, 3]
    assert r2.result_data["inferred"] is True
    assert proc.pit_status["inferred"] is True


def test_rescue_infer_missing_target_visible(cfg):
    """目标形状本身已见: 补位不影响它在最右坑以外的判定(圆锥=中坑)。"""
    ai = StubAI([Det(60, 100, 80, 120, class_id=1),      # 左 圆柱(1)
                 Det(280, 100, 80, 120, class_id=2)])    # 中 圆锥(2)
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(ai)
    proc.set_task(MODE_RESCUE, (0, 0, 2))
    for _ in range(2):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["pos"] == 2


class SeqAI(StubAI):
    """按帧序列返回不同检测集(模拟相机扫过三坑：先出现一个, 再出现其余)。"""

    def __init__(self, seq):
        super().__init__([])
        self._seq = seq
        self._i = 0

    def detect(self, frame, conf_th=0.5, iou_th=0.5, **kwargs):
        dets = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return list(dets)


def test_rescue_order_is_first_seen_across_frames(cfg):
    """默认 order_by=first_seen: 跨帧"首次检出先后"决定坑位, 不是画面 x 顺序。

    先单独检出的形状(画面最右的腰鼓)就是 pit1 —— 若按 x 排序会得到 pit3。
    """
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(SeqAI([[Det(500, 100, 80, 120, class_id=0)],          # 第1帧: 只有腰鼓(3)
                       [Det(60, 100, 80, 120, class_id=1)],           # 第2帧: 圆柱(1)
                       [Det(280, 100, 80, 120, class_id=2)]]))        # 第3帧: 圆锥(2)
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(3):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["latched_order"] == [3, 1, 2]
    assert r.result_data["pos"] == 1
    assert r.result_data["inferred"] is False    # 锁存那一帧三形状已齐 -> 未用补位


def test_rescue_first_seen_multi_frame_ignores_mirror(cfg):
    """跨帧的首次检出顺序是物理顺序, pit_left_is_1(图像镜像) 不应影响它。"""
    out = []
    for mirror in (True, False):
        cfg["pit_left_is_1"] = mirror
        proc = GksTaskProcessor(config=cfg)
        proc.set_ai(SeqAI([[Det(500, 100, 80, 120, class_id=0)],
                           [Det(60, 100, 80, 120, class_id=1)],
                           [Det(280, 100, 80, 120, class_id=2)]]))
        proc.set_task(MODE_RESCUE, (0, 0, 3))
        for _ in range(3):
            r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
        out.append(r.result_data["latched_order"])
    assert out[0] == out[1] == [3, 1, 2]


def test_rescue_same_frame_uses_left_right_not_model_order(cfg):
    """同帧同时检出多个形状: 帧内按画面左右定序, 不受模型返回顺序影响。"""
    dets = [Det(500, 100, 80, 120, class_id=0),            # 模型先返回最右
            Det(60, 100, 80, 120, class_id=1),
            Det(280, 100, 80, 120, class_id=2)]
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(StubAI(dets))
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(2):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["latched_order"] == [1, 2, 3]     # 帧内按 x, 不是返回顺序 [3,1,2]

    cfg["pit_left_is_1"] = False                           # 镜像: 帧内反向定序
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(StubAI(dets))
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(2):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["latched_order"] == [3, 2, 1]


def test_rescue_order_by_x_mode(cfg):
    """order_by=x 退回旧语义: 画面 x 从左到右排序, pit_left_is_1=False 时整体翻转。"""
    dets = [Det(500, 100, 80, 120, class_id=0),           # 画面最右 = 腰鼓(3)
            Det(60, 100, 80, 120, class_id=1)]            # 画面最左 = 圆柱(1)
    cfg["rescue"]["order_by"] = "x"
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(StubAI(dets))
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(2):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["pos"] == 2                       # x 序 [1,3,2] -> 腰鼓在 pit2
    assert r.result_data["latched_order"] == [1, 3, 2]

    cfg["pit_left_is_1"] = False                           # 镜像翻转: [3,1,2]
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(StubAI(dets))
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(2):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert r.result_data["pos"] == 1 and r.result_data["latched_order"] == [3, 1, 2]


def test_rescue_infer_missing_disabled(cfg):
    """infer_missing=none: 退回旧行为, 三形状不齐见永不锁存。"""
    cfg["rescue"]["infer_missing"] = "none"
    ai = StubAI([Det(60, 100, 80, 120, class_id=1),
                 Det(280, 100, 80, 120, class_id=2)])
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(ai)
    proc.set_task(MODE_RESCUE, (0, 0, 3))
    for _ in range(5):
        r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not r.result_data["found"] and r.result_data["pos"] == 0
    assert r.result_data["inferred"] is False


def test_rescue_three_shapes_unchanged_by_infer_missing(cfg):
    """三形状齐见时, 开/关补位必须给出完全一致的结果(保证不打乱已台架验证的路径)。"""
    dets = [Det(60, 100, 80, 120, class_id=1),
            Det(280, 100, 80, 120, class_id=2),
            Det(500, 100, 80, 120, class_id=0)]
    out = []
    for mode in ("tail", "none"):
        cfg["rescue"]["infer_missing"] = mode
        proc = GksTaskProcessor(config=cfg)
        proc.set_ai(StubAI(dets))
        proc.set_task(MODE_RESCUE, (0, 0, 3))
        for _ in range(2):
            r = proc.process(np.zeros((480, 640, 3), dtype=np.uint8))
        out.append((r.result_data["pos"], r.result_data["latched_order"],
                    r.result_data["inferred"]))
    assert out[0] == out[1] == (3, [1, 2, 3], False)


# ---------- 阈值文件 ----------

def test_threshold_file_loader(tmp_path):
    f = tmp_path / "my_64threshold.txt"
    f.write_text(
        "exp_red|0,80,40,80,10,80|\n"
        "\n"
        "exp_green|0,80,-120,-10,0,30|\n"
        "exp_blue|0,80,30,100,-120,-60|\n"
        "tgt_red|0,60,50,90,20,70|\n"
        "tgt_green|0,70,-100,-20,0,40|\n"
        "tgt_blue|0,90,40,110,-110,-50|\n"
        "bad line without pipe\n", encoding="utf-8")
    groups = load_threshold_file(str(f))
    assert len(groups) == 6
    assert groups[0] == [0, 80, 40, 80, 10, 80]
    assert groups[5] == [0, 90, 40, 110, -110, -50]


def test_threshold_hot_reload_preserves_latch(cfg, tmp_path, pit_frame):
    f = tmp_path / "my_64threshold.txt"
    f.write_text("x|0,1,2,3,4,5\n", encoding="utf-8")   # 不足 6 组: 不覆盖
    cfg["thresholds_file"] = str(f)
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (2, 1, 3))
    for _ in range(2):
        proc.process(pit_frame)
    assert proc._pos_flag == 1
    import os, time
    os.utime(str(f), None)                      # mtime 变化: 重载但不清锁存
    # ⚠ _maybe_reload_thresholds 现在**限频到 2Hz**（每帧 stat 在真机上要 ~1ms,
    #   见其 docstring）。这里把上次轮询时间戳归零, 强制本次真的去 stat,
    #   否则"热重载"这条断言会因为节流而根本没走到重载分支。
    proc._thresholds_poll_ts = 0.0
    r = proc.process(pit_frame)
    assert r.result_data["found"]


def test_threshold_poll_is_rate_limited(cfg, tmp_path, pit_frame, monkeypatch):
    """mtime 轮询必须限频: 连续多帧不得各做一次 os.stat。

    动机是台架实测 —— 未订阅/画面无目标时 `[PERF] task=1.1ms`, 而那条路径上
    唯一真正做事的只有这次 stat（`_mode` 为 None, 识别分支一个都没进）。
    真机上一次文件系统调用约 1ms, 每帧一次就是整帧预算的 4%+。
    阈值文件由人手在调参页写出, 2Hz 轮询绰绰有余。
    """
    import modules.zw_opencv_module.processors.gks_task_processor as gks_mod

    f = tmp_path / "my_64threshold.txt"
    f.write_text("x|0,1,2,3,4,5\n", encoding="utf-8")
    cfg["thresholds_file"] = str(f)
    proc = GksTaskProcessor(config=cfg)

    class _CountingOs:
        """包住真 os, 只对目标路径计数; 其余属性（os.path 等）原样透出。"""

        def __init__(self, real):
            self._real = real
            self.calls = 0

        def stat(self, path, *a, **kw):
            if str(path) == str(f):
                self.calls += 1
            return self._real.stat(path, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    fake = _CountingOs(gks_mod.os)
    monkeypatch.setattr(gks_mod, "os", fake)

    proc.process(pit_frame)          # 首帧: 时间戳初值 0.0 -> 允许轮询
    assert fake.calls == 1, "首帧应轮询一次"
    for _ in range(5):               # 紧接着的 5 帧: 全部被节流挡住
        proc.process(pit_frame)
    assert fake.calls == 1, f"节流期内不应再 stat（实际 {fake.calls} 次）"

    proc._thresholds_poll_ts = 0.0   # 手动放开 -> 下一次 process 又该查一次
    proc.process(pit_frame)
    assert fake.calls == 2, "放开节流后应恢复轮询"


# ---------- 打靶跨帧跟踪窄窗（target.track_roi_px, 2026-10-09）----------
#
# 设计上的硬不变量: 跟踪**只在命中时省时间**, 绝不改变结果 ——
# 窄窗未命中时同一帧内回退全 ROI 复扫, 所以检出率与精度与"全程全 ROI"逐位一致。
# 这几条测试就是钉住这个不变量; pad 取值是否合适属于台架调参, 不在这里断言。

def test_target_track_window_matches_full_roi(cfg, pit_frame):
    """命中时 err 必须与全 ROI 路径**逐位一致**（跟踪只省时间, 不改结果）。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))

    r0 = proc.process(pit_frame).result_data           # 无历史 -> 走全 ROI
    assert r0["found"] is True
    assert r0["err_fixed"] == -6656                    # 红块 cx=107 @640 宽

    cfg["target"]["track_roi_px"] = 120               # 窗口夹到 [0,227], 含整个红块
    r1 = proc.process(pit_frame).result_data
    assert (r1["found"], r1["err_fixed"]) == (r0["found"], r0["err_fixed"])
    assert proc._target_track_hits == 1, "第二次应当走的是窄窗路径"
    assert proc._target_track_miss == 0


def test_target_track_miss_falls_back_in_same_frame(cfg, pit_frame):
    """窄窗落空时**当帧回退全 ROI**: 仍要检出, 且 err 与全 ROI 一致。

    这条是安全底线。若退化成"窄窗没命中就报 found=0", 主站会以为本帧无结果,
    而实际目标就在画面里 —— 那比慢一点严重得多。
    """
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    cfg["target"]["track_roi_px"] = 8
    proc.process(pit_frame)                            # 建立历史

    proc._target_last_cx = 620.0                       # 挪到右侧空白区 -> 窄窗必落空
    r = proc.process(pit_frame).result_data
    assert r["found"] is True, "窄窗落空必须当帧回退全 ROI, 不能漏检"
    assert r["err_fixed"] == -6656
    assert proc._target_track_miss == 1
    assert proc._target_track_hits == 0


def test_target_track_off_by_default(cfg, pit_frame):
    """默认 track_roi_px=0 -> 一次都不走窄窗（改动必须是 opt-in）。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    assert cfg["target"]["track_roi_px"] == 0
    for _ in range(3):
        proc.process(pit_frame)
    assert proc._target_track_hits == 0
    assert proc._target_track_miss == 0


def test_target_track_window_clamped_and_degenerate():
    """窄窗必须夹回 ROI; 退化窗口返回 None（调用方回退全 ROI）。"""
    roi = [0, 169, 640, 197]
    assert GksTaskProcessor._target_track_window(0, roi, 100) == [0, 169, 100, 197]
    assert GksTaskProcessor._target_track_window(639, roi, 100) == [539, 169, 101, 197]
    assert GksTaskProcessor._target_track_window(320, roi, 50) == [270, 169, 100, 197]
    # pad 太小 -> 窗口宽 <= 2px, 无意义, 返回 None
    assert GksTaskProcessor._target_track_window(319, roi, 1) is None
    assert GksTaskProcessor._target_track_window(640, roi, 0) is None


def test_target_track_not_used_by_explosive_or_rescue(cfg, pit_frame):
    """排爆/人质不得缩小搜索域: 三色定序与 YOLO 全帧都依赖完整画面。

    这里只钉最直接的一条: 排爆模式下 `target.track_roi_px` 再大也不会被读到
    （`_target_tracked_scan` 只在 `_process_target` 里被调用）。
    """
    cfg["target"]["track_roi_px"] = 200
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_EXPLOSIVE, (1, 2, 3))
    for _ in range(3):
        proc.process(pit_frame)
    assert proc._target_track_hits == 0
    assert proc._target_track_miss == 0


# ---------- 动态跟踪窗（target.track_mode=dynamic, 2026-10-15）----------
#
# 不变量同 fixed: 跟踪只省时间不改结果, 未命中当帧回退全 ROI。

def test_target_dynamic_window_geometry():
    """dynamic 窗几何: 缩放/封顶/夹回/退化。"""
    roi = [0, 169, 640, 197]
    c = {"track_win_scale": 4.0, "track_win_min_w": 48, "track_v_pad_px": 16}
    box = {"x": 300, "y": 250, "w": 40, "h": 40, "cx": 320.0, "cy": 270.0}
    # 4*40=160 超横向封顶 -> 宽 128; 高 40+32=72
    assert GksTaskProcessor._target_dynamic_window(box, roi, 64, c) == [256, 234, 128, 72]
    small = {"x": 310, "y": 260, "w": 20, "h": 20, "cx": 320.0, "cy": 270.0}
    assert GksTaskProcessor._target_dynamic_window(small, roi, 64, c) == [280, 244, 80, 52]
    # 贴边夹回 ROI
    left = {"x": 0, "y": 170, "w": 100, "h": 100, "cx": 10.0, "cy": 175.0}
    assert GksTaskProcessor._target_dynamic_window(left, roi, 64, c) == [0, 169, 128, 132]
    # 退化窗口 -> None（回退全 ROI）
    assert GksTaskProcessor._target_dynamic_window(box, [0, 0, 640, 2], 64, c) is None


def test_target_dynamic_track_hit_matches_full_roi(cfg, pit_frame):
    """dynamic 命中: err 与全 ROI 逐位一致。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    r0 = proc.process(pit_frame).result_data
    assert r0["found"] is True
    cfg["target"]["track_mode"] = "dynamic"
    cfg["target"]["track_roi_px"] = 64
    r1 = proc.process(pit_frame).result_data
    assert (r1["found"], r1["err_fixed"]) == (r0["found"], r0["err_fixed"])
    assert proc._target_track_hits == 1 and proc._target_track_miss == 0


def test_target_dynamic_track_miss_falls_back(cfg, pit_frame):
    """dynamic 未命中: 当帧回退全 ROI, 不漏检。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    cfg["target"]["track_mode"] = "dynamic"
    cfg["target"]["track_roi_px"] = 64
    proc.process(pit_frame)                            # 建立历史
    proc._target_last_box = {"x": 600, "y": 200, "w": 20, "h": 20,
                             "cx": 610.0, "cy": 210.0}  # 空白区 -> 窗必落空
    r = proc.process(pit_frame).result_data
    assert r["found"] is True and r["err_fixed"] == -6656
    assert proc._target_track_miss == 1 and proc._target_track_hits == 0


def test_target_track_mode_off_and_invalid(cfg, pit_frame):
    """off 不扫; 非法 mode 按 fixed。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    cfg["target"]["track_roi_px"] = 64
    cfg["target"]["track_mode"] = "off"
    for _ in range(3):
        proc.process(pit_frame)
    assert proc._target_track_hits == 0 and proc._target_track_miss == 0
    cfg["target"]["track_mode"] = "nonsense"
    proc.process(pit_frame)   # last_cx 已有历史 -> 立即按 fixed 走窄窗
    assert proc._target_track_hits == 1


def test_target_dynamic_state_cleared_on_reset(cfg, pit_frame):
    """换任务/STOP 后 box 与窗必须清, 首帧不得继承旧窗。"""
    proc = GksTaskProcessor(config=cfg)
    proc.set_task(MODE_TARGET, (1, 1, 1))
    cfg["target"]["track_mode"] = "dynamic"
    cfg["target"]["track_roi_px"] = 64
    proc.process(pit_frame)
    assert proc._target_last_box is not None
    proc._reset_latch()
    assert proc._target_last_box is None and proc.target_track_win() is None


def test_target_filter_by_fill():
    """填充率过滤: 环留矩形踢; 默认不过滤; 全被滤掉回退原列表。"""
    ring = {"pixels": 2000, "w": 60, "h": 60}   # fill ≈ 0.556
    rect = {"pixels": 8000, "w": 95, "h": 90}   # fill ≈ 0.934
    f = GksTaskProcessor._filter_by_fill
    assert f([rect, ring], {"fill_min": 0.3, "fill_max": 0.85}) == [ring]
    assert f([rect, ring], {}) == [rect, ring]                        # 默认全开区间
    assert f([rect], {"fill_min": 0.3, "fill_max": 0.85}) == [rect]   # 空集回退
