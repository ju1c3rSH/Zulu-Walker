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
    r = proc.process(pit_frame)
    assert r.result_data["found"]
