# -*- coding: utf-8 -*-
"""人质坑位 —— 入表连续帧门槛（shape_hits）。

现场根因（docs/eod/rescue_shape_identity.md）：目标从画面右侧入画的首帧半出画 +
运动模糊，YOLO 判成邻类（yuan_zhu 0.59 而非 yuan_tai），下一帧起 0.80~0.92 全对。
旧代码首帧即写 `_shape_seen`、键永驻，后续正确帧只能新增另一个键 —— 集合凭空多一个
形状，"三形状齐见" → 静默锁死错误坑位。Brain 侧首次收到非零 pos 即锁死
`align_target_pit`，事后无法纠正，所以必须在**锁存前**挡住。

这里钉住四件事：
1. 幽灵 1 帧 → 被门槛挡住（复刻 2026-11-04 03:37 段）；
2. shape_hits=1 → 与改前逐位等价（默认值即回归开关）；
3. 幽灵**连续** 3 帧 → 仍然穿透（方案的已知上限，不是回归）；
4. 先入画的坑中途漏检、比后入画的晚达标 → 按"首次出现"重排，顺序不反。

class_id 映射（model.mud labels 顺序）: 0=yao_gu(形状3), 1=yuan_zhu(1), 2=yuan_tai(2)
"""

import numpy as np
import pytest

from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor,
    MODE_RESCUE,
    load_gks_config,
)

W, H = 640, 480
Y, BW, BH = 100, 80, 120


class Det:
    def __init__(self, x, y, w, h, class_id, score=0.9):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.class_id, self.score = class_id, score


def _det(cx, class_id, score=0.9):
    """按画面中心 x 构造一个框（宽 80），便于直接抄日志里的 cx。"""
    return Det(cx - BW // 2, Y, BW, BH, class_id, score)


class SeqAI:
    """按帧序列返回检测集；序列走完后粘住最后一帧（模拟目标停在画内）。"""

    loaded = True
    labels = ["yao_gu", "yuan_zhu", "yuan_tai"]

    def __init__(self, seq):
        self._seq = seq
        self._i = 0

    def detect(self, frame, conf_th=0.5, iou_th=0.5, **kwargs):
        dets = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return list(dets)


@pytest.fixture()
def cfg():
    # 只读代码默认值（shape_hits=1），与设备端 YAML 解耦
    c = load_gks_config(config_path="/nonexistent/eod_config.yaml")
    c["thresholds_file"] = ""
    return c


def _run(cfg, seq, frames=None, shape_code=2):
    proc = GksTaskProcessor(config=cfg)
    proc.set_ai(SeqAI(seq))
    proc.set_task(MODE_RESCUE, (0, 0, shape_code))
    r = None
    for _ in range(frames or len(seq)):
        r = proc.process(np.zeros((H, W, 3), dtype=np.uint8))
    data = r.result_data
    return data, proc.pit_status


# 复刻 03:37 段：pit1=yao_gu(3), pit2=yuan_tai(2), pit3=yuan_zhu(1 补位)
# 帧3 的 yuan_zhu(570) 是幽灵，随后同一轨迹 539→516→467→438→403 全是 yuan_tai。
GHOST_1_FRAME = [
    [_det(572, 0)],
    [_det(383, 0)],
    [_det(129, 0), _det(570, 1, 0.59)],   # ← 幽灵：入画首帧判成圆柱
    [_det(81, 0), _det(539, 2)],
    [_det(71, 0), _det(516, 2)],
    [_det(55, 0), _det(467, 2)],
    [_det(35, 0), _det(438, 2)],
    [_det(403, 2)],
]


def test_one_frame_ghost_is_dropped(cfg):
    """幽灵只活 1 帧：shape_hits=3 时它攒不够 3 帧，进不了 _shape_seen。"""
    cfg["rescue"]["shape_hits"] = 3
    data, st = _run(cfg, GHOST_1_FRAME)
    assert st["shapes_seen"].keys() == {3, 2}          # 没有幽灵 1
    assert data["latched_order"] == [3, 2, 1]          # 未见者(圆柱)补到 pit3
    assert data["inferred"] is True
    assert data["pos"] == 2                            # 任务码 2(yuan_tai) 在 pit2


def test_shape_hits_one_is_legacy_behaviour(cfg):
    """默认 shape_hits=1 必须与改前完全一致：首帧即占坑 → 三形状齐见 → pos 错。"""
    assert cfg["rescue"]["shape_hits"] == 1
    data, st = _run(cfg, GHOST_1_FRAME)
    assert list(st["shapes_seen"]) == [3, 1, 2]
    assert data["latched_order"] == [3, 1, 2]
    assert data["inferred"] is False
    assert data["pos"] == 3                            # 改前的错误结果（文档 §6）


def test_ghost_three_frames_still_passes(cfg):
    """已知上限：幽灵连续 3 帧就攒够门槛，照样穿透。不是回归，改门槛前必读。

    要挡住它只能把 shape_hits 抬到 4，代价是每坑多要求 1 帧可见；
    幽灵连续多帧说明"半出画+模糊"不是唯一成因，得回去看模型/入画位置。
    """
    cfg["rescue"]["shape_hits"] = 3
    seq = [
        [_det(572, 0)],
        [_det(500, 0), _det(570, 1)],       # 幽灵第 1 帧
        [_det(400, 0), _det(540, 1)],       # 幽灵第 2 帧
        [_det(300, 0), _det(500, 1)],       # 幽灵第 3 帧 -> 达标入表
        [_det(200, 0), _det(460, 2)],
        [_det(150, 0), _det(430, 2)],
        [_det(100, 0), _det(400, 2)],       # 真 yuan_tai 凑够 3 帧
        [_det(60, 0), _det(370, 2)],
    ]
    data, st = _run(cfg, seq)
    assert list(st["shapes_seen"]) == [3, 1, 2]        # 幽灵已占位
    assert data["pos"] == 3                            # 已知错误结果


def test_late_shape_is_not_pushed_behind(cfg):
    """先入画的坑漏检 1 帧 -> 比后入画的晚达标，仍要排在前面。

    帧2 整个空掉，把 yao_gu 的 streak 清零：它在帧 7 才达标，而 yuan_tai 帧 5 就达标。
    若不按"首次出现"重排，_shape_seen 会变成 [2, 3]，坑位整体反掉。
    """
    cfg["rescue"]["shape_hits"] = 3
    seq = [
        [_det(560, 0)],                     # 帧1 yao_gu 首现 (first=0)
        [],                                 # 帧2 漏检 -> streak 清零
        [_det(500, 0), _det(300, 2)],       # 帧3 yuan_tai 首现 (first=1)
        [_det(300, 2)],                     # 帧4
        [_det(400, 0), _det(260, 2)],       # 帧5 yuan_tai 达标入表
        [_det(360, 0), _det(220, 2)],       # 帧6
        [_det(320, 0), _det(180, 2)],       # 帧7 yao_gu 达标 -> 重排回 [3, 2]
        [_det(280, 0), _det(140, 2)],
        [_det(240, 0), _det(100, 2)],
    ]
    data, st = _run(cfg, seq)
    assert list(st["shapes_seen"]) == [3, 2]
    assert data["latched_order"] == [3, 2, 1]
    assert data["pos"] == 2
