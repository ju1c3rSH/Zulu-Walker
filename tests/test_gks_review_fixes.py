# -*- coding: utf-8 -*-
"""审查修复回归测试 — GKS processor 状态流转 4 项改动。

- 改动1: PipelineCamera.enable_task 重注 set_ai（STOP 后 RESCUE 不永久失效）
- 改动2: 相机停帧时 process_frame 不产出结果（陈旧归零可达）
- 改动3: 各平台 read_raw/last_raw 接口一致（linux/mock 可跑主流水线）
- 改动4: AI 为 None 时订阅 RESCUE 回 NACK NOT_READY（不误导 Brain）
"""
import numpy as np
import pytest

from framework.event_bus import EventBus
from app.coordinator import GksCoordinator
from modules.zw_opencv_module.pipeline_camera import PipelineCamera
from modules.zw_opencv_module.processors.gks_task_processor import (
    GksTaskProcessor,
    MODE_RESCUE,
)
from modules.zw_opencv_module.task_manager import TaskManager, Task
from modules.zw_uart_module.events import CmdRequestEvent
from modules.zw_uart_module.protocol import (
    build_cmd_nack_frame,
    DATA_RESCUE,
    NACK_NOT_READY,
)


# ---------- 改动1: enable_task 重注 AI ----------

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


def _rescue_ai_left123():
    # 左圆柱(1) 中圆台(2) 右腰鼓(3)
    return StubAI([Det(60, 100, 80, 120, class_id=1),
                   Det(280, 100, 80, 120, class_id=2),
                   Det(500, 100, 80, 120, class_id=0)])


class FakeCamera:
    """最小相机桩：有 read_raw/last_raw，无 maix（走 cv2 回退路径）。"""

    def __init__(self, frame=None):
        self._frame = frame
        self.last_raw = None

    @property
    def camera_id(self):
        return "fake"

    @property
    def fps(self):
        return 30.0

    def read(self):
        return self._frame

    def read_raw(self):
        return self._frame

    def set(self, prop_id, value):
        return True

    def release(self):
        pass


def _make_pipeline(frame, ai=None):
    cam = FakeCamera(frame)
    pipe = PipelineCamera.__new__(PipelineCamera)
    pipe.pipeline_id = "cam_main"
    pipe.camera = cam
    pipe.task_manager = TaskManager()
    pipe.focal_calculator = None
    pipe._last_frame = None
    pipe._last_results = {}
    pipe._ai = ai
    proc = GksTaskProcessor("gks_task")
    if ai is not None:
        proc.set_ai(ai)
    pipe.task_manager.register_task(Task("gks_task", proc, True))
    return pipe


def test_enable_task_reinjects_ai_after_release():
    """STOP(disable→release清空_ai) 后重订阅(enable) 必须恢复 AI 句柄，
    否则 RESCUE 永久走模型缺失零值分支。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ai = _rescue_ai_left123()
    pipe = _make_pipeline(frame, ai=ai)
    proc = pipe.get_task("gks_task").processor
    proc.set_task(MODE_RESCUE, (0, 0, 3))

    for _ in range(2):
        r = pipe.process_frame()[1]["gks_task"]
    assert r.result_data["found"] and r.result_data["pos"] == 3

    pipe.disable_task("gks_task")     # Task.disable → processor.release → _ai=None
    assert proc._ai is None
    assert pipe.enable_task("gks_task") is True
    assert proc._ai is ai             # 重启用时重新注入共享 AI

    r = pipe.process_frame()[1]["gks_task"]
    assert r.result_data["found"] and r.result_data["pos"] == 3


# ---------- 改动2: 停帧不产出 ----------

def test_process_frame_no_frame_yields_no_results():
    """read_raw()->None 时直接 (None, {})：不复用旧帧、不产出结果，
    coordinator 的 _latest_task_ts 停滞，1s 后归零心跳可达。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipe = _make_pipeline(frame)
    out_frame, results = pipe.process_frame()
    assert out_frame is not None and results != {}
    pipe.camera._frame = None          # 模拟相机停帧
    out_frame, results = pipe.process_frame()
    assert out_frame is None
    assert results == {}


# ---------- 改动3: 平台接口一致 ----------

def test_mock_camera_has_read_raw():
    from framework.hal.platforms.mock.camera import MockCamera
    cam = MockCamera("mock")
    assert callable(getattr(cam, "read_raw", None))
    assert cam.last_raw is None
    frame = cam.read_raw()             # 不抛、返回合成 BGR 帧
    assert isinstance(frame, np.ndarray) and frame.shape == (480, 640, 3)


def test_linux_camera_read_raw_empty_queue_returns_none():
    from framework.hal.platforms.linux.camera import LinuxCamera
    cam = LinuxCamera(camera_id="t", source=0)   # 构造不碰硬件
    assert callable(getattr(cam, "read_raw", None))
    assert cam.last_raw is None
    assert cam.read_raw() is None      # 空队列 = 停帧语义


# ---------- 改动4: AI 为 None 时 NACK ----------

def test_rescue_without_ai_nacks_not_ready():
    """AI 未初始化（self._ai is None）时订阅 RESCUE：回 NACK NOT_READY，
    不订阅不推流，不误导 Brain 以为订阅成功。"""
    bus = EventBus()
    coord = GksCoordinator(bus)
    sent = []
    coord.set_uart_sender(lambda f: sent.append(f) or True)
    coord.start()
    assert coord._ai is None           # 默认无 AI

    event = CmdRequestEvent(data_type=DATA_RESCUE, min_interval_ms=50)
    event.explosive_color, event.target_color, event.rescue_shape = (0, 0, 3)
    event.has_codes = True
    coord._on_cmd_request(event)

    assert sent[0] == build_cmd_nack_frame(DATA_RESCUE, NACK_NOT_READY)
    assert coord._sub_id == 0          # 未订阅
