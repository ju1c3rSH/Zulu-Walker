# -*- coding: utf-8 -*-
"""审查修复回归测试 — GKS processor 状态流转 4 项改动。

- 改动1: PipelineCamera.enable_task 重注 set_ai（STOP 后 RESCUE 不永久失效）
- 改动2: 相机停帧时 process_frame 不产出结果（陈旧归零可达）
- 改动3: 各平台 read_raw/last_raw 接口一致（linux/mock 可跑主流水线;
          maixcam2 停帧不得回退旧帧 —— 2026-09-28 补）
- 改动4: AI 为 None 时订阅 RESCUE 回 NACK NOT_READY（不误导 Brain）
"""
import os

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


# ---------- 改动3b: maixcam2 停帧不得回退旧帧（2026-09-28） ----------

_MAIXCAM2_CAMERA_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "framework", "hal", "platforms", "maixcam2", "camera.py")


class _FakeRawCam:
    """maix.camera.Camera 桩: read(block=False) 的返回值可控。"""

    def __init__(self, frame):
        self._frame = frame

    def read(self, block=True):        # noqa: A002 - 对齐固件签名
        return self._frame


def _load_maixcam2_camera(monkeypatch, image2cv):
    """注入假 maix 后按文件路径加载相机模块。

    不走 `framework.hal.platforms.maixcam2` 包（其 __init__ 还拉 .ai/.display/.uart
    等一堆 maix 依赖），直接按文件 exec —— 这样 PC 端也能测设备平台实现。
    """
    import importlib.util
    import sys
    import types

    image_mod = types.ModuleType("maix.image")
    image_mod.image2cv = image2cv
    camera_mod = types.ModuleType("maix.camera")
    camera_mod.Camera = object
    maix_mod = types.ModuleType("maix")
    maix_mod.image = image_mod
    maix_mod.camera = camera_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.camera", camera_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)

    spec = importlib.util.spec_from_file_location("_maixcam2_camera_ut",
                                                  _MAIXCAM2_CAMERA_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maixcam2_read_raw_returns_none_on_stall(monkeypatch):
    """空队列（相机停帧）必须返回 None, **不得回退 _last_frame**。

    回退旧帧会让 PipelineCamera 的"停帧不复用旧帧"早退在设备上失效: 视觉线程
    把同一帧反复当新帧算（any_fresh 恒真、只 sleep(0) 忙转），结果时间戳持续
    刷新 -> coordinator 的 1s 陈旧归零永不触发 -> 相机挂死时错误结果会以
    found=1 的姿态被持续推给主站。显示侧仍读 last_raw 画旧画面, 不受影响。
    """
    mod = _load_maixcam2_camera(
        monkeypatch, lambda raw, **kw: np.zeros((4, 4, 3), np.uint8))
    cam = object.__new__(mod.MaixCam2Camera)
    cam._last_frame = np.full((4, 4, 3), 7, np.uint8)   # 上一帧仍在缓存
    cam._last_raw = object()
    cam._cam = _FakeRawCam(None)                        # 队列空 = 停帧

    assert cam.read_raw() is None
    assert cam.last_raw is not None                     # 显示侧照旧
    assert int(cam._last_frame[0, 0, 0]) == 7           # 缓存未被旧帧刷新


def test_maixcam2_read_raw_converts_new_frame(monkeypatch):
    """有新帧时返回 RGB->BGR 翻转后的 numpy, 并同步 last_raw。"""
    raw = object()
    seen = []

    def image2cv(img, **kw):
        seen.append((img, kw))
        arr = np.zeros((2, 2, 3), np.uint8)
        arr[:, :, 0] = 10          # image2cv 交回 RGB 序
        return arr

    mod = _load_maixcam2_camera(monkeypatch, image2cv)
    cam = object.__new__(mod.MaixCam2Camera)
    cam._last_frame = None
    cam._last_raw = None
    cam._cam = _FakeRawCam(raw)

    frame = cam.read_raw()
    assert [s[0] for s in seen] == [raw]
    assert cam.last_raw is raw
    assert frame is cam._last_frame
    assert tuple(frame[0, 0]) == (0, 0, 10)   # 通道翻转到 BGR


# ---------- maixcam1 回移植: 停帧契约与固件降级（2026-10-07） ----------

_MAIXCAM1_CAMERA_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "framework", "hal", "platforms", "maixcam1", "camera.py")


class _FakeCam1:
    """CAM1 固件桩。`block_ok=False` 模拟老固件不认 read(block=) kwarg。"""

    def __init__(self, frame, block_ok=True):
        self._frame = frame
        self._block_ok = block_ok

    def read(self, block=True):        # noqa: A002 - 对齐固件签名
        if not block and not self._block_ok:
            raise TypeError("read() got an unexpected keyword argument 'block'")
        return self._frame


def _load_maixcam1_camera(monkeypatch, image2cv):
    """按文件路径加载 maixcam1 相机模块（理由同 _load_maixcam2_camera）。"""
    import importlib.util
    import sys
    import types

    image_mod = types.ModuleType("maix.image")
    image_mod.image2cv = image2cv
    fmt = types.SimpleNamespace(FMT_RGB888=0)
    image_mod.Format = fmt
    camera_mod = types.ModuleType("maix.camera")
    camera_mod.Camera = object
    maix_mod = types.ModuleType("maix")
    maix_mod.image = image_mod
    maix_mod.camera = camera_mod
    monkeypatch.setitem(sys.modules, "maix", maix_mod)
    monkeypatch.setitem(sys.modules, "maix.camera", camera_mod)
    monkeypatch.setitem(sys.modules, "maix.image", image_mod)

    spec = importlib.util.spec_from_file_location("_maixcam1_camera_ut",
                                                  _MAIXCAM1_CAMERA_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_cam1(mod, fake_cam):
    """绕过 __init__ 造一个只设关键字段的 CAM1 相机实例。"""
    cam = object.__new__(mod.MaixCam1Camera)
    cam._last_frame = None
    cam._last_raw = None
    cam._cam = fake_cam
    cam._block_kwarg_ok = None
    return cam


def test_maixcam1_read_raw_returns_none_on_stall(monkeypatch):
    """CAM1 停帧同样必须返回 None, 不得回退 _last_frame。

    与 maixcam2 同款契约（interface/camera.py::read_raw）。CAM1 上这一条
    尤其要守: 取图延时本就30+ms，若误回退旧帧，视觉线程会把同一帧反复当
    新帧算-> 1s 陈旧归零永不触发 -> 相机挂死时冻结误差持续喂主站。
    """
    mod = _load_maixcam1_camera(
        monkeypatch, lambda raw, **kw: np.zeros((4, 4, 3), np.uint8))
    cam = _make_cam1(mod, _FakeCam1(None))
    cam._last_frame = np.full((4, 4, 3), 7, np.uint8)

    assert cam.read_raw() is None
    assert cam.last_raw is None                    # 无新帧不更新
    assert int(cam._last_frame[0, 0, 0]) == 7      # 缓存未被旧帧"刷新"


def test_maixcam1_read_raw_converts_new_frame(monkeypatch):
    """CAM1 有新帧时同样做 RGB->BGR 翻转并同步 last_raw。"""
    raw = object()
    seen = []

    def image2cv(img, **kw):
        seen.append((img, kw))
        arr = np.zeros((2, 2, 3), np.uint8)
        arr[:, :, 0] = 10
        return arr

    mod = _load_maixcam1_camera(monkeypatch, image2cv)
    cam = _make_cam1(mod, _FakeCam1(raw))

    frame = cam.read_raw()
    assert [s[0] for s in seen] == [raw]
    assert cam.last_raw is raw
    assert frame is cam._last_frame
    assert tuple(frame[0, 0]) == (0, 0, 10)      # 通道翻转到 BGR


def test_maixcam1_falls_back_when_firmware_lacks_block_kwarg(monkeypatch):
    """老固件不认 read(block=) 时降级为阻塞读，并置位降级标记。

    降级后停帧检测能力下降（相���挂死会卡在 read()），故必须打标记，
    供启动日志/台架验收确认走的是哪条分支 —— 不能静默劣化。
    """
    mod = _load_maixcam1_camera(
        monkeypatch, lambda raw, **kw: np.zeros((2, 2, 3), np.uint8))
    # 桩相机: 非阻塞调用抛 TypeError，阻塞调用返回帧
    cam = _make_cam1(mod, _FakeCam1(object(), block_ok=False))

    frame = cam.read_raw()                          # 触发降级探测
    assert cam._block_kwarg_ok is False# 标记已置位
    assert frame is not None                        # 阻塞分支仍取到帧
    assert cam._last_frame is not None


def test_maixcam1_open_degrades_to_minimal_signature(monkeypatch):
    """固件只认最小签名 Camera(w, h) 时，_open 逐级降级仍能建成相机。

    CAM1 各固件对 device=/buff_num=/open= 支持程度不一，构造不得假设版本。
    """
    mod = _load_maixcam1_camera(
        monkeypatch, lambda raw, **kw: np.zeros((2, 2, 3), np.uint8))

    calls = []

    class _OnlyMinimalCamera:
        def __init__(self, w, h):
            calls.append((w, h))

    mod.maix.camera.Camera = _OnlyMinimalCamera
    cam = mod.MaixCam1Camera(source=None, width=640, height=480,
                            camera_id="main", buff_num=2)
    assert cam.is_opened is False or cam._cam is not None
    assert (640, 480) in calls


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
