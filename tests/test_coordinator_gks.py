# -*- coding: utf-8 -*-
"""GksCoordinator tests — v2.1 订阅语义（对照 Brain interrupt.c 消费逻辑）。"""
import time
from types import SimpleNamespace

import pytest

from framework.event_bus import EventBus
from app.coordinator import GksCoordinator
from modules.zw_opencv_module.processors.base import VisionResult
from modules.zw_uart_module.events import CmdRequestEvent, CmdStopEvent
from modules.zw_uart_module.protocol import (
    build_cmd_ack_frame, build_cmd_nack_frame, build_data_stream_frame,
    build_explosive_sub_payload, build_target_sub_payload, build_rescue_sub_payload,
    DATA_EXPLOSIVE, DATA_TARGET, DATA_RESCUE,
    NACK_UNSUPPORTED_TYPE, NACK_NOT_READY,
)


def hexs(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


class FakePipe:
    def __init__(self, processor):
        self.task_manager = SimpleNamespace(
            get_task=lambda name: SimpleNamespace(processor=processor))


class FakeVM:
    def __init__(self, processor):
        self._processor = processor
        self._queued = []
        self.enabled = False
        self.pipe = FakePipe(processor)

    def queue_result(self, data: dict):
        self._queued.append(
            {"cam_main": {"gks_task": VisionResult("gks_task", data, success=True)}})

    # VisionManager API
    def drain_results(self):
        out, self._queued = self._queued, []
        return out

    def get_pipeline_fps(self, pid):
        return 60.0

    def get_pipeline(self, pid):
        return self.pipe

    def enable_task(self, pid, task):
        self.enabled = True

    def disable_task(self, pid, task):
        self.enabled = False


class FakeProc:
    """GksTaskProcessor 门控接口桩: 记录 set_task/reset, 不做识别。"""

    def __init__(self):
        self.mode = None
        self.codes = (0, 0, 0)
        self.set_task_calls = []
        self.resets = 0
        self._pos_flag = 0
        self._position = [0, 0, 0]

    def set_task(self, mode, codes=(0, 0, 0)):
        self.set_task_calls.append((mode, tuple(codes)))
        self.mode = mode
        self.codes = tuple(codes)

    def reset(self):
        self.resets += 1
        self._pos_flag = 0
        self._position = [0, 0, 0]


def make_coordinator():
    """返回 (coordinator, fake_vm, fake_processor, sent_frames)。"""
    bus = EventBus()
    coord = GksCoordinator(bus)
    proc = FakeProc()
    vm = FakeVM(proc)
    coord.connect_vision(vm)

    sent = []

    def sender(frame):
        sent.append(frame)
        return True

    coord.set_uart_sender(sender)
    coord.start()
    return coord, vm, proc, sent


def subscribe(coord, data_id, interval=50, codes=None):
    event = CmdRequestEvent(data_type=data_id, min_interval_ms=interval)
    if codes is not None:
        event.explosive_color, event.target_color, event.rescue_shape = codes
        event.has_codes = True
    coord._on_cmd_request(event)


# ---------- 订阅/ACK ----------

def test_subscribe_full_sends_ack_and_gates():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()                              # 门控在主循环应用

    assert hexs(sent[0]) == hexs(build_cmd_ack_frame(DATA_EXPLOSIVE))
    assert hexs(sent[0]) == "AA 55 06 21 01 3C 0A 13 B8"   # md §6 字面值
    assert vm.enabled
    assert proc.set_task_calls == [("EXPLOSIVE", (2, 1, 3))]


def test_resubscribe_short_keeps_codes():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    subscribe(coord, DATA_EXPLOSIVE, 50)                    # 短帧重订阅
    coord.loop()
    # 每次订阅都会调 set_task, 但同模式同码由处理器内部判重保持锁存
    assert proc.set_task_calls == [("EXPLOSIVE", (2, 1, 3)), ("EXPLOSIVE", (2, 1, 3))]
    assert proc.codes == (2, 1, 3)


def test_unknown_data_id_nacks():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, 0x10)                                  # QR 属远端
    assert hexs(sent[0]) == hexs(build_cmd_nack_frame(0x10, NACK_UNSUPPORTED_TYPE))
    assert not vm.enabled and proc.mode is None


def test_rescue_ai_switch_fail_nacks_not_ready():
    coord, vm, proc, sent = make_coordinator()
    coord.set_ai(SimpleNamespace(active_model="", switch=lambda nick: False))
    subscribe(coord, DATA_RESCUE, 50, codes=(0, 0, 3))
    assert hexs(sent[0]) == hexs(build_cmd_nack_frame(DATA_RESCUE, NACK_NOT_READY))
    assert not vm.enabled


def test_rescue_ai_ready_acks_and_switches():
    coord, vm, proc, sent = make_coordinator()
    switched = []
    ai = SimpleNamespace(active_model="",
                         switch=lambda nick: switched.append(nick) or True)
    coord.set_ai(ai)
    subscribe(coord, DATA_RESCUE, 50, codes=(0, 0, 3))
    assert switched == ["eod_hostage"]
    assert hexs(sent[0]) == hexs(build_cmd_ack_frame(DATA_RESCUE))


# ---------- 数据流 / 零值心跳 ----------

def _pump(coord):
    """置零节流后跑一轮 loop()。"""
    coord._last_send_time = 0.0
    coord.loop()


def test_stream_matches_md6_vector():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    vm.queue_result({"cx": 1234, "pit": 2, "found": True})
    coord._stream_seq = 0x2A
    _pump(coord)
    expected = build_data_stream_frame(0x2A, DATA_EXPLOSIVE,
                                       build_explosive_sub_payload(1234, 2, True))
    assert hexs(sent[-1]) == hexs(expected)
    assert hexs(expected) == "AA 55 0D 24 2A 01 D2 04 02 01 00 00 00 00 30 49"


def test_zero_heartbeat_when_no_result():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(1, 0, 0))
    _pump(coord)                          # 无任何视觉结果
    assert len(sent) == 2                 # ACK + DATA_STREAM
    stream = sent[1]
    assert stream[3] == 0x24
    assert stream[4:6] == bytes([0, DATA_EXPLOSIVE])
    assert stream[6:14] == b"\x00" * 8    # 全零子载荷: 心跳不断流
    assert stream[14:16] == stream[-2:]   # CRC 在尾部


def test_stream_payload_target_and_rescue():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_TARGET, 50, codes=(0, 2, 0))
    vm.queue_result({"err_fixed": -2500, "found": True})
    _pump(coord)
    assert sent[-1] == build_data_stream_frame(0, DATA_TARGET,
                                               build_target_sub_payload(-2500, True))

    subscribe(coord, DATA_RESCUE, 50, codes=(0, 0, 2))
    vm.queue_result({"pos": 2, "found": True})
    _pump(coord)
    assert sent[-1] == build_data_stream_frame(0, DATA_RESCUE,
                                               build_rescue_sub_payload(2, True))


def test_interval_throttles_push():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 200, codes=(1, 0, 0))
    _pump(coord)
    n_after_first = len(sent)
    coord._last_send_time = time.monotonic()   # 刚推过
    coord.loop()                               # 200ms 内: 不推
    assert len(sent) == n_after_first


def test_seq_wraps_255():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 30, codes=(1, 0, 0))
    coord._stream_seq = 0xFF
    _pump(coord)
    assert sent[-1][4] == 0xFF
    coord._last_send_time = 0.0
    coord.loop()
    assert sent[-1][4] == 0x00


# ---------- STOP / 静默 / 锁存 ----------

def test_stop_disables_and_resets():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    proc._pos_flag = 1                          # 模拟已锁存
    coord._on_cmd_stop(CmdStopEvent())
    coord.loop()                                # 门控 idle 在主循环应用
    assert not vm.enabled
    assert proc.resets == 1                     # 处理器清锁存
    _pump(coord)
    assert len(sent) == 2                       # ACK + 订阅期一帧; STOP 后不再推流


def test_master_silence_does_not_idle():
    """Brain 健康推流期间主站本就静默（看门狗只在收不到 CAM 数据时发 REQUEST）,
    从站不得按主站静默自动 idle —— 否则每 5s 周期性断流 + 清锁存。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(1, 0, 0))
    coord.loop()
    proc._pos_flag = 1
    coord._last_cmd_time = time.monotonic() - 600.0   # 主站长时间无指令
    coord._last_send_time = 0.0
    coord.loop()
    assert vm.enabled                                  # 订阅保持
    assert proc._pos_flag == 1                         # 锁存保持
    assert coord.get_info()["state"] == "EXPLOSIVE"
    assert sent[-1][3] == 0x24                         # 持续推流


def test_stale_vision_result_streams_zero():
    """相机停帧/处理器异常陈旧化后: 不透传陈旧 found=1, 回零值心跳。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(1, 0, 0))
    coord.loop()
    vm.queue_result({"cx": 1234, "pit": 2, "found": True})
    coord._last_send_time = 0.0
    coord.loop()
    assert sent[-1][6:8] == b"\xD2\x04"                # 新鲜结果正常透传
    coord._latest_task_ts = time.monotonic() - 2.0     # 结果陈旧 >1s
    coord._last_send_time = 0.0
    coord.loop()
    assert sent[-1][6:14] == b"\x00" * 8               # 归零心跳不断流


def test_latch_preserved_across_resubscribe():
    """审查 §5.3: 重复订阅帧同码重发不清 latch。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    proc._pos_flag = 1
    proc._position = [1, 2, 3]
    for _ in range(3):
        subscribe(coord, DATA_EXPLOSIVE, 50)    # Brain 1s 看门狗同参重订阅
        coord.loop()
    assert proc._pos_flag == 1                  # 锁存未被清
    assert proc._position == [1, 2, 3]


def test_data_id_change_resets():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    proc._pos_flag = 1
    subscribe(coord, DATA_TARGET, 50, codes=(0, 2, 0))   # 换 data_id
    coord.loop()
    assert proc.set_task_calls[-1] == ("TARGET", (0, 2, 0))
    assert proc.resets == 0                     # 换 id 由 set_task 清锁存（非 reset()）


def test_get_info_shape():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    vm.queue_result({"cx": 320, "pit": 2, "found": True})
    _pump(coord)
    info = coord.get_info()
    assert info["state"] == "EXPLOSIVE"
    assert info["link_active"] and info["codes"] == [2, 1, 3]
    assert info["found"] and info["pit"] == 2
