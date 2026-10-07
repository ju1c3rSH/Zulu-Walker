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
    vm.queue_result({"mode": "EXPLOSIVE", "cx": 1234, "pit": 2, "found": True})
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
    vm.queue_result({"mode": "TARGET", "err_fixed": -2500, "found": True})
    _pump(coord)
    assert sent[-1] == build_data_stream_frame(0, DATA_TARGET,
                                               build_target_sub_payload(-2500, True))

    coord.set_ai(SimpleNamespace(active_model="eod_hostage",
                                 switch=lambda nick: True))
    subscribe(coord, DATA_RESCUE, 50, codes=(0, 0, 2))
    vm.queue_result({"mode": "RESCUE", "pos": 2, "found": True})
    _pump(coord)
    assert sent[-1] == build_data_stream_frame(0, DATA_RESCUE,
                                               build_rescue_sub_payload(2, True))


def test_heartbeat_throttled_by_interval():
    """无新结果时按 interval 推心跳（interval = 心跳周期, 不是每帧最小间隔）。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 200, codes=(1, 0, 0))
    _pump(coord)
    n_after_first = len(sent)
    coord._last_send_time = time.monotonic()   # 刚推过
    coord.loop()                               # 200ms 内 + 无新结果: 不推
    assert len(sent) == n_after_first
    coord._last_send_time = 0.0
    coord.loop()                               # 到点 -> 心跳
    assert len(sent) == n_after_first + 1


# ---------- 结果驱动推流（"有一帧误差就实时输出"） ----------

def test_new_result_pushes_immediately_within_interval():
    """有一帧新结果就立即推一帧, **不受 interval 限制**。

    旧实现是纯时间闸门: Brain 下发 50ms 而相机 60fps 时, 约 2/3 帧的误差
    根本没发（不是延迟, 是丢失）。这里把 interval 拉到 1s 都无法阻止新结果推送,
    即证明闸门已由"时间"改为"结果"。
    """
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_TARGET, 1000, codes=(0, 2, 0))   # 心跳周期 1s
    _pump(coord)
    n = len(sent)

    coord._last_send_time = time.monotonic()               # 刚推过, 离 1s 还远
    vm.queue_result({"mode": "TARGET", "err_fixed": -2500, "found": True})
    coord.loop()
    assert len(sent) == n + 1
    assert sent[-1][6:14] == build_target_sub_payload(-2500, True)


def test_no_new_result_does_not_repush():
    """没有新结果时不得重复推同一帧（旧实现每 interval 就重发一次）。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_TARGET, 1000, codes=(0, 2, 0))
    _pump(coord)
    n = len(sent)

    coord._last_send_time = time.monotonic()
    for _ in range(5):
        coord.loop()                                       # 无新结果
    assert len(sent) == n


def test_batched_results_collapse_to_latest_frame():
    """一次主循环迭代内到达多帧结果时只推一帧, 且是**最新**那帧。

    推流侧不排队: 主循环被拖慢时宁可漏掉中间帧, 也不送陈旧误差 —— 载荷永远
    取 _latest_task 快照, 与 Brain 的控制环只关心"当前误差"一致。
    """
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_TARGET, 1000, codes=(0, 2, 0))
    _pump(coord)
    n = len(sent)

    coord._last_send_time = time.monotonic()
    vm.queue_result({"mode": "TARGET", "err_fixed": 100, "found": True})
    vm.queue_result({"mode": "TARGET", "err_fixed": 200, "found": True})
    vm.queue_result({"mode": "TARGET", "err_fixed": 300, "found": True})
    coord.loop()
    assert len(sent) == n + 1
    assert sent[-1][6:14] == build_target_sub_payload(300, True)   # 只推最新


def test_min_push_interval_floor_limits_burst():
    """min_push_interval_ms 硬保护: 相机恢复瞬间的突发连推被压住（新结果也压）。"""
    coord, vm, proc, sent = make_coordinator()
    coord.set_stream_policy(min_push_interval_ms=200)
    subscribe(coord, DATA_TARGET, 1000, codes=(0, 2, 0))
    _pump(coord)
    n = len(sent)

    coord._last_send_time = time.monotonic()
    for err in (100, 200, 300):
        vm.queue_result({"mode": "TARGET", "err_fixed": err, "found": True})
        coord.loop()
    assert len(sent) == n                                  # floor 内一帧都不推

    coord._last_send_time = time.monotonic() - 0.3         # 超过 floor
    vm.queue_result({"mode": "TARGET", "err_fixed": 400, "found": True})
    coord.loop()
    assert len(sent) == n + 1
    assert sent[-1][6:14] == build_target_sub_payload(400, True)


def test_policy_can_fall_back_to_interval_throttle():
    """push_on_new_result=False -> 回到旧的纯 interval 限速（现场退路）。"""
    coord, vm, proc, sent = make_coordinator()
    coord.set_stream_policy(push_on_new_result=False)
    subscribe(coord, DATA_TARGET, 1000, codes=(0, 2, 0))
    _pump(coord)
    n = len(sent)

    coord._last_send_time = time.monotonic()
    vm.queue_result({"mode": "TARGET", "err_fixed": 100, "found": True})
    coord.loop()
    assert len(sent) == n                                  # 新结果也等 interval


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

def test_stop_disables_and_keeps_latch():
    """STOP 只停推, **不清锁存**(对齐 main.py handle_stop 只置 sub_id=0/seq=0)。

    锁存是否作废交给下次订阅判定: STOP 把"上次订阅 id"归零, 于是 STOP 后的
    完整帧重订阅必然命中"data_id 变了"而清 —— 时机与基准一致。
    """
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    proc._pos_flag = 1                          # 模拟已锁存
    before = proc.resets                        # 首次订阅已清过一次(与基准同)
    coord._on_cmd_stop(CmdStopEvent())
    coord.loop()                                # 门控 idle 在主循环应用
    assert not vm.enabled
    assert proc.resets == before                # STOP 当场不清锁存
    assert proc._pos_flag == 1                  # 锁存仍在
    _pump(coord)
    assert len(sent) == 2                       # ACK + 订阅期一帧; STOP 后不再推流

    # STOP 后的完整帧重订阅: 与归零后的基准比较, data_id 不同 -> 清锁存
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    assert proc.resets == before + 1


def test_stop_then_short_resubscribe_keeps_latch():
    """STOP 后短帧重订阅(无任务码)沿用旧码, 不清锁存。"""
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    coord._on_cmd_stop(CmdStopEvent())
    coord.loop()
    before = proc.resets
    subscribe(coord, DATA_EXPLOSIVE, 50)        # 短帧重订阅
    coord.loop()
    assert proc.resets == before                # 短帧不清锁存
    assert proc.codes == (2, 1, 3)              # 沿用旧任务码


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
    vm.queue_result({"mode": "EXPLOSIVE", "cx": 1234, "pit": 2, "found": True})
    coord._last_send_time = 0.0
    coord.loop()
    assert sent[-1][6:8] == b"\xD2\x04"                # 新鲜结果正常透传
    coord._latest_task_ts = time.monotonic() - 2.0     # 结果陈旧 >1s
    coord._last_send_time = 0.0
    coord.loop()
    assert sent[-1][6:14] == b"\x00" * 8               # 归零心跳不断流


def test_cross_mode_residual_frame_never_streamed():
    """切任务后的异步残留帧不得按新 data_id 打包推出。

    视觉结果由后台线程异步产出, 而 RX 线程在收到订阅帧那一刻就改了 _sub_id:
    切到 TARGET 后, 上一模式"正在处理的那一帧"仍带 mode=EXPLOSIVE, 且会在模式
    切换之后才落进结果缓存。若按 TARGET 打包, 就会推出 err=0 且 found=1 ——
    底盘据此判"已收敛"并刹停。归属校验必须把它拦成零值心跳。
    """
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    coord.loop()
    subscribe(coord, DATA_TARGET, 50, codes=(0, 2, 0))   # 切到打靶
    coord.loop()

    # 残留帧: 旧模式的 found=1 结果, 在切换之后才到达
    vm.queue_result({"mode": "EXPLOSIVE", "cx": 1234, "pit": 2, "found": True})
    _pump(coord)
    assert sent[-1][3] == 0x24
    assert sent[-1][6:14] == b"\x00" * 8                # 未采纳: 零值心跳(不断流)

    # 新模式自己的结果照常透传
    vm.queue_result({"mode": "TARGET", "err_fixed": -2500, "found": True})
    _pump(coord)
    assert sent[-1] == build_data_stream_frame(sent[-1][4], DATA_TARGET,
                                               build_target_sub_payload(-2500, True))


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
    before = proc.resets
    subscribe(coord, DATA_TARGET, 50, codes=(0, 2, 0))   # 换 data_id
    coord.loop()
    assert proc.set_task_calls[-1] == ("TARGET", (0, 2, 0))
    assert proc.resets == before + 1            # 换 data_id: 由统一判据显式清锁存


def test_get_info_shape():
    coord, vm, proc, sent = make_coordinator()
    subscribe(coord, DATA_EXPLOSIVE, 50, codes=(2, 1, 3))
    vm.queue_result({"mode": "EXPLOSIVE", "cx": 320, "pit": 2, "found": True})
    _pump(coord)
    info = coord.get_info()
    assert info["state"] == "EXPLOSIVE"
    assert info["link_active"] and info["codes"] == [2, 1, 3]
    assert info["found"] and info["pit"] == 2
