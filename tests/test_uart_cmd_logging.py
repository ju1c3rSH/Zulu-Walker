# -*- coding: utf-8 -*-
"""串口指令命令行日志测试 — 指令接收与触发动作的打印语义。

对应改动: coordinator apply_gate_task/apply_gate_idle 动作日志（锁存
kept/cleared 判重描述）+ uart_driver RX 行 payload hex 与意外帧节流。
"""
import pytest
from types import SimpleNamespace

import app.coordinator as coord_mod
import modules.zw_uart_module.uart_driver as drv_mod

from app.coordinator import GksCoordinator
from framework.event_bus import EventBus
from modules.zw_uart_module.uart_driver import STM32UartInterface
from modules.zw_uart_module.protocol import (
    build_cmd_stop_frame, DATA_EXPLOSIVE, DATA_TARGET,
)


# ---------- 测试桩 ----------

class FakeProcWithSnapshot:
    """带 current_task() 的最小门控桩, 模拟 GksTaskProcessor 判重语义。"""

    def __init__(self):
        self.mode = None
        self.codes = (0, 0, 0)
        self.resets = 0

    def set_task(self, mode, codes=(0, 0, 0)):
        if mode == self.mode and tuple(codes) == self.codes:
            return
        self.mode = mode
        self.codes = tuple(codes)

    def current_task(self):
        return self.mode, self.codes

    def reset(self):
        self.resets += 1


class FakeVM:
    def __init__(self, processor):
        self._task = SimpleNamespace(processor=processor)

    def get_pipeline(self, pid):
        return SimpleNamespace(task_manager=SimpleNamespace(
            get_task=lambda name: self._task))

    def drain_results(self):
        return []

    def get_pipeline_fps(self, pid):
        return 0.0

    def enable_task(self, pid, task):
        pass

    def disable_task(self, pid, task):
        pass


def make_coord(logs, monkeypatch):
    bus = EventBus()
    coord = GksCoordinator(bus)
    proc = FakeProcWithSnapshot()
    coord.connect_vision(FakeVM(proc))
    coord.set_uart_sender(lambda frame: True)
    monkeypatch.setattr(coord_mod, "log_print", lambda msg: logs.append(msg))
    coord.start()
    return coord, proc


def request_event(data_id, interval=50, codes=None):
    from modules.zw_uart_module.events import CmdRequestEvent
    ev = CmdRequestEvent(data_type=data_id, min_interval_ms=interval)
    if codes is not None:
        ev.explosive_color, ev.target_color, ev.rescue_shape = codes
        ev.has_codes = True
    return ev


# ---------- coordinator 动作日志 ----------

def test_gate_log_first_subscribe_latch_cleared(monkeypatch):
    logs = []
    coord, proc = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))

    assert proc.mode is None                       # 门控在主循环应用
    coord.loop()
    assert proc.mode == "EXPLOSIVE"

    gate = [m for m in logs if "[UART ACTION]" in m]
    assert len(gate) == 1
    assert "gate=EXPLOSIVE" in gate[0]
    assert "cleared" in gate[0] and "kept" not in gate[0]
    assert "任务首次启用" in gate[0]


def test_gate_log_resubscribe_latch_kept(monkeypatch):
    logs = []
    coord, proc = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))
    coord.loop()

    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, interval=50))  # 短帧重订阅
    coord.loop()

    gate = [m for m in logs if "[UART ACTION]" in m]
    assert len(gate) == 2
    assert "kept" in gate[1]
    # 短帧继承任务码: 处理器收到同码
    assert proc.codes == (2, 1, 3)


def test_gate_log_mode_change_latch_cleared(monkeypatch):
    logs = []
    coord, proc = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))
    coord.loop()
    coord._on_cmd_request(request_event(DATA_TARGET, codes=(0, 2, 0)))
    coord.loop()

    gate = [m for m in logs if "[UART ACTION]" in m]
    assert len(gate) == 2
    assert "gate=TARGET" in gate[1] and "cleared" in gate[1]
    assert "任务已启用" in gate[1]


def test_gate_log_stop_idle(monkeypatch):
    logs = []
    coord, proc = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))
    coord.loop()
    from modules.zw_uart_module.events import CmdStopEvent
    coord._on_cmd_stop(CmdStopEvent())
    coord.loop()

    gate = [m for m in logs if "[UART ACTION]" in m]
    assert len(gate) == 2
    # STOP 只停推, 锁存是否清理由下次订阅判定
    assert "gate=IDLE" in gate[1] and "kept" in gate[1]
    assert "任务已停用" in gate[1]
    assert proc.resets == 1                  # 首次订阅那次; STOP 未再清


def test_gate_log_resubscribe_after_stop_not_kept(monkeypatch):
    """STOP 把"上次订阅 id"归零: 之后的完整帧重订阅必须报 cleared, 不得报 kept。

    （旧实现靠 STOP 当场 reset() 达到同一效果, 但那会让"STOP 后短帧重订阅"也
    丢掉锁存, 与基准 main.py 不等价。）
    """
    logs = []
    coord, proc = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))
    coord.loop()
    from modules.zw_uart_module.events import CmdStopEvent
    coord._on_cmd_stop(CmdStopEvent())
    coord.loop()
    coord._on_cmd_request(request_event(DATA_EXPLOSIVE, codes=(2, 1, 3)))
    coord.loop()

    gate = [m for m in logs if "[UART ACTION]" in m]
    assert len(gate) == 3
    assert "kept" not in gate[2]
    assert "cleared" in gate[2]


def test_nack_log_has_action_desc(monkeypatch):
    logs = []
    coord, _ = make_coord(logs, monkeypatch)
    coord._on_cmd_request(request_event(0x10))     # QR 属远端

    nack = [m for m in logs if "CMD_NACK" in m]
    assert len(nack) == 1
    assert "不订阅不推流" in nack[0]


# ---------- uart_driver RX 日志 ----------

class FakeUart:
    def __init__(self):
        self.connected = True

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        return True

    def disconnect(self):
        pass

    def start_receiver(self, cb):
        self._cb = cb

    def stop_receiver(self):
        pass

    def send(self, frame):
        return len(frame)


def feed_frame(uart_iface, frame, logs, monkeypatch):
    """直接驱动 RX 回调（不经串口）, 捕获 log_print 输出。"""
    monkeypatch.setattr(drv_mod, "log_print", lambda msg: logs.append(msg))
    uart_iface._on_data_received(frame)


def build_rx_frame(frame_bytes: bytes) -> bytes:
    """AA-AA-55 length type crc16 ... 完整帧需要 CRC 头。

    用 parser 自举校验: 拼出帧再喂, 若 CRC 坏则换法构造。
    """
    from modules.zw_uart_module.protocol import crc16_ccitt
    # 帧格式见协议文档: SOF AA 55, len low, type, payload..., CRC hi lo
    length = len(frame_bytes) + 2      # payload + crc16
    body = frame_bytes
    crc = crc16_ccitt(body)
    return b"\xAA\xAA\x55" + bytes([length]) + body + bytes(
        [(crc >> 8) & 0xFF, crc & 0xFF])


def make_iface():
    iface = STM32UartInterface(FakeUart())
    return iface


def test_rx_request_log_includes_payload_hex(monkeypatch):
    logs = []
    iface = make_iface()
    # CMD_REQUEST 完整 5B 订阅帧: [data_id, interval(1B), 2,1,3]
    req = bytes([0x20, DATA_EXPLOSIVE, 50, 2, 1, 3])
    feed_frame(iface, build_rx_frame(req), logs, monkeypatch)

    rx = [m for m in logs if "[UART RX]" in m]
    assert len(rx) == 1
    assert "CMD_REQUEST" in rx[0]
    assert "EXPLOSIVE" in rx[0]
    assert "codes=[2, 1, 3]" in rx[0]
    # payload hex 为纯子载荷: [01 32 02 01 03], 不含帧头/类型/CRC
    assert "payload=0132020103" in rx[0]


def test_rx_stop_log_with_payload_hex(monkeypatch):
    logs = []
    iface = make_iface()
    feed_frame(iface, build_rx_frame(bytes([0x23])), logs, monkeypatch)

    rx = [m for m in logs if "[UART RX]" in m]
    assert len(rx) == 1
    assert "CMD_STOP" in rx[0] and "payload=23" not in rx[0]
    # hex 里只含 payload(空帧), 不含帧头/CRC
    assert "payload=\n" not in rx[0]


def test_unexpected_frames_throttled(monkeypatch):
    logs = []
    iface = make_iface()
    # DATA_STREAM 0x24: payload = [seq, data_id, 8B sub]
    ds = bytes([0x24, 0x2A, DATA_TARGET]) + b"\x00" * 8
    for _ in range(3):
        feed_frame(iface, build_rx_frame(ds), logs, monkeypatch)

    unex = [m for m in logs if "意外帧" in m]
    assert len(unex) == 1            # 首见打印, 后续节流
    assert "0x24" in unex[0] and "DATA_STREAM" in unex[0]


def test_unknown_frame_type_throttled_and_counted(monkeypatch):
    logs = []
    iface = make_iface()
    for _ in range(4):
        feed_frame(iface, build_rx_frame(bytes([0x7F])), logs, monkeypatch)

    unex = [m for m in logs if "意外帧" in m]
    assert len(unex) == 1
    assert "0x7F" in unex[0] and "Unknown" in unex[0]
    assert iface._rx_frames_unknown == 4
