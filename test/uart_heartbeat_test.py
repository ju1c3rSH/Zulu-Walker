#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob as glob_mod
import logging
import os
import platform as _platform
import random
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.interface import Uart
from modules.zw_uart_module.protocol import (
    SOF,
    TYPE_HEARTBEAT,
    TYPE_ARRIVED, TYPE_PICK, TYPE_SET, TYPE_CMD_FROM_MCU,
    TYPE_ACTION_DONE, TYPE_REQUEST_SYNC, TYPE_EMERGENCY_STOP,
    TYPE_STATUS_FROM_VISION, TYPE_QR_RESULT, TYPE_COLOR_RESULT,
    TYPE_VISUAL_SERVO_DATA, TYPE_ERROR,
    build_heartbeat_frame,
    parse_heartbeat_payload,
    parse_zone_payload, parse_cmd_payload, parse_action_done_payload,
    parse_emergency_stop_payload, parse_request_sync_payload,
    parse_frame,
)
from modules.zw_uart_module.uart_driver import FrameParser, ParserState

from rich.align import Align
from rich.box import ROUNDED, HEAVY_EDGE
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


_TX_STYLE = Style(color="cyan", bold=True)
_RX_STYLE = Style(color="green", bold=True)
_RX_OK_STYLE = Style(color="green")
_LOSS_STYLE = Style(color="red", bold=True)
_WARN_STYLE = Style(color="yellow")
_DIM_STYLE = Style(dim=True)
_OK_STYLE = Style(color="green", bold=True)
_ERR_STYLE = Style(color="red", bold=True)
_HEADER_STYLE = Style(bold=True)
_TITLE_STYLE = Style(color="bright_white", bold=True)

_FRAME_LABEL = {
    TYPE_ARRIVED: "ARRIVED",
    TYPE_PICK: "PICK",
    TYPE_SET: "SET",
    TYPE_CMD_FROM_MCU: "CMD",
    TYPE_ACTION_DONE: "ACTION_DONE",
    TYPE_HEARTBEAT: "HEARTBEAT",
    TYPE_REQUEST_SYNC: "REQ_SYNC",
    TYPE_EMERGENCY_STOP: "EM_STOP",
    TYPE_STATUS_FROM_VISION: "STATUS",
    TYPE_QR_RESULT: "QR",
    TYPE_COLOR_RESULT: "COLOR",
    TYPE_ERROR: "ERROR",
    TYPE_VISUAL_SERVO_DATA: "SERVO_DATA",
}


def _beep_alert():
    if _platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(800, 300)
            return
        except Exception:
            pass
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _detect_port() -> str:
    if _platform.system() == "Windows":
        for i in range(1, 256):
            cand = f"COM{i}"
            try:
                import serial
                s = serial.Serial(cand, 921600, timeout=0.1)
                s.close()
                return cand
            except Exception:
                continue
        return "COM1"
    candidates = []
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyS*"):
        candidates.extend(sorted(glob_mod.glob(pat)))
    if candidates:
        return candidates[0]
    return "/dev/ttyS4"


class LoopbackUart:
    def __init__(self, loss_rate: float = 0.0, delay: float = 0.001):
        self._loss_rate = loss_rate
        self._delay = delay
        self._connected = False
        self._running = False
        self._rx_buffer = bytearray()
        self._tx_buffer = bytearray()
        self._callback = None
        self._echo_thread = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger("LoopbackUart")

    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._rx_buffer)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        self._logger.info("LoopbackUart connected")
        return True

    def disconnect(self) -> None:
        self.stop_receiver()
        self._connected = False
        with self._lock:
            self._rx_buffer.clear()
            self._tx_buffer.clear()
        self._logger.info("LoopbackUart disconnected")

    def send(self, data: bytes) -> int:
        with self._lock:
            self._tx_buffer.extend(data)
        return len(data)

    def receive(self, size: int = 1) -> Optional[bytes]:
        with self._lock:
            if not self._rx_buffer:
                return None
            chunk = bytes(self._rx_buffer[:size])
            self._rx_buffer = self._rx_buffer[size:]
            return chunk

    def receive_all(self) -> Optional[bytes]:
        with self._lock:
            if not self._rx_buffer:
                return None
            chunk = bytes(self._rx_buffer)
            self._rx_buffer.clear()
            return chunk

    def start_receiver(self, callback) -> None:
        self._callback = callback
        if self._echo_thread and self._echo_thread.is_alive():
            return
        self._running = True
        self._echo_thread = threading.Thread(target=self._echo_loop, daemon=True)
        self._echo_thread.start()

    def _echo_loop(self) -> None:
        while self._running:
            data = None
            with self._lock:
                if self._tx_buffer:
                    data = bytes(self._tx_buffer)
                    self._tx_buffer.clear()
            if data:
                time.sleep(self._delay)
                if random.random() >= self._loss_rate:
                    with self._lock:
                        self._rx_buffer.extend(data)
                    if self._callback:
                        try:
                            self._callback(data)
                        except Exception as e:
                            self._logger.error("callback error: %s", e)
            time.sleep(0.001)

    def stop_receiver(self) -> None:
        self._running = False
        if self._echo_thread and self._echo_thread.is_alive():
            self._echo_thread.join(timeout=1)


class DiagnosticFrameParser(FrameParser):
    def __init__(self):
        super().__init__()
        self.checksum_errors = 0
        self.format_errors = 0

    def _parse_complete_frame(self):
        frame = parse_frame(bytes(self._buffer))
        if frame is None:
            self.checksum_errors += 1
            return None
        return frame

    def _process_byte(self, byte: int):
        if self._state == ParserState.WAITING_SOF:
            if byte == SOF:
                self._buffer = bytearray([SOF])
                self._state = ParserState.GOT_SOF
            return None
        elif self._state == ParserState.GOT_SOF:
            if 2 <= byte <= 253:
                self._expected_length = byte
                self._buffer.append(byte)
                self._state = ParserState.GOT_LEN
            else:
                self.format_errors += 1
                self.reset()
            return None
        elif self._state == ParserState.GOT_LEN:
            self._buffer.append(byte)
            expected_size = 2 + self._expected_length
            if len(self._buffer) == expected_size:
                frame = self._parse_complete_frame()
                self.reset()
                return frame
            return None
        return None


class HeartbeatTestController:
    def __init__(self, uart: Uart, seq_start: int = 0):
        self._uart = uart
        self._parser = DiagnosticFrameParser()

        self._lock = threading.RLock()
        self._running = False
        self._seq = seq_start
        self._tx_total = 0
        self._rx_total = 0
        self._last_rx_seq = -1
        self._expected_rx_seq = 0
        self._lost_total = 0
        self._consecutive_loss = 0
        self._max_consecutive_loss = 0
        self._last_rx_time = 0.0
        self._start_time = 0.0

        self._tx_log: deque[tuple[float, int]] = deque(maxlen=10000)
        self._rx_log: deque[tuple[float, int]] = deque(maxlen=10000)

        self._rx_bytes_total = 0
        self._rx_other_frames = 0
        self._last_callback_time = 0.0

    @property
    def is_connected(self) -> bool:
        return self._uart.is_connected

    @property
    def uptime(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.monotonic() - self._start_time

    def start(self) -> bool:
        try:
            if not self._uart.connect():
                return False
            self._uart.start_receiver(self._on_data_received)
            self._running = True
            self._start_time = time.monotonic()
            return True
        except Exception as e:
            logging.getLogger("HeartbeatTestController").error("start failed: %s", e)
            return False

    def stop(self) -> None:
        self._running = False
        try:
            self._uart.stop_receiver()
            self._uart.disconnect()
        except Exception as e:
            logging.getLogger("HeartbeatTestController").error("stop error: %s", e)

    def _on_data_received(self, data: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            self._rx_bytes_total += len(data)
            self._last_callback_time = now

        frames = self._parser.feed(data)
        for frame in frames:
            if frame.frame_type == TYPE_HEARTBEAT:
                parsed = parse_heartbeat_payload(frame.payload)
                if parsed is not None:
                    seq = parsed[0]
                    with self._lock:
                        self._rx_total += 1
                        self._rx_log.append((now, seq))
                        self._last_rx_time = now
                        if self._last_rx_seq >= 0:
                            if seq == self._expected_rx_seq:
                                self._consecutive_loss = 0
                            else:
                                gap = (seq - self._expected_rx_seq) % 256
                                self._lost_total += gap
                                self._consecutive_loss += gap
                                if self._consecutive_loss > self._max_consecutive_loss:
                                    self._max_consecutive_loss = self._consecutive_loss
                        else:
                            self._consecutive_loss = 0
                        self._last_rx_seq = seq
                        self._expected_rx_seq = (seq + 1) % 256
                continue

            with self._lock:
                self._rx_other_frames += 1

    def send_heartbeat(self, mission_state: int = 0, visual_state: int = 0) -> bool:
        now = time.monotonic()
        with self._lock:
            seq = self._seq
            self._seq = (self._seq + 1) % 256
            self._tx_total += 1
        frame = build_heartbeat_frame(seq, mission_state, visual_state)
        ok = self._uart.send(frame) == len(frame)
        if ok:
            with self._lock:
                self._tx_log.append((now, seq))
        return ok

    def get_stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            cutoff = now - 30.0

            tx_win = [(t, s) for t, s in self._tx_log if t >= cutoff]
            rx_win = [(t, s) for t, s in self._rx_log if t >= cutoff]
            tx_win_count = len(tx_win)
            rx_win_count = len(rx_win)

            rx_win_seqs = [s for _, s in rx_win]
            lost_win = 0
            max_consec_win = 0
            consec_win = 0
            if rx_win_seqs:
                sorted_seqs = sorted(rx_win_seqs)
                exp_seq = sorted_seqs[0]
                for s in sorted_seqs:
                    if s == exp_seq:
                        consec_win = 0
                    else:
                        gap = (s - exp_seq) % 256
                        lost_win += gap
                        consec_win += gap
                        if consec_win > max_consec_win:
                            max_consec_win = consec_win
                    exp_seq = (s + 1) % 256

            diff_seq = (self._seq - 1 - self._last_rx_seq) % 256 if self._last_rx_seq >= 0 else 0
            if diff_seq > 128:
                diff_seq = 256 - diff_seq

            last_tx_seq_win = tx_win[-1][1] if tx_win else -1
            last_rx_seq_win = rx_win[-1][1] if rx_win else -1
            diff_seq_win = -1
            if last_tx_seq_win >= 0 and last_rx_seq_win >= 0:
                d = (last_tx_seq_win - last_rx_seq_win) % 256
                diff_seq_win = d if d <= 128 else 256 - d

            elapsed = now - self._last_rx_time if self._last_rx_time > 0 else -1.0

            loss_rate_all = (self._lost_total / max(self._tx_total, 1)) * 100
            loss_rate_win = (lost_win / max(tx_win_count, 1)) * 100

            return {
                "tx_total": self._tx_total,
                "rx_total": self._rx_total,
                "lost_total": self._lost_total,
                "tx_win": tx_win_count,
                "rx_win": rx_win_count,
                "lost_win": lost_win,
                "loss_rate_all": loss_rate_all,
                "loss_rate_win": loss_rate_win,
                "diff_seq": diff_seq,
                "diff_seq_win": diff_seq_win,
                "last_rx_seq": self._last_rx_seq,
                "last_tx_seq": (self._seq - 1) % 256,
                "max_consecutive_loss": self._max_consecutive_loss,
                "max_consecutive_loss_win": max_consec_win,
                "consecutive_loss": self._consecutive_loss,
                "elapsed_since_rx": elapsed,
                "tx_log_raw": list(self._tx_log),
                "rx_log_raw": list(self._rx_log),
                "uptime": self.uptime,
                "rx_bytes_total": self._rx_bytes_total,
                "rx_checksum_errors": self._parser.checksum_errors,
                "rx_format_errors": self._parser.format_errors,
                "rx_other_frames": self._rx_other_frames,
                "last_callback_time": self._last_callback_time,
            }


def _build_flow_diagram(stats: dict) -> Panel:
    tx_seq = stats["last_tx_seq"]
    rx_seq = stats["last_rx_seq"]
    consec = stats["consecutive_loss"]
    elapsed = stats["elapsed_since_rx"]
    loss_rate = stats["loss_rate_win"]
    link_ok = loss_rate < 10 and elapsed < 0.5

    lines = []

    t = Text()
    t.append("    ", Style())
    t.append("\u250c", Style(color="bright_black"))
    t.append("\u2500" * 10, Style(color="bright_black"))
    t.append("\u2510", Style(color="bright_black"))
    t.append("      ", Style())
    t.append("\u250c", Style(color="bright_black"))
    t.append("\u2500" * 10, Style(color="bright_black"))
    t.append("\u2510", Style(color="bright_black"))
    lines.append(t)

    t = Text()
    t.append("    \u2502", Style(color="bright_black"))
    t.append("    OP    ", _HEADER_STYLE)
    t.append("\u2502", Style(color="bright_black"))
    t.append("  TX ", _TX_STYLE)
    if tx_seq >= 0:
        t.append(f"seq={tx_seq}", _TX_STYLE)
    else:
        t.append("---", _DIM_STYLE)
    t.append("  ", Style())
    t.append("\u2502", Style(color="bright_black"))
    t.append("  STM32  ", _HEADER_STYLE)
    t.append("\u2502", Style(color="bright_black"))
    lines.append(t)

    t = Text()
    t.append("    \u2502", Style(color="bright_black"))
    t.append("  (\u672c\u673a) ", _DIM_STYLE)
    t.append("\u2502", Style(color="bright_black"))
    t.append("     \u25c0", _LOSS_STYLE if stats["lost_total"] > 0 else _RX_OK_STYLE)
    t.append("RX ", _RX_STYLE)
    if rx_seq >= 0:
        t.append(f"seq={rx_seq}", _RX_STYLE)
    else:
        t.append("---", _DIM_STYLE)
    t.append("  ", Style())
    t.append("\u2502", Style(color="bright_black"))
    t.append("(\u8fdc\u7aef) ", _DIM_STYLE)
    t.append("\u2502", Style(color="bright_black"))
    lines.append(t)

    t = Text()
    t.append("    \u2514", Style(color="bright_black"))
    t.append("\u2500" * 10, Style(color="bright_black"))
    t.append("\u2518", Style(color="bright_black"))
    t.append("      ", Style())
    t.append("\u2514", Style(color="bright_black"))
    t.append("\u2500" * 10, Style(color="bright_black"))
    t.append("\u2518", Style(color="bright_black"))
    lines.append(t)

    lines.append(Text())

    loss_info = Text()
    loss_info.append("    \u2716 ", _LOSS_STYLE if stats["lost_total"] > 0 else _RX_OK_STYLE)
    if stats["lost_total"] > 0:
        loss_info.append(
            f"\u4e22\u5305: seq gap = +{stats['lost_total']}  ",
            _LOSS_STYLE,
        )
    else:
        loss_info.append("\u65e0\u4e22\u5305  ", _RX_OK_STYLE)
    if consec > 0:
        loss_info.append(
            f"\u5f53\u524d\u8fde\u7eed\u4e22\u5305: {consec}",
            _WARN_STYLE if consec < 5 else _LOSS_STYLE,
        )
    if elapsed > 0:
        loss_info.append(
            f"  \u4e0a\u6b21\u63a5\u6536: {elapsed*1000:.1f}ms\u524d",
            _WARN_STYLE if elapsed > 0.5 else _RX_OK_STYLE,
        )
    lines.append(loss_info)

    link_text = Text()
    link_text.append("    ")
    link_text.append("\u25cf LINK ", _OK_STYLE if link_ok else _ERR_STYLE)
    link_text.append("ACTIVE" if link_ok else "DOWN", _OK_STYLE if link_ok else _ERR_STYLE)
    chk = stats["rx_checksum_errors"]
    if chk > 0:
        link_text.append(f"  \u6821\u9a8c\u9519={chk}", _LOSS_STYLE)
    link_text.append(
        f"  \u5b57\u8282={stats['rx_bytes_total']:,d}",
        _DIM_STYLE,
    )
    lines.append(link_text)

    return Panel(Group(*lines), title="Data Flow", border_style=_DIM_STYLE, box=ROUNDED)


def _build_stats_table(stats: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=_HEADER_STYLE)
    table.add_column(justify="right")
    table.add_column(justify="right")
    table.add_column(justify="center", width=6)

    tx_total = f"{stats['tx_total']:,d}"
    rx_total = f"{stats['rx_total']:,d}"
    tx_win = f"{stats['tx_win']:,d}"
    rx_win = f"{stats['rx_win']:,d}"

    lr_all = stats["loss_rate_all"]
    lr_win = stats["loss_rate_win"]
    lr_all_s = f"{lr_all:.2f}%"
    lr_win_s = f"{lr_win:.2f}%"

    trend = Text("\u2191", _LOSS_STYLE) if lr_win > 5 else Text("\u2192", _RX_OK_STYLE)

    table.add_row(
        Text("\u5355\u9879", _DIM_STYLE),
        Text("\u7d2f\u8ba1\u503c", _DIM_STYLE),
        Text("30s\u7a97\u53e3", _DIM_STYLE),
        Text("\u8d8b\u52bf", _DIM_STYLE),
    )
    table.add_row(
        Text("\u53d1\u9001\u5fc3\u8df3", _HEADER_STYLE),
        Text(tx_total, _TX_STYLE),
        Text(tx_win, _TX_STYLE),
        Text(""),
    )
    table.add_row(
        Text("\u63a5\u6536\u5fc3\u8df3", _HEADER_STYLE),
        Text(rx_total, _RX_STYLE),
        Text(rx_win, _RX_STYLE),
        Text(""),
    )
    table.add_row(
        Text("\u4e22\u5305\u7387", _HEADER_STYLE),
        Text(lr_all_s, _LOSS_STYLE if lr_all > 1 else _RX_OK_STYLE),
        Text(lr_win_s, _LOSS_STYLE if lr_win > 1 else _RX_OK_STYLE),
        trend,
    )
    table.add_row(
        Text("\u5e8f\u53f7\u5dee(TX-RX)", _HEADER_STYLE),
        Text(
            f"{stats['diff_seq']}",
            _WARN_STYLE if stats['diff_seq'] > 0 else _RX_OK_STYLE,
        ),
        Text(
            f"{stats['diff_seq_win']}" if stats['diff_seq_win'] >= 0 else "-",
            _WARN_STYLE if (stats['diff_seq_win'] or 0) > 0 else _RX_OK_STYLE,
        ),
        Text(""),
    )

    elapsed = stats["elapsed_since_rx"]
    elapsed_s = f"{elapsed*1000:.1f}ms" if elapsed >= 0 else "-"
    elapsed_style = _LOSS_STYLE if elapsed > 0.5 else (_WARN_STYLE if elapsed > 0.1 else _RX_OK_STYLE)

    max_lose = stats["max_consecutive_loss"]
    max_lose_win = stats["max_consecutive_loss_win"]
    table.add_row(
        Text("\u4e0a\u6b21\u63a5\u6536", _HEADER_STYLE),
        Text(elapsed_s, elapsed_style),
        Text(""),
        Text(""),
    )
    table.add_row(
        Text("\u6700\u5927\u8fde\u7eed\u4e22\u5305", _HEADER_STYLE),
        Text(f"{max_lose}", _LOSS_STYLE if max_lose > 2 else _RX_OK_STYLE),
        Text(f"{max_lose_win}", _LOSS_STYLE if max_lose_win > 2 else _RX_OK_STYLE),
        Text(""),
    )

    chk = stats["rx_checksum_errors"]
    fmt = stats["rx_format_errors"]
    oth = stats["rx_other_frames"]
    byt = stats["rx_bytes_total"]
    cbk = stats["last_callback_time"]
    now_mono = time.monotonic()
    cbk_delta = now_mono - cbk if cbk > 0 else -1.0

    table.add_row(
        Text("\u6821\u9a8c\u9519/XOR", _HEADER_STYLE),
        Text(f"{chk:,d}", _LOSS_STYLE if chk > 0 else _RX_OK_STYLE),
        Text(""),
        Text(""),
    )
    table.add_row(
        Text("\u683c\u5f0f\u9519/Frame", _HEADER_STYLE),
        Text(f"{fmt:,d}", _LOSS_STYLE if fmt > 0 else _RX_OK_STYLE),
        Text(""),
        Text(""),
    )
    table.add_row(
        Text("\u5176\u4ed6\u5e27", _HEADER_STYLE),
        Text(f"{oth:,d}", _RX_STYLE),
        Text(""),
        Text(""),
    )
    table.add_row(
        Text("\u63a5\u6536\u5b57\u8282", _HEADER_STYLE),
        Text(f"{byt:,d}", _DIM_STYLE),
        Text(""),
        Text(""),
    )
    rx_health = Text()
    if cbk_delta >= 0:
        rx_health.append(f"{cbk_delta*1000:.0f}ms", _LOSS_STYLE if cbk_delta > 0.5 else _RX_OK_STYLE)
        rx_health.append(" \u524d\u56de\u8c03", _DIM_STYLE)
    else:
        rx_health.append("\u672a\u63a5\u6536", _DIM_STYLE)
    table.add_row(
        Text("\u63a5\u6536\u5065\u5eb7", _HEADER_STYLE),
        rx_health,
        Text(""),
        Text(""),
    )

    tx_total_int = stats["tx_total"]
    rx_total_int = stats["rx_total"]
    status_text = "\u2705 \u8fd0\u884c\u4e2d"
    if chk > 0 or fmt > 0:
        status_text = "\u26a0 \u7ebf\u8def\u5f02\u5e38"
        status_style = _LOSS_STYLE
    elif tx_total_int > 0 and rx_total_int == 0:
        status_text = "\u26a0 \u672a\u63a5\u6536"
        status_style = _WARN_STYLE
    elif stats["consecutive_loss"] > 10:
        status_text = "\u274c \u8fde\u7eed\u4e22\u5305"
        status_style = _LOSS_STYLE
    elif stats["loss_rate_all"] > 5:
        status_text = "\u26a0 \u4e22\u5305\u8fc7\u9ad8"
        status_style = _WARN_STYLE
    else:
        status_style = _RX_OK_STYLE
    table.add_row(
        Text("\u72b6\u6001", _HEADER_STYLE),
        Text(status_text, status_style),
        Text(""),
        Text(""),
    )

    return Panel(
        Align.center(table),
        title="Statistics",
        border_style=_DIM_STYLE,
        box=ROUNDED,
    )


def _build_timeline_chart(stats: dict) -> Panel:
    now = time.monotonic()
    tx_raw = stats["tx_log_raw"]
    rx_raw = stats["rx_log_raw"]

    def _render(label: str, events: list, span: float, slots: int, style: Style) -> Text:
        start = now - span
        bins = [False] * slots
        for t, _ in events:
            if t >= start:
                idx = int((t - start) / span * slots)
                if idx >= slots:
                    idx = slots - 1
                elif idx < 0:
                    idx = 0
                bins[idx] = True
        line = "".join("\u2588" if b else "\u2591" for b in bins)
        t = Text()
        t.append(f" {label} ", _DIM_STYLE)
        t.append(line, style)
        return t

    lines = []

    t = Text()
    t.append(" 30s (1s/\u6829):", _DIM_STYLE)
    lines.append(t)
    lines.append(_render("TX", tx_raw, 30.0, 30, _TX_STYLE))
    lines.append(_render("RX", rx_raw, 30.0, 30, _RX_STYLE))

    lines.append(Text())
    t = Text()
    t.append(" 5s  (100ms/\u6829):", _DIM_STYLE)
    lines.append(t)
    lines.append(_render("TX", tx_raw, 5.0, 50, _TX_STYLE))
    lines.append(_render("RX", rx_raw, 5.0, 50, _RX_STYLE))

    legend = Text()
    legend.append(" \u2588 = \u4e8b\u4ef6  ", _RX_OK_STYLE)
    legend.append("\u2591 = \u65e0\u4e8b\u4ef6", _DIM_STYLE)
    lines.append(legend)

    return Panel(Group(*lines), title="Timeline", border_style=_DIM_STYLE, box=ROUNDED)


def _build_renderable(stats: dict) -> Panel:
    header_left = Text()
    header_left.append("\u26a1 UART Heartbeat Test", _TITLE_STYLE)
    header_left.append(f"  \u8fd0\u884c {stats['uptime']:.0f}s", _DIM_STYLE)

    connected = stats.get("connected", False)
    status_color = _OK_STYLE if connected else _ERR_STYLE
    status_dot = "\u25cf" if connected else "\u25cb"
    header_right = Text()
    header_right.append(f" {status_dot} ", status_color)
    header_right.append("CONNECTED" if connected else "DISCONNECTED", status_color)

    header = Table.grid(padding=(0, 2))
    header.add_column()
    header.add_column(justify="right")
    header.add_row(header_left, header_right)

    main = Table.grid(padding=(0, 1))
    main.add_column(ratio=1)
    main.add_column(ratio=1)

    flow = _build_flow_diagram(stats)
    timeline = _build_timeline_chart(stats)
    stats_panel = _build_stats_table(stats)

    body = Layout()
    body.split_column(
        Layout(header, size=1),
        Layout(
            Columns([flow, stats_panel], equal=True, expand=True),
            size=12,
        ),
        Layout(timeline, size=11),
    )

    outer = Panel(body, box=HEAVY_EDGE, border_style="bright_black")
    return outer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UART Heartbeat Test - \u53cc\u5411\u4e32\u53e3\u5fc3\u8df3\u6d4b\u8bd5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", default=None, help="\u4e32\u53e3 (default: auto-detect)")
    p.add_argument("--baud", type=int, default=921600, help="\u6ce2\u7279\u7387 (default: 921600)")
    p.add_argument("--mock", action="store_true", help="Mock\u6a21\u5f0f (\u65e0\u9700\u786c\u4ef6)")
    p.add_argument("--loss", type=float, default=0.0, help="Mock\u4e22\u5305\u7387 0.0-1.0")
    p.add_argument("--delay", type=float, default=0.001, help="Mock\u6a21\u62df\u5ef6\u8fdf\u79d2")
    p.add_argument("--seq-start", type=int, default=0, help="\u8d77\u59cb\u5e8f\u53f7 (default: 0)")
    p.add_argument("--auto", action="store_true", help="\u81ea\u52a8\u626b\u63cf\u53ef\u7528\u4e32\u53e3")
    p.add_argument("--interval", type=float, default=0.1, help="\u53d1\u9001\u95f4\u9694\u79d2 (default: 0.1)")
    p.add_argument("--no-beep", action="store_true", help="\u7981\u7528\u8702\u9e23\u62a5\u8b66")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.auto:
        port = _detect_port()
        sys.stdout.write(f"Auto-detected port: {port}\n")
    elif args.mock:
        port = "mock"
    elif args.port:
        port = args.port
    else:
        port = _detect_port()

    if args.mock:
        uart: Uart = LoopbackUart(loss_rate=args.loss, delay=args.delay)
        sys.stdout.write(
            f"Mock mode: loss={args.loss:.0%} delay={args.delay*1000:.1f}ms\n"
        )
    else:
        from hal.platforms.linux import LinuxUart
        uart = LinuxUart(port=port, baudrate=args.baud)
        sys.stdout.write(f"Port: {port} @ {args.baud} baud\n")

    controller = HeartbeatTestController(uart, seq_start=args.seq_start)
    if not controller.start():
        sys.stderr.write("Failed to start controller\n")
        return 1

    console = Console()

    beep_enabled = not args.no_beep
    send_interval = args.interval
    running = True
    next_ui = 0.0
    _prev_loss = 0

    stop_send = threading.Event()

    def _send_loop():
        next_send = time.monotonic()
        while not stop_send.is_set():
            now = time.monotonic()
            if now >= next_send:
                if not controller.send_heartbeat():
                    pass
                next_send += send_interval
                if next_send <= now:
                    next_send = now + send_interval
            delta = next_send - time.monotonic() - 0.0005
            if delta > 0:
                time.sleep(delta)
            else:
                time.sleep(0.001)

    send_thread = threading.Thread(target=_send_loop, daemon=True)
    send_thread.start()

    def _signal_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        with Live(console=console, refresh_per_second=5, screen=True) as live:
            while running:
                now = time.monotonic()

                if now >= next_ui:
                    stats = controller.get_stats()
                    stats["connected"] = controller.is_connected

                    if beep_enabled:
                        loss = stats["consecutive_loss"]
                        if loss > _prev_loss:
                            _beep_alert()
                        _prev_loss = loss

                    renderable = _build_renderable(stats)
                    live.update(renderable)
                    next_ui = now + 0.2

                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stop_send.set()
        send_thread.join(timeout=1)
        controller.stop()

    lifecycle = controller.get_stats()
    sys.stdout.write(
        f"\nTest complete: sent={lifecycle['tx_total']:,d} "
        f"recv={lifecycle['rx_total']:,d} "
        f"loss={lifecycle['lost_total']:,d} "
        f"({lifecycle['loss_rate_all']:.2f}%)\n"
    )
    return 0 if lifecycle['lost_total'] == 0 else 1


if __name__ == "__main__":
    _setup_logging()
    sys.exit(main())
