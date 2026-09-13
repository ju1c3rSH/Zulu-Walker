# -*- coding: utf-8 -*-
"""
zw_uart_module - UART driver (近端视觉从站, panzer_links_protocol.md v2.1)

STM32UartInterface 包装 HAL Uart，提供：
- 基于 ProtocolParser（C 协议解析器 1:1 移植）的后台收帧
- v2.1 从站帧分发: CMD_REQUEST / CMD_STOP -> EventBus 事件
- 原始帧发送（线程安全）

从站只会收到主站的 0x20/0x23；其余帧型按协议意外帧记录（不响应）。
"""

import logging
import threading
from typing import List, Optional

from .parser import ProtocolParser
from .protocol import (
    DATA_ID_NAMES, NACK_REASON_NAMES,
    FrameData,
    parse_cmd_request_payload,
    parse_cmd_stop_payload,
)
from framework.hal.interface import Uart
from utils.log_util import log_print


class FrameParser:
    """C 状态机包装：feed() 返回完整帧列表（RX 线程接口），CRC 坏帧走 reject 回调。"""

    def __init__(self, body_capacity: int = 255):
        self._parser = ProtocolParser(body_capacity, on_crc_error=self._on_crc_error)
        self._on_reject = None

    def set_reject_callback(self, callback) -> None:
        self._on_reject = callback

    def _on_crc_error(self, body: bytes) -> None:
        if self._on_reject:
            try:
                self._on_reject(body)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset parser state."""
        self._parser.reset()

    @property
    def stats(self):
        return self._parser.stats

    def feed(self, data: bytes) -> List[FrameData]:
        """Feed data to the parser and return complete frames."""
        return self._parser.feed_block(data)


class STM32UartInterface:
    """
    UART interface for the near-end vision slave link.

    Wraps the HAL Uart instance and provides:
    - Background frame parsing (C-faithful state machine)
    - v2.1 slave-role frame dispatch via EventBus
    - Thread-safe raw frame transmission
    """

    def __init__(self, uart: Uart):
        """
        Initialize UART interface.

        Args:
            uart: HAL Uart Protocol instance
        """
        self._uart = uart
        self._parser = FrameParser()
        self._parser.set_reject_callback(self._on_frame_rejected)

        # Write lock for send operations
        self._write_lock = threading.Lock()

        # EventBus (set after init to avoid circular imports)
        self._event_bus = None

        # Logger
        self._logger = logging.getLogger(__name__)
        self._debug_hex = False

        self._rx_bytes_total = 0
        self._rx_frames_ok = 0
        self._rx_frames_bad = 0
        self._rx_frames_unknown = 0
        self._stats_log_counter = 0

    @property
    def is_connected(self) -> bool:
        """Check if serial port is connected."""
        return self._uart.is_connected

    def set_log_level(self, level: int):
        """Set logging level."""
        self._logger.setLevel(level)

    def set_debug_hex(self, enabled: bool = True):
        """Enable/disable hex dump of frames in debug logs."""
        self._debug_hex = enabled

    def start(self) -> bool:
        """Connect to serial port and start background receiver."""
        try:
            if not self._uart.connect():
                self._logger.error("Failed to connect to UART")
                return False

            self._uart.start_receiver(self._on_data_received)
            self._logger.info("STM32UartInterface started (v2.1 near-cam slave)")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start UART interface: {e}")
            return False

    def stop(self):
        """Stop the UART interface and cleanup resources."""
        try:
            self._uart.stop_receiver()
            self._uart.disconnect()
            self._parser.reset()
            self._logger.info("STM32UartInterface stopped")
        except Exception as e:
            self._logger.error(f"Error stopping UART interface: {e}")

    def __enter__(self) -> "STM32UartInterface":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def _on_data_received(self, data: bytes):
        """RX callback: feed parser, dispatch frames (runs on receiver thread)."""
        try:
            self._rx_bytes_total += len(data)

            if self._debug_hex:
                self._logger.debug(f"Received: {data.hex()}")

            frames = self._parser.feed(data)

            for frame in frames:
                self._handle_frame_with_stats(frame)

        except Exception as e:
            self._logger.error(f"Error processing received data: {e}")

    def set_event_bus(self, bus):
        self._event_bus = bus

    def send_raw(self, frame: bytes) -> bool:
        try:
            with self._write_lock:
                if not self._uart.is_connected:
                    return False
                return self._uart.send(frame) == len(frame)
        except Exception as e:
            self._logger.error(f"Failed to send raw frame: {e}")
            return False

    def _handle_frame_with_stats(self, frame: FrameData):
        self._rx_frames_ok += 1
        self._handle_frame(frame)

        self._stats_log_counter += 1
        if self._stats_log_counter >= 500:
            self._stats_log_counter = 0
            stats = self._parser.stats
            log_print(
                f"[UART STATS] rx_bytes={self._rx_bytes_total} "
                f"ok={self._rx_frames_ok} bad={self._rx_frames_bad} "
                f"crc_err={stats.crc_errors} len_err={stats.length_errors} "
                f"unknown={self._rx_frames_unknown}"
            )

    def _on_frame_rejected(self, raw: bytes):
        self._rx_frames_bad += 1
        log_print(f"[UART] CRC/parse fail: {raw.hex()}")

    def _publish(self, event) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(event)
        except Exception as e:
            self._logger.error(f"Event publish failed: {e}")

    def _handle_frame(self, frame: FrameData):
        """Handle a parsed master frame (v2.1 slave role)."""
        if frame.frame_type == 0x20:   # TYPE_CMD_REQUEST
            parsed = parse_cmd_request_payload(frame.payload)
            if parsed is not None:
                data_id, min_interval_ms, codes = parsed
                dt_name = DATA_ID_NAMES.get(data_id, f"0x{data_id:02X}")
                if codes is not None:
                    log_print(f"[UART RX] CMD_REQUEST type={dt_name} "
                              f"interval={min_interval_ms}ms codes={list(codes)}")
                else:
                    log_print(f"[UART RX] CMD_REQUEST type={dt_name} "
                              f"interval={min_interval_ms}ms (resubscribe)")
                from .events import CmdRequestEvent
                event = CmdRequestEvent(
                    data_type=data_id,
                    min_interval_ms=min_interval_ms,
                    has_codes=codes is not None,
                )
                if codes is not None:
                    event.explosive_color, event.target_color, event.rescue_shape = codes
                self._publish(event)
            else:
                self._logger.warning("CMD_REQUEST with empty payload ignored")

        elif frame.frame_type == 0x23:   # TYPE_CMD_STOP
            if parse_cmd_stop_payload(frame.payload):
                log_print("[UART RX] CMD_STOP")
                from .events import CmdStopEvent
                self._publish(CmdStopEvent())
            else:
                self._logger.warning("CMD_STOP with non-empty payload ignored")

        elif frame.frame_type == 0x21:   # TYPE_CMD_ACK
            self._logger.warning("Unexpected CMD_ACK received (slave role)")

        elif frame.frame_type == 0x22:   # TYPE_CMD_NACK
            self._logger.warning("Unexpected CMD_NACK received (slave role)")

        elif frame.frame_type == 0x24:   # TYPE_DATA_STREAM
            self._logger.warning("Unexpected DATA_STREAM received (slave role)")

        else:
            self._rx_frames_unknown += 1
            self._logger.warning(f"Unknown frame type: 0x{frame.frame_type:02X}")


# === Module test / demo ===
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    log_print("=== Frame Parser Test (C-faithful) ===")
    parser = FrameParser()

    from .protocol import build_cmd_ack_frame, build_data_stream_frame
    from .protocol import DATA_EXPLOSIVE, build_explosive_sub_payload

    ack = build_cmd_ack_frame(DATA_EXPLOSIVE)
    log_print(f"ACK frame: {ack.hex()}")
    for f in parser.feed(ack):
        log_print(f"  parsed type=0x{f.frame_type:02X} payload={f.payload.hex()}")

    stream = build_data_stream_frame(0x2A, DATA_EXPLOSIVE,
                                     build_explosive_sub_payload(1234, 2, True))
    log_print(f"STREAM frame: {stream.hex()}")
    for f in parser.feed(b"\x00\xAA" + stream + b"\xAA\xAA\x55" + stream[2:]):
        log_print(f"  parsed type=0x{f.frame_type:02X} payload={f.payload.hex()}")
