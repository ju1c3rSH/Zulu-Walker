# -*- coding: utf-8 -*-
"""
zw_uart_module - UART driver for STM32 communication

Implements STM32UartInterface class that wraps SerialController for
binary frame-based communication with state machine parsing.
"""

import enum
import logging
import threading
from typing import Callable, List, Optional

from .protocol import (
    SOF1, SOF2,
    TYPE_HEARTBEAT, TYPE_EMERGENCY_STOP,
    TYPE_CMD_REQUEST, TYPE_CMD_STOP,
    TYPE_CMD_ACK, TYPE_CMD_NACK, TYPE_DATA_STREAM,
    TYPE_PING, TYPE_PONG,
    FrameData, parse_frame,
    parse_emergency_stop_payload,
    parse_cmd_request_payload,
    parse_cmd_stop_payload,
    parse_ping_payload,
    build_pong_frame,
)
from framework.hal.interface import Uart
from .exceptions import UartError
from utils.log_util import log_print


class ParserState(enum.Enum):
    """State machine states for frame parsing."""

    WAITING_SOF = "WAITING_SOF"
    GOT_SOF1 = "GOT_SOF1"   # Got 0xAA, waiting for 0x55
    GOT_SOF = "GOT_SOF"     # Got 0xAA 0x55, reading Length
    GOT_LEN = "GOT_LEN"



class FrameParser:
    """
    State machine parser for binary frames.

    Processes incoming bytes and outputs complete frames when parsed.
    No Python-level ring buffer needed - Linux kernel TTY handles buffering.
    """

    def __init__(self):
        self._state = ParserState.WAITING_SOF
        self._buffer = bytearray()
        self._expected_length = 0
        self._logger = logging.getLogger(__name__)
        self._on_reject = None

    def set_reject_callback(self, callback) -> None:
        self._on_reject = callback

    def reset(self):
        """Reset parser state."""
        self._state = ParserState.WAITING_SOF
        self._buffer.clear()
        self._expected_length = 0

    def feed(self, data: bytes) -> List[FrameData]:
        """
        Feed data to the parser and return complete frames.

        Args:
            data: Raw bytes received from serial port

        Returns:
            List of successfully parsed FrameData objects
        """
        frames = []

        for byte in data:
            frame = self._process_byte(byte)
            if frame is not None:
                frames.append(frame)

        return frames

    def _process_byte(self, byte: int) -> Optional[FrameData]:
        """
        Process a single byte through the state machine.

        Args:
            byte: Single byte to process

        Returns:
            FrameData if a complete frame was parsed, None otherwise
        """
        if self._state == ParserState.WAITING_SOF:
            if byte == SOF1:
                self._buffer = bytearray([SOF1])
                self._state = ParserState.GOT_SOF1
            return None

        elif self._state == ParserState.GOT_SOF1:
            if byte == SOF2:
                self._buffer.append(byte)
                self._state = ParserState.GOT_SOF
            elif byte == SOF1:
                # Re-arm: previous 0xAA was noise, this one is new SOF1
                self._buffer = bytearray([SOF1])
            else:
                self.reset()
            return None

        elif self._state == ParserState.GOT_SOF:
            # This is the Length field
            # Length = Type(1) + Payload(var) + CRC16(2)
            # Minimum length is 3 (Type + CRC16, no payload)
            # Maximum length is 255 (Type + 252 payload + CRC16)
            if 3 <= byte <= 255:
                self._expected_length = byte
                self._buffer.append(byte)
                self._state = ParserState.GOT_LEN
            else:
                # Invalid length, reset
                self._logger.warning(f"Invalid frame length: {byte}")
                self.reset()
            return None

        elif self._state == ParserState.GOT_LEN:
            # Accumulate bytes until we have complete frame
            self._buffer.append(byte)

            # Expected total size: SOF1(1) + SOF2(1) + Length(1) + (Length bytes)
            expected_size = 3 + self._expected_length

            if len(self._buffer) == expected_size:
                # Complete frame received
                frame = self._parse_complete_frame()
                self.reset()
                return frame

            return None

        return None

    def _parse_complete_frame(self) -> Optional[FrameData]:
        """Parse the accumulated buffer as a complete frame."""
        raw = bytes(self._buffer)
        frame = parse_frame(raw)
        if frame is None:
            self._logger.warning(
                f"Frame validation failed: {raw.hex()}"
            )
            if self._on_reject:
                self._on_reject(raw)
        return frame


class STM32UartInterface:
    """
    UART interface for STM32 communication.

    Wraps SerialController and provides:
    - Error frame transmission
    - Background frame parsing with state machine
    - Frame dispatch via EventBus
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
        """
        Set logging level.

        Args:
            level: logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self._logger.setLevel(level)

    def set_debug_hex(self, enabled: bool = True):
        """
        Enable/disable hex dump of frames in debug logs.

        Args:
            enabled: True to enable hex dumps
        """
        self._debug_hex = enabled

    def start(self) -> bool:
        """
        Start the UART interface.

        Connects to serial port and starts background receiver.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if not self._uart.connect():
                self._logger.error(f"Failed to connect to UART")
                return False

            # Start async receiver with callback
            self._uart.start_receiver(self._on_data_received)

            self._logger.info(
                f"STM32UartInterface started"
            )
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
        """
        Callback for received data from SerialController.

        Feeds data to parser and processes parsed frames.

        Args:
            data: Raw bytes received from serial port
        """
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
            log_print(
                f"[UART STATS] rx_bytes={self._rx_bytes_total} "
                f"ok={self._rx_frames_ok} bad={self._rx_frames_bad} "
                f"unknown={self._rx_frames_unknown}"
            )

    def _on_frame_rejected(self, raw: bytes):
        self._rx_frames_bad += 1
        log_print(f"[UART] CRC/parse fail: {raw.hex()}")

    def _handle_frame(self, frame: FrameData):
        """
        Handle a parsed frame from master (MSPM0).

        Args:
            frame: Parsed frame data
        """
        if frame.frame_type == TYPE_CMD_REQUEST:
            parsed = parse_cmd_request_payload(frame.payload)
            if parsed is not None and self._event_bus:
                try:
                    from .events import CmdRequestEvent
                    self._event_bus.publish(
                        CmdRequestEvent(parsed[0], parsed[1]))
                except ImportError:
                    pass

        elif frame.frame_type == TYPE_CMD_STOP:
            if parse_cmd_stop_payload(frame.payload) and self._event_bus:
                try:
                    from .events import CmdStopEvent
                    self._event_bus.publish(CmdStopEvent())
                except ImportError:
                    pass

        elif frame.frame_type == TYPE_CMD_ACK:
            self._logger.warning(
                "Unexpected CMD_ACK received (slave role)")

        elif frame.frame_type == TYPE_CMD_NACK:
            self._logger.warning(
                "Unexpected CMD_NACK received (slave role)")

        elif frame.frame_type == TYPE_DATA_STREAM:
            self._logger.warning(
                "Unexpected DATA_STREAM received (slave role)")

        elif frame.frame_type == TYPE_HEARTBEAT:
            # [DEPRECATED] Silently ignore
            pass

        elif frame.frame_type == TYPE_EMERGENCY_STOP:
            parsed = parse_emergency_stop_payload(frame.payload)
            if parsed is not None:
                self._logger.error(f"EMERGENCY_STOP reason={parsed}")
                log_print(f"[UART RX] EMERGENCY_STOP reason={parsed}")
                if self._event_bus:
                    try:
                        from .events import EmergencyStopEvent
                        self._event_bus.publish(EmergencyStopEvent(parsed))
                    except ImportError:
                        pass

        elif frame.frame_type == TYPE_PING:
            seq = parse_ping_payload(frame.payload)
            if seq is not None:
                log_print(f"[UART RX] PING seq={seq}")
                pong_frame = build_pong_frame(seq)
                if self.send_raw(pong_frame):
                    log_print(f"[UART TX] PONG seq={seq}")

        elif frame.frame_type == TYPE_PONG:
            seq = parse_ping_payload(frame.payload)
            if seq is not None:
                log_print(f"[UART RX] PONG seq={seq}  (unexpected, slave role)")

        else:
            self._rx_frames_unknown += 1
            self._logger.warning(f"Unknown frame type: 0x{frame.frame_type:02X}")

    def send_error(self, error_type: int, error_value: int) -> bool:
        """
        Send an error frame to STM32.

        Args:
            error_type: Error type (0=X, 1=Y, 2=Z, 3=Other)
            error_value: Error value (int16, -32768~32767)

        Returns:
            True if sent successfully, False otherwise
        """
        from .protocol import build_error_frame

        try:
            frame = build_error_frame(error_type, error_value)

            with self._write_lock:
                if not self._uart.is_connected:
                    self._logger.warning("Cannot send: not connected")
                    return False

                bytes_sent = self._uart.send(frame)

                if bytes_sent == len(frame):
                    if self._debug_hex:
                        self._logger.debug(f"Sent: {frame.hex()}")
                    self._logger.info(
                        f"Sent error frame: type={error_type}, value={error_value}"
                    )
                    log_print(f"[UART TX] ERROR type={error_type} value={error_value}")
                    return True
                else:
                    self._logger.error(
                        f"Send incomplete: {bytes_sent}/{len(frame)} bytes"
                    )
                    return False

        except ValueError as e:
            self._logger.error(f"Invalid error parameters: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Failed to send error frame: {e}")
            return False


# === Module test / demo ===
if __name__ == "__main__":
    import time

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test frame parser
    log_print("=== Frame Parser Test ===")
    parser = FrameParser()

    # Test heartbeat frame
    from .protocol import build_heartbeat_frame
    test_frame = build_heartbeat_frame(seq=1, mission_state=2, visual_state=3)
    frames = parser.feed(test_frame)
    log_print(f"Parsed frames: {frames}")
    if frames:
        log_print(f"  Type: 0x{frames[0].frame_type:02X}, Payload: {frames[0].payload.hex()}")

    # Test ERROR frame: type=0, value=-3
    log_print("\n=== Error Frame Build Test ===")
    from .protocol import build_error_frame
    error_frame = build_error_frame(0, -3)
    log_print(f"Error frame: {error_frame.hex()}")

    # Demo usage (requires actual hardware)
    log_print("\n=== Hardware Demo ===")
    log_print("To test with hardware, run:")
    log_print("  with STM32UartInterface('/dev/ttyS4', 921600) as uart:")
    log_print("      while True:")
    log_print("          log_print(f'Zone: cur={uart.get_current_zone()}')")
    log_print("          time.sleep(0.1)")
