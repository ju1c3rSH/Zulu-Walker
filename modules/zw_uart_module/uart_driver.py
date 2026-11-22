# -*- coding: utf-8 -*-
"""Wrap the HAL UART with a binary frame parser and event dispatch.

FrameParser reassembles frames byte by byte; STM32UartInterface owns the
receiver callback, transmits frames under a lock, and publishes decoded events.
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
    DATA_TYPE_NAMES, NACK_REASON_NAMES,
    FrameData, parse_frame,
    parse_emergency_stop_payload,
    parse_cmd_request_payload,
    parse_cmd_stop_payload,
    parse_cmd_ack_payload,
    parse_cmd_nack_payload,
    parse_ping_payload,
    build_pong_frame,
)
from framework.hal.interface import Uart
from .exceptions import UartError
from utils.log_util import log_print


class ParserState(enum.Enum):
    """States of the frame parsing state machine."""

    WAITING_SOF = "WAITING_SOF"
    GOT_SOF1 = "GOT_SOF1"   # Received SOF1 (0xAA); waiting for SOF2 (0x55).
    GOT_SOF = "GOT_SOF"     # Received both SOF bytes; reading the Length field.
    GOT_LEN = "GOT_LEN"



class FrameParser:
    """Reassemble binary frames from a byte stream using a state machine.

    Bytes are fed one at a time and complete frames are returned. The kernel
    TTY layer already buffers the serial stream, so no Python ring buffer is
    needed.
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
        """Reset the parser to its initial state and discard buffered bytes."""
        self._state = ParserState.WAITING_SOF
        self._buffer.clear()
        self._expected_length = 0

    def feed(self, data: bytes) -> List[FrameData]:
        """
        Feed received bytes into the parser and return complete frames.

        Args:
            data: Raw bytes received from the serial port.

        Returns:
            Every FrameData completed by this batch of bytes.
        """
        frames = []

        for byte in data:
            frame = self._process_byte(byte)
            if frame is not None:
                frames.append(frame)

        return frames

    def _process_byte(self, byte: int) -> Optional[FrameData]:
        """
        Process one byte through the state machine.

        Args:
            byte: A single received byte.

        Returns:
            A FrameData when a frame completes, otherwise None.
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
                # Treat the previous 0xAA as noise and start a new frame here.
                self._buffer = bytearray([SOF1])
            else:
                self.reset()
            return None

        elif self._state == ParserState.GOT_SOF:
            # Length = Type (1) + Payload + CRC16 (2), so valid values are 3..255.
            if 3 <= byte <= 255:
                self._expected_length = byte
                self._buffer.append(byte)
                self._state = ParserState.GOT_LEN
            else:
                # Invalid length: discard the partial frame.
                self._logger.warning(f"Invalid frame length: {byte}")
                self.reset()
            return None

        elif self._state == ParserState.GOT_LEN:
            # Accumulate content bytes until the frame is complete.
            self._buffer.append(byte)

            # Total frame size: three header bytes plus Length content bytes.
            expected_size = 3 + self._expected_length

            if len(self._buffer) == expected_size:
                # Validate the complete frame.
                frame = self._parse_complete_frame()
                self.reset()
                return frame

            return None

        return None

    def _parse_complete_frame(self) -> Optional[FrameData]:
        """Validate the buffered bytes as a frame and report rejections."""
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
    """UART interface for STM32 communication.

    Wraps the HAL Uart, parses frames from the receiver callback, serializes
    transmits with a write lock, and publishes decoded events to an EventBus.
    """

    def __init__(self, uart: Uart):
        """
        Initialize the interface around a HAL UART.

        Args:
            uart: HAL UART instance.
        """
        self._uart = uart
        self._parser = FrameParser()
        self._parser.set_reject_callback(self._on_frame_rejected)

        # Serialize all transmit operations.
        self._write_lock = threading.Lock()

        # EventBus reference; assigned later to avoid a circular import.
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
        """Return whether the underlying UART is connected."""
        return self._uart.is_connected

    def set_log_level(self, level: int):
        """
        Set the logger level.

        Args:
            level: A logging level constant.
        """
        self._logger.setLevel(level)

    def set_debug_hex(self, enabled: bool = True):
        """
        Enable or disable hex dumps of frames in debug logs.

        Args:
            enabled: True to dump frame bytes as hex.
        """
        self._debug_hex = enabled

    def start(self) -> bool:
        """
        Connect the UART and start the background receiver.

        Returns:
            True when the interface started successfully.
        """
        try:
            if not self._uart.connect():
                self._logger.error(f"Failed to connect to UART")
                return False

            # Register the receive callback; the HAL runs it on its receiver thread.
            self._uart.start_receiver(self._on_data_received)

            self._logger.info(
                f"STM32UartInterface started"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to start UART interface: {e}")
            return False

    def stop(self):
        """Stop the receiver, disconnect, and reset the parser."""
        try:
            self._uart.stop_receiver()
            self._uart.disconnect()
            self._parser.reset()
            self._logger.info("STM32UartInterface stopped")
        except Exception as e:
            self._logger.error(f"Error stopping UART interface: {e}")

    def __enter__(self) -> "STM32UartInterface":
        """Start the interface and return it."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the interface when the context exits."""
        self.stop()
        return False

    def _on_data_received(self, data: bytes):
        """
        Handle bytes delivered by the HAL UART receiver callback.

        Args:
            data: Raw bytes received from the serial port.
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
        Dispatch a validated frame received from the master (MSPM0).

        Args:
            frame: Validated frame data.
        """
        if frame.frame_type == TYPE_CMD_REQUEST:
            parsed = parse_cmd_request_payload(frame.payload)
            if parsed is not None:
                dt_name = DATA_TYPE_NAMES.get(parsed[0], f"0x{parsed[0]:02X}")
                log_print(f"[UART RX] CMD_REQUEST type={dt_name} interval={parsed[1]}ms")
                if self._event_bus:
                    try:
                        from .events import CmdRequestEvent
                        self._event_bus.publish(
                            CmdRequestEvent(parsed[0], parsed[1]))
                    except ImportError:
                        pass

        elif frame.frame_type == TYPE_CMD_STOP:
            if parse_cmd_stop_payload(frame.payload):
                log_print("[UART RX] CMD_STOP")
                if self._event_bus:
                    try:
                        from .events import CmdStopEvent
                        self._event_bus.publish(CmdStopEvent())
                    except ImportError:
                        pass

        elif frame.frame_type == TYPE_CMD_ACK:
            parsed = parse_cmd_ack_payload(frame.payload)
            if parsed is not None:
                dt_name = DATA_TYPE_NAMES.get(parsed[0], f"0x{parsed[0]:02X}")
                log_print(f"[UART RX] CMD_ACK type={dt_name} freq={parsed[1]}Hz size={parsed[2]}B (unexpected, slave)")
            self._logger.warning(
                "Unexpected CMD_ACK received (slave role)")

        elif frame.frame_type == TYPE_CMD_NACK:
            parsed = parse_cmd_nack_payload(frame.payload)
            if parsed is not None:
                dt_name = DATA_TYPE_NAMES.get(parsed[0], f"0x{parsed[0]:02X}")
                reason_name = NACK_REASON_NAMES.get(parsed[1], f"0x{parsed[1]:02X}")
                log_print(f"[UART RX] CMD_NACK type={dt_name} reason={reason_name} (unexpected, slave)")
            self._logger.warning(
                "Unexpected CMD_NACK received (slave role)")

        elif frame.frame_type == TYPE_DATA_STREAM:
            self._logger.warning(
                "Unexpected DATA_STREAM received (slave role)")

        elif frame.frame_type == TYPE_HEARTBEAT:
            # Deprecated heartbeat: ignore it.
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
        Build and send an error frame to the STM32.

        Args:
            error_type: Error category (0=X, 1=Y, 2=Z, 3=other).
            error_value: Signed 16-bit error value.

        Returns:
            True when the entire frame was transmitted.
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


# Module test / demo
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

    # Build an ERROR frame: type 0, value -3.
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
