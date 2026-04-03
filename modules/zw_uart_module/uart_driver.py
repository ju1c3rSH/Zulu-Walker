# -*- coding: utf-8 -*-
"""
zw_uart_module - UART driver for STM32 communication

Implements STM32UartInterface class that wraps SerialController for
binary frame-based communication with state machine parsing.
"""

import enum
import logging
import threading
from typing import List, Optional

from .protocol import (
    SOF, TYPE_ARRIVED, TYPE_PICK, TYPE_SET,
    FrameData, parse_frame, parse_zone_payload
)
from .exceptions import UartError

# Import SerialController from utils
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.serial_controller import SerialController


class ParserState(enum.Enum):
    """State machine states for frame parsing."""
    WAITING_SOF = "WAITING_SOF"
    GOT_SOF = "GOT_SOF"
    GOT_LEN = "GOT_LEN"
    READING_DATA = "READING_DATA"


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
            if byte == SOF:
                self._buffer = bytearray([SOF])
                self._state = ParserState.GOT_SOF
            return None

        elif self._state == ParserState.GOT_SOF:
            # This is the Length field
            # Length = Type(1) + Payload(var) + Checksum(1)
            # Minimum length is 2 (Type + Checksum, no payload)
            # Maximum length is 253 (Type + 252 payload + Checksum)
            if 2 <= byte <= 253:
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

            # Expected total size: SOF(1) + Length(1) + (Length bytes)
            expected_size = 2 + self._expected_length

            if len(self._buffer) == expected_size:
                # Complete frame received
                frame = self._parse_complete_frame()
                self.reset()
                return frame

            return None

        return None

    def _parse_complete_frame(self) -> Optional[FrameData]:
        """Parse the accumulated buffer as a complete frame."""
        frame = parse_frame(bytes(self._buffer))
        if frame is None:
            self._logger.warning(
                f"Frame validation failed: {self._buffer.hex()}"
            )
        return frame


class STM32UartInterface:
    """
    UART interface for STM32 communication.

    Wraps SerialController and provides:
    - Thread-safe state queries (zone information)
    - Error frame transmission
    - Background frame parsing with state machine

    Usage:
        with STM32UartInterface("/dev/ttyS0", 115200) as uart:
            uart.send_error(0, -3)
            zone = uart.get_current_zone()
    """

    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 115200):
        """
        Initialize UART interface.

        Args:
            port: Serial port device path
            baudrate: Baud rate
        """
        self._port = port
        self._baudrate = baudrate
        self._serial = SerialController(port, baudrate)
        self._parser = FrameParser()

        # Thread-safe state variables
        self._state_lock = threading.RLock()
        self._current_zone: int = 0
        self._last_arrived_zone: int = 0
        self._last_pick_zone: int = 0

        # Write lock for send operations
        self._write_lock = threading.Lock()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._debug_hex = False

    @property
    def port(self) -> str:
        """Get the serial port path."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Get the baud rate."""
        return self._baudrate

    @property
    def is_connected(self) -> bool:
        """Check if serial port is connected."""
        return self._serial.is_connected

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
            if not self._serial.connect():
                self._logger.error(f"Failed to connect to {self._port}")
                return False

            # Start async receiver with callback
            self._serial.start_receiver(self._on_data_received)

            self._logger.info(
                f"STM32UartInterface started on {self._port} @ {self._baudrate}bps"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to start UART interface: {e}")
            return False

    def stop(self):
        """Stop the UART interface and cleanup resources."""
        try:
            self._serial.stop_receiver()
            self._serial.disconnect()
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
            if self._debug_hex:
                self._logger.debug(f"Received: {data.hex()}")

            frames = self._parser.feed(data)

            for frame in frames:
                self._handle_frame(frame)

        except Exception as e:
            self._logger.error(f"Error processing received data: {e}")

    def _handle_frame(self, frame: FrameData):
        """
        Handle a parsed frame by updating internal state.

        Args:
            frame: Parsed frame data
        """
        if frame.frame_type == TYPE_ARRIVED:
            zone_id = parse_zone_payload(frame.payload)
            if zone_id is not None:
                with self._state_lock:
                    self._last_arrived_zone = zone_id
                self._logger.info(f"ARRIVED_AT_ZONE: zone={zone_id}")

        elif frame.frame_type == TYPE_PICK:
            zone_id = parse_zone_payload(frame.payload)
            if zone_id is not None:
                with self._state_lock:
                    self._last_pick_zone = zone_id
                self._logger.info(f"PICK_AT_ZONE: zone={zone_id}")

        elif frame.frame_type == TYPE_SET:
            zone_id = parse_zone_payload(frame.payload)
            if zone_id is not None:
                with self._state_lock:
                    self._current_zone = zone_id
                self._logger.info(f"SET_ZONE: zone={zone_id}")

        else:
            self._logger.warning(f"Unknown frame type: 0x{frame.frame_type:02X}")

    def get_current_zone(self) -> int:
        """
        Get the current zone ID.

        Returns:
            Current zone ID set by STM32
        """
        with self._state_lock:
            return self._current_zone

    def get_last_arrived_zone(self) -> int:
        """
        Get the last arrived zone ID.

        Returns:
            Zone ID from last ARRIVED_AT_ZONE event
        """
        with self._state_lock:
            return self._last_arrived_zone

    def get_last_pick_zone(self) -> int:
        """
        Get the last pick zone ID.

        Returns:
            Zone ID from last PICK_AT_ZONE event
        """
        with self._state_lock:
            return self._last_pick_zone

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
                if not self._serial.is_connected:
                    self._logger.warning("Cannot send: not connected")
                    return False

                bytes_sent = self._serial.send_bytes(frame)

                if bytes_sent == len(frame):
                    if self._debug_hex:
                        self._logger.debug(f"Sent: {frame.hex()}")
                    self._logger.info(
                        f"Sent error frame: type={error_type}, value={error_value}"
                    )
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
    print("=== Frame Parser Test ===")
    parser = FrameParser()

    # Test SET_ZONE frame for zone 5: AA 02 04 05 01
    test_frame = bytes([0xAA, 0x02, 0x04, 0x05, 0x01])
    frames = parser.feed(test_frame)
    print(f"Parsed frames: {frames}")
    if frames:
        print(f"  Type: 0x{frames[0].frame_type:02X}, Payload: {frames[0].payload.hex()}")

    # Test ERROR frame: type=0, value=-3
    print("\n=== Error Frame Build Test ===")
    from protocol import build_error_frame
    error_frame = build_error_frame(0, -3)
    print(f"Error frame: {error_frame.hex()}")

    # Demo usage (requires actual hardware)
    print("\n=== Hardware Demo ===")
    print("To test with hardware, run:")
    print("  with STM32UartInterface('/dev/ttyS0', 115200) as uart:")
    print("      while True:")
    print("          print(f'Zone: cur={uart.get_current_zone()}')")
    print("          time.sleep(0.1)")
