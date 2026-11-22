# -*- coding: utf-8 -*-
"""Define UART framing constants, frame builders, and parsers.

Frames use the layout SOF1 | SOF2 | Length | Type | Payload | CRC16, where
Length counts Type + Payload + CRC16 and CRC16 is a big-endian CRC16-CCITT
(poly 0x1021, init 0xFFFF) computed over the Type and Payload bytes.
"""

from dataclasses import dataclass
from typing import Optional


# Protocol constants
SOF1 = 0xAA
SOF2 = 0x55

# Frame types
TYPE_ERROR = 0x01               # Bidirectional: error_type(1B) + error_value(2B int16 LE)
TYPE_HEARTBEAT = 0x15           # Deprecated: bidirectional; seq, mission_state, visual_state
TYPE_VISUAL_SERVO_DATA = 0x17   # Deprecated: to MCU; error_x, error_y, flags, state
TYPE_EMERGENCY_STOP = 0x18      # Bidirectional: reason(1B)

# Master-slave protocol v3.0
TYPE_CMD_REQUEST     = 0x20  # Master to slave: request a data stream subscription
TYPE_CMD_ACK         = 0x21  # Slave to master: acknowledge a subscription
TYPE_CMD_NACK        = 0x22  # Slave to master: reject a subscription
TYPE_CMD_STOP        = 0x23  # Master to slave: stop the data stream
TYPE_DATA_STREAM     = 0x24  # Slave to master: streaming data frame

# Ping-pong (link heartbeat)
TYPE_PING            = 0x25  # Master to slave: keepalive probe
TYPE_PONG            = 0x26  # Slave to master: keepalive reply

# NACK reasons
NACK_UNSUPPORTED_TYPE = 0x01
NACK_NOT_READY        = 0x02
NACK_BUSY             = 0x03

# Data stream types (for CMD_REQUEST.data_type)
DATA_LINE_POSITION    = 0x01
DATA_TARGET_POSITION  = 0x02
DATA_TARGET_COUNT     = 0x03
DATA_DETECTION_STATUS = 0x04
DATA_ALL_TARGETS           = 0x05
DATA_SEGMENTATION_MASK    = 0x06
DATA_PENDULUM_POSITION    = 0x07

# Payload size in bytes per data type; None means variable length.
# Total TYPE_DATA_STREAM payload: seq + data_type + sub_payload.
DATA_PAYLOAD_SIZES = {
    DATA_LINE_POSITION: 8,
    DATA_TARGET_POSITION: 8,
    DATA_TARGET_COUNT: 3,
    DATA_DETECTION_STATUS: 6,
    DATA_ALL_TARGETS: None,
    DATA_SEGMENTATION_MASK: None,
    DATA_PENDULUM_POSITION: 10,
}

SUPPORTED_DATA_TYPES = set(DATA_PAYLOAD_SIZES.keys())

DATA_TYPE_NAMES: dict[int, str] = {
    DATA_LINE_POSITION:     "LINE_POS",
    DATA_TARGET_POSITION:   "TARGET_POS",
    DATA_TARGET_COUNT:      "TARGET_CNT",
    DATA_DETECTION_STATUS:  "DET_STAT",
    DATA_ALL_TARGETS:       "ALL_TAR",
    DATA_SEGMENTATION_MASK: "SEG_MASK",
    DATA_PENDULUM_POSITION: "PEND_POS",
}

NACK_REASON_NAMES: dict[int, str] = {
    NACK_UNSUPPORTED_TYPE: "UNSUPPORTED_TYPE",
    NACK_NOT_READY:        "NOT_READY",
    NACK_BUSY:             "BUSY",
}


# Error types for TYPE_ERROR
ERROR_TYPE_X = 0            # X-axis error
ERROR_TYPE_Y = 1            # Y-axis error
ERROR_TYPE_Z = 2            # Z-axis error
ERROR_TYPE_OTHER = 3        # Other error

# Frame size limits
MIN_FRAME_SIZE = 6          # SOF1 + SOF2 + Length + Type + CRC16, no payload
MAX_FRAME_SIZE = 258        # SOF1 + SOF2 + Length + Type + 252 payload + CRC16
MAX_PAYLOAD_SIZE = 252      # One-byte Length caps payload at 1 + 252 + 2 = 255


@dataclass
class FrameData:
    """A validated frame: type identifier and raw payload bytes."""
    frame_type: int
    payload: bytes


# CRC16-CCITT (poly 0x1021, init 0xFFFF)

def _build_crc16_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
        table.append(crc)
    return table

_CRC16_TABLE = _build_crc16_table()

def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Compute CRC16-CCITT (poly 0x1021, init 0xFFFF, no reflection or final XOR)."""
    crc = init
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def _build_frame(frame_type: int, payload: bytes) -> bytes:
    """Build a frame: SOF1, SOF2, Length, Type, Payload, big-endian CRC16."""
    content = bytes([frame_type]) + payload
    checksum = crc16_ccitt(content)
    length = len(content) + 2  # Length covers Type + Payload + the 2 CRC16 bytes.
    return bytes([SOF1, SOF2, length]) + content + checksum.to_bytes(2, byteorder='big')


def parse_frame(data: bytes) -> Optional[FrameData]:
    """Parse and validate a complete frame; return None if it is malformed."""
    if len(data) < MIN_FRAME_SIZE:
        return None

    if data[0] != SOF1 or data[1] != SOF2:
        return None

    length = data[2]
    expected_size = 3 + length  # Three header bytes plus Length content bytes.
    if len(data) != expected_size:
        return None

    frame_type = data[3]
    payload = data[4:expected_size - 2]
    received_checksum = int.from_bytes(data[expected_size - 2:expected_size], byteorder='big')
    content = data[3:expected_size - 2]  # CRC16 covers Type and Payload only.
    calculated_checksum = crc16_ccitt(content)

    if calculated_checksum != received_checksum:
        return None

    return FrameData(frame_type=frame_type, payload=payload)


def build_error_frame(error_type: int, error_value: int) -> bytes:
    """
    Build a TYPE_ERROR frame.

    Args:
        error_type: Error category (0=X, 1=Y, 2=Z, 3=other).
        error_value: Signed 16-bit error value.

    Returns:
        A complete frame ready to transmit.

    Raises:
        ValueError: If either parameter is out of range.
    """
    if not 0 <= error_type <= 3:
        raise ValueError(f"error_type must be 0-3, got {error_type}")
    if not -32768 <= error_value <= 32767:
        raise ValueError(f"error_value must be -32768~32767, got {error_value}")

    payload = bytes([error_type]) + error_value.to_bytes(2, byteorder='little', signed=True)
    return _build_frame(TYPE_ERROR, payload)


def build_heartbeat_frame(seq: int, mission_state: int, visual_state: int) -> bytes:
    """Build a TYPE_HEARTBEAT frame."""
    return _build_frame(TYPE_HEARTBEAT, bytes([seq, mission_state, visual_state]))


def parse_heartbeat_payload(payload: bytes) -> Optional[tuple]:
    """Decode a heartbeat payload into (seq, mission_state, visual_state); None on bad length."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def build_visual_servo_data_frame(
    error_x: int, error_y: int, flags: int, state: int
) -> bytes:
    """
    Build a TYPE_VISUAL_SERVO_DATA frame.

    error_x and error_y are little-endian signed int16 values; flags is a
    bitmask; state is the visual state (0=IDLE, 1=SEARCH, 2=TRACKING,
    3=RECOVERY, 4=FAIL).
    """
    payload = (
        error_x.to_bytes(2, byteorder='little', signed=True)
        + error_y.to_bytes(2, byteorder='little', signed=True)
        + bytes([flags, state])
    )
    return _build_frame(TYPE_VISUAL_SERVO_DATA, payload)


def parse_visual_servo_data_payload(payload: bytes) -> Optional[tuple]:
    """Decode a servo payload into (error_x, error_y, flags, state); None on bad length."""
    if len(payload) != 6:
        return None
    error_x = int.from_bytes(payload[0:2], byteorder='little', signed=True)
    error_y = int.from_bytes(payload[2:4], byteorder='little', signed=True)
    flags = payload[4]
    state = payload[5]
    return error_x, error_y, flags, state


def build_emergency_stop_frame(reason: int) -> bytes:
    """Build a TYPE_EMERGENCY_STOP frame."""
    return _build_frame(TYPE_EMERGENCY_STOP, bytes([reason]))


def parse_emergency_stop_payload(payload: bytes) -> Optional[int]:
    """Decode an emergency-stop payload into its reason; None on bad length."""
    if len(payload) != 1:
        return None
    return payload[0]


# Master-slave protocol v3.0 builders

def build_cmd_request_frame(data_type: int, min_interval_ms: int = 0) -> bytes:
    """Build a TYPE_CMD_REQUEST frame."""
    payload = bytes([data_type, min_interval_ms, 0x00])
    return _build_frame(TYPE_CMD_REQUEST, payload)


def build_cmd_ack_frame(data_type: int, max_freq_hz: int = 60, payload_size: int = 0) -> bytes:
    """Build a TYPE_CMD_ACK frame."""
    payload = bytes([data_type, max_freq_hz, payload_size])
    return _build_frame(TYPE_CMD_ACK, payload)


def build_cmd_nack_frame(data_type: int, reason: int) -> bytes:
    """Build a TYPE_CMD_NACK frame."""
    payload = bytes([data_type, reason])
    return _build_frame(TYPE_CMD_NACK, payload)


def build_cmd_stop_frame() -> bytes:
    """Build a TYPE_CMD_STOP frame with an empty payload."""
    return _build_frame(TYPE_CMD_STOP, b'')


def build_data_stream_frame(seq: int, data_type: int, sub_payload: bytes) -> bytes:
    """
    Build a TYPE_DATA_STREAM frame.

    Args:
        seq: Sequence counter (0-255).
        data_type: The data stream type being sent.
        sub_payload: Type-specific bytes, excluding the seq and data_type header.
    """
    inner = bytes([seq, data_type]) + sub_payload
    return _build_frame(TYPE_DATA_STREAM, inner)


# Ping-pong builders

def build_ping_frame(seq: int) -> bytes:
    """Build a TYPE_PING frame carrying a one-byte sequence number."""
    return _build_frame(TYPE_PING, bytes([seq & 0xFF]))


def build_pong_frame(seq: int) -> bytes:
    """Build a TYPE_PONG frame echoing the sequence received from the MCU."""
    return _build_frame(TYPE_PONG, bytes([seq & 0xFF]))


# Master-slave protocol v3.0 parsers

def parse_cmd_request_payload(payload: bytes) -> Optional[tuple]:
    """Decode a CMD_REQUEST payload into (data_type, min_interval_ms, reserved).

    Return None when the payload length is wrong.
    """
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def parse_cmd_ack_payload(payload: bytes) -> Optional[tuple]:
    """Decode a CMD_ACK payload into (data_type, max_freq_hz, payload_size); None if malformed."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def parse_cmd_nack_payload(payload: bytes) -> Optional[tuple]:
    """Decode a CMD_NACK payload into (data_type, reason); None on bad length."""
    if len(payload) != 2:
        return None
    return payload[0], payload[1]


def parse_cmd_stop_payload(payload: bytes) -> bool:
    """Return True when a CMD_STOP payload is empty."""
    return len(payload) == 0


def parse_data_stream_header(payload: bytes) -> Optional[tuple]:
    """Decode the data stream header into (seq, data_type, sub_payload).

    The caller interprets sub_payload according to data_type.
    """
    if len(payload) < 2:
        return None
    return payload[0], payload[1], payload[2:]


# Ping-pong parsers

def parse_ping_payload(payload: bytes) -> Optional[int]:
    """Decode a ping payload into its sequence number; None on bad length."""
    if len(payload) != 1:
        return None
    return payload[0]


parse_pong_payload = parse_ping_payload  # Same payload layout as the ping frame.
