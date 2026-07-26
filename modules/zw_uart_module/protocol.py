# -*- coding: utf-8 -*-
"""
zw_uart_module - Protocol definitions

Defines protocol constants, frame building, and parsing functions for
UART communication between Orange Pi and STM32.

Protocol Frame Structure (v2.1):
| SOF1 | SOF2 | Length | Type | Payload | CRC16_LO | CRC16_HI |
|  1   |  1   |   1    |  1   |  0~252  |    1     |    1     |

- SOF1: Start of Frame byte 1, fixed 0xAA
- SOF2: Start of Frame byte 2, fixed 0x55
- Length: Bytes from Type to CRC16 (Type + Payload + CRC16)
- Type: Frame type identifier
- Payload: Variable length data
- CRC16: CRC16-CCITT (poly 0x1021, init 0xFFFF) of Type to Payload bytes, big-endian
"""

from dataclasses import dataclass
from typing import Optional


# Protocol constants
SOF1 = 0xAA
SOF2 = 0x55

# Frame types
TYPE_ERROR = 0x01           # Orange Pi -> STM32: error_type(1B) + error_value(2B, int16 LE)
TYPE_HEARTBEAT = 0x15           # Bidirectional: seq + mission_state + visual_state
TYPE_VISUAL_SERVO_DATA = 0x17   # Orange Pi -> MCU: error_x(2B) + error_y(2B) + flags(1B) + state(1B)
TYPE_EMERGENCY_STOP = 0x18      # Bidirectional: reason(1B)


# Error types for TYPE_ERROR
ERROR_TYPE_X = 0            # X direction error
ERROR_TYPE_Y = 1            # Y direction error
ERROR_TYPE_Z = 2            # Z direction error
ERROR_TYPE_OTHER = 3        # Other error

# Frame size limits
MIN_FRAME_SIZE = 6          # SOF1 + SOF2 + Length + Type + CRC16 (no payload)
MAX_FRAME_SIZE = 258        # SOF1 + SOF2 + Length + Type + 252 payload + CRC16
MAX_PAYLOAD_SIZE = 252      # Limited by Length field: Length(1+252+2) ≤ 255


@dataclass
class FrameData:
    """Parsed frame data."""
    frame_type: int
    payload: bytes


# === CRC16-CCITT (poly 0x1021, init 0xFFFF) ===

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
    """CRC16-CCITT (poly 0x1021, init 0xFFFF, no reflection, no final xor)."""
    crc = init
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def _build_frame(frame_type: int, payload: bytes) -> bytes:
    """Build a standard SOF1/SOF2/Length/Type/Payload/CRC16 frame."""
    content = bytes([frame_type]) + payload
    checksum = crc16_ccitt(content)
    length = len(content) + 2  # Type + Payload + CRC16(2 bytes)
    return bytes([SOF1, SOF2, length]) + content + checksum.to_bytes(2, byteorder='big')


def parse_frame(data: bytes) -> Optional[FrameData]:
    """Parse and validate a complete frame."""
    if len(data) < MIN_FRAME_SIZE:
        return None

    if data[0] != SOF1 or data[1] != SOF2:
        return None

    length = data[2]
    expected_size = 3 + length  # SOF1(1) + SOF2(1) + Length(1) + content(length)
    if len(data) != expected_size:
        return None

    frame_type = data[3]
    payload = data[4:expected_size - 2]
    received_checksum = int.from_bytes(data[expected_size - 2:expected_size], byteorder='big')
    content = data[3:expected_size - 2]  # Type + Payload
    calculated_checksum = crc16_ccitt(content)

    if calculated_checksum != received_checksum:
        return None

    return FrameData(frame_type=frame_type, payload=payload)


def build_error_frame(error_type: int, error_value: int) -> bytes:
    """
    Build an error frame for sending to STM32.

    Args:
        error_type: Error type (0=X, 1=Y, 2=Z, 3=Other)
        error_value: Error value (int16, -32768~32767)

    Returns:
        Complete frame bytes ready to send

    Raises:
        ValueError: If parameters are out of range
    """
    if not 0 <= error_type <= 3:
        raise ValueError(f"error_type must be 0-3, got {error_type}")
    if not -32768 <= error_value <= 32767:
        raise ValueError(f"error_value must be -32768~32767, got {error_value}")

    payload = bytes([error_type]) + error_value.to_bytes(2, byteorder='little', signed=True)
    return _build_frame(TYPE_ERROR, payload)


def build_heartbeat_frame(seq: int, mission_state: int, visual_state: int) -> bytes:
    """Build TYPE_HEARTBEAT frame."""
    return _build_frame(TYPE_HEARTBEAT, bytes([seq, mission_state, visual_state]))


def parse_heartbeat_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_HEARTBEAT payload -> (seq, mission_state, visual_state)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def build_visual_servo_data_frame(
    error_x: int, error_y: int, flags: int, state: int
) -> bytes:
    """
    Build TYPE_VISUAL_SERVO_DATA frame (每帧必发).
    error_x/error_y: signed int16 LE.
    flags: bitmask.
    state: visual_state (0=IDLE, 1=SEARCH, 2=TRACKING, 3=RECOVERY, 4=FAIL).
    """
    payload = (
        error_x.to_bytes(2, byteorder='little', signed=True)
        + error_y.to_bytes(2, byteorder='little', signed=True)
        + bytes([flags, state])
    )
    return _build_frame(TYPE_VISUAL_SERVO_DATA, payload)


def parse_visual_servo_data_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_VISUAL_SERVO_DATA payload -> (error_x, error_y, flags, state)."""
    if len(payload) != 6:
        return None
    error_x = int.from_bytes(payload[0:2], byteorder='little', signed=True)
    error_y = int.from_bytes(payload[2:4], byteorder='little', signed=True)
    flags = payload[4]
    state = payload[5]
    return error_x, error_y, flags, state


def build_emergency_stop_frame(reason: int) -> bytes:
    """Build TYPE_EMERGENCY_STOP frame."""
    return _build_frame(TYPE_EMERGENCY_STOP, bytes([reason]))


def parse_emergency_stop_payload(payload: bytes) -> Optional[int]:
    """Parse TYPE_EMERGENCY_STOP payload -> reason."""
    if len(payload) != 1:
        return None
    return payload[0]
