# -*- coding: utf-8 -*-
"""
zw_uart_module - Protocol definitions

Defines protocol constants, frame building, and parsing functions for
UART communication between Orange Pi and STM32.

Protocol Frame Structure:
| SOF | Length | Type | Payload | Checksum |
|  1  |   1    |  1   |  0~252  |    1     |

- SOF: Start of Frame, fixed 0xAA
- Length: Bytes from Type to Checksum (Type + Payload + Checksum)
- Type: Frame type identifier
- Payload: Variable length data
- Checksum: XOR of Type to Payload bytes
"""

from dataclasses import dataclass
from typing import Optional


# Protocol constants
SOF = 0xAA

# Frame types
TYPE_ERROR = 0x01           # Orange Pi -> STM32: error_type(1B) + error_value(2B, int16 LE)
TYPE_ARRIVED = 0x02         # STM32 -> Orange Pi: zone_id(1B)
TYPE_PICK = 0x03            # STM32 -> Orange Pi: zone_id(1B)
TYPE_SET = 0x04             # STM32 -> Orange Pi: zone_id(1B)

# Error types for TYPE_ERROR
ERROR_TYPE_X = 0            # X direction error
ERROR_TYPE_Y = 1            # Y direction error
ERROR_TYPE_Z = 2            # Z direction error
ERROR_TYPE_OTHER = 3        # Other error

# Orange Send 状态枚举（与 VisualStateMachine.States 对应）
ORANGE_STATE_IDLE = 0       # 待机
ORANGE_STATE_SEARCH = 1     # 搜索
ORANGE_STATE_TRACKING = 2   # 跟踪
ORANGE_STATE_RECOVERY = 3   # 恢复
ORANGE_STATE_FAIL = 4       # 失败

# Orange Send 协议常量
ORANGE_SEND_HEADER_1 = 0xAA
ORANGE_SEND_HEADER_2 = 0xBB
ORANGE_SEND_TAIL = 0xEE
ORANGE_SEND_FRAME_SIZE = 19  # 2(header) + 4(state) + 4(deta_x) + 4(deta_y) + 4(distance) + 1(tail)

# Frame size limits
MIN_FRAME_SIZE = 4          # SOF + Length + Type + Checksum (no payload)
MAX_FRAME_SIZE = 255        # Limited by Length field (1 byte)
MAX_PAYLOAD_SIZE = 252      # MAX_FRAME_SIZE - SOF - Length - Checksum


@dataclass
class FrameData:
    """Parsed frame data."""
    frame_type: int
    payload: bytes


def xor_checksum(data: bytes) -> int:
    """
    Calculate XOR checksum of data bytes.

    Args:
        data: Bytes to calculate checksum for

    Returns:
        XOR checksum byte
    """
    result = 0
    for byte in data:
        result ^= byte
    return result


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

    # Build payload: error_type(1B) + error_value(2B, little-endian)
    payload = bytes([error_type]) + error_value.to_bytes(2, byteorder='little', signed=True)

    # Build frame content (Type + Payload)
    content = bytes([TYPE_ERROR]) + payload

    # Calculate checksum (Type to Payload)
    checksum = xor_checksum(content)

    # Length = Type(1) + Payload(len) + Checksum(1)
    length = len(content) + 1

    # Complete frame: SOF + Length + Content + Checksum
    frame = bytes([SOF, length]) + content + bytes([checksum])

    return frame


def parse_frame(data: bytes) -> Optional[FrameData]:
    """
    Parse and validate a complete frame.

    Args:
        data: Complete frame bytes (including SOF, Length, Type, Payload, Checksum)

    Returns:
        FrameData if valid, None if invalid

    Note:
        This function validates checksum but does not raise exceptions.
        Invalid frames are silently rejected.
    """
    if len(data) < MIN_FRAME_SIZE:
        return None

    # Check SOF
    if data[0] != SOF:
        return None

    # Get length
    length = data[1]
    expected_size = 2 + length  # SOF(1) + Length(1) + (Type + Payload + Checksum)

    if len(data) != expected_size:
        return None

    # Extract components
    frame_type = data[2]
    payload = data[3:expected_size - 1]  # Exclude checksum
    received_checksum = data[expected_size - 1]

    # Validate checksum
    content = data[2:expected_size - 1]  # Type + Payload
    calculated_checksum = xor_checksum(content)

    if calculated_checksum != received_checksum:
        return None

    return FrameData(frame_type=frame_type, payload=payload)


def parse_zone_payload(payload: bytes) -> Optional[int]:
    """
    Parse zone_id from payload (for ARRIVED, PICK, SET events).

    Args:
        payload: Payload bytes

    Returns:
        zone_id if valid, None if invalid
    """
    if len(payload) != 1:
        return None
    return payload[0]


def parse_error_payload(payload: bytes) -> Optional[tuple]:
    """
    Parse error frame payload.

    Args:
        payload: Payload bytes (error_type + error_value)

    Returns:
        (error_type, error_value) tuple if valid, None if invalid
    """
    if len(payload) != 3:
        return None

    error_type = payload[0]
    error_value = int.from_bytes(payload[1:3], byteorder='little', signed=True)

    return (error_type, error_value)


def build_orange_send_frame(state: int, deta_x: int, deta_y: int, distance_mm: float = 0.0) -> bytes:
    """
    Build a frame matching STM32 orange_send protocol.

    Frame format: AA BB + state(int32) + deta_x(int32) + deta_y(int32) + distance(float32) + EE
    Total: 19 bytes

    Args:
        state: 状态值 (0=IDLE, 1=SEARCH, 2=TRACKING, 3=RECOVERY, 4=FAIL)
        deta_x: X 方向误差 (int32)
        deta_y: Y 方向误差 (int32)
        distance_mm: 目标距离 (float32, mm)

    Returns:
        Complete frame bytes ready to send
    """
    import struct
    frame = bytearray()
    frame.append(ORANGE_SEND_HEADER_1)  # Header 1: 0xAA
    frame.append(ORANGE_SEND_HEADER_2)  # Header 2: 0xBB
    frame.extend(state.to_bytes(4, byteorder='little', signed=True))
    frame.extend(deta_x.to_bytes(4, byteorder='little', signed=True))
    frame.extend(deta_y.to_bytes(4, byteorder='little', signed=True))
    frame.extend(struct.pack('<f', distance_mm))  # float32 little-endian
    frame.append(ORANGE_SEND_TAIL)  # Tail: 0xEE
    return bytes(frame)
