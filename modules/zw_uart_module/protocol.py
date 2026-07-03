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

# Mission synchronization frame types (added for logistics competition)
TYPE_CMD_FROM_MCU = 0x10        # MCU -> Orange Pi: cmd_id(1B) + args
TYPE_STATUS_FROM_VISION = 0x11  # Orange Pi -> MCU: mission_state + visual_state + flags + cargo_count
TYPE_QR_RESULT = 0x12           # Orange Pi -> MCU: len(1B) + ascii QR string
TYPE_COLOR_RESULT = 0x13        # Orange Pi -> MCU: slot_idx + color_id + confidence
TYPE_ACTION_DONE = 0x14         # MCU -> Orange Pi: action_id + result
TYPE_HEARTBEAT = 0x15           # Bidirectional: seq + mission_state + visual_state
TYPE_REQUEST_SYNC = 0x16        # Bidirectional: requested_state
TYPE_VISUAL_SERVO_DATA = 0x17   # Orange Pi -> MCU: error_x(2B) + error_y(2B) + distance(2B) + state(1B)
TYPE_EMERGENCY_STOP = 0x18      # Bidirectional: reason(1B)

# Sub-commands for TYPE_CMD_FROM_MCU
CMD_START_QR = 0x01             # Start QR detection
CMD_START_COLOR_DETECT = 0x02   # No arg
CMD_TRACK_TARGET = 0x03         # arg: color_id(1B)
CMD_TRACK_RING = 0x04           # arg: color_id(1B)
CMD_TRACK_TOP = 0x05            # arg: color_id(1B)
CMD_STOP_VISUAL = 0x06          # No arg

class VisualFlags:
    """Bit flags for STATUS_FROM_VISION."""
    TARGET_FOUND = 0x01
    READY_TO_PICK = 0x02
    READY_TO_PLACE = 0x04
    VISUAL_FAIL = 0x08
    QR_OK = 0x10
    CARGO_CONFIRMED = 0x20
    COLOR_MISMATCH = 0x40

# Action result codes for TYPE_ACTION_DONE
ACTION_OK = 0x00
ACTION_BUSY = 0x01
ACTION_TIMEOUT = 0x02
ACTION_FAIL = 0x03
ACTION_NO_CARGO = 0x04

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


# ===== Mission synchronization helpers =====

def _build_frame(frame_type: int, payload: bytes) -> bytes:
    """Build a standard SOF/Length/Type/Payload/Checksum frame."""
    content = bytes([frame_type]) + payload
    checksum = xor_checksum(content)
    length = len(content) + 1
    return bytes([SOF, length]) + content + bytes([checksum])


def build_cmd_frame(cmd_id: int, args: bytes = b"") -> bytes:
    """Build TYPE_CMD_FROM_MCU frame."""
    return _build_frame(TYPE_CMD_FROM_MCU, bytes([cmd_id]) + args)


def parse_cmd_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_CMD_FROM_MCU payload -> (cmd_id, args)."""
    if len(payload) < 1:
        return None
    return payload[0], payload[1:]


def build_status_from_vision_frame(
    mission_state: int, visual_state: int, flags: int, cargo_count: int
) -> bytes:
    """Build TYPE_STATUS_FROM_VISION frame."""
    payload = bytes([mission_state, visual_state, flags, cargo_count])
    return _build_frame(TYPE_STATUS_FROM_VISION, payload)


def parse_status_from_vision_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_STATUS_FROM_VISION payload."""
    if len(payload) != 4:
        return None
    return payload[0], payload[1], payload[2], payload[3]


def build_qr_result_frame(qr_str: str) -> bytes:
    """Build TYPE_QR_RESULT frame."""
    data = qr_str.encode('ascii', errors='ignore')
    if len(data) > MAX_PAYLOAD_SIZE - 1:
        raise ValueError(f"QR string too long: {len(data)}")
    payload = bytes([len(data)]) + data
    return _build_frame(TYPE_QR_RESULT, payload)


def parse_qr_result_payload(payload: bytes) -> Optional[str]:
    """Parse TYPE_QR_RESULT payload."""
    if len(payload) < 1:
        return None
    length = payload[0]
    if len(payload) != 1 + length:
        return None
    return payload[1:].decode('ascii', errors='ignore')


def build_color_result_frame(color_id: int, confidence: int) -> bytes:
    """Build TYPE_COLOR_RESULT frame."""
    return _build_frame(TYPE_COLOR_RESULT, bytes([color_id, confidence]))


def parse_color_result_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_COLOR_RESULT payload -> (color_id, confidence)."""
    if len(payload) != 2:
        return None
    return payload[0], payload[1]


def build_action_done_frame(action_id: int, result: int) -> bytes:
    """Build TYPE_ACTION_DONE frame."""
    return _build_frame(TYPE_ACTION_DONE, bytes([action_id, result]))


def parse_action_done_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_ACTION_DONE payload -> (action_id, result)."""
    if len(payload) != 2:
        return None
    return payload[0], payload[1]


def build_heartbeat_frame(seq: int, mission_state: int, visual_state: int) -> bytes:
    """Build TYPE_HEARTBEAT frame."""
    return _build_frame(TYPE_HEARTBEAT, bytes([seq, mission_state, visual_state]))


def parse_heartbeat_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_HEARTBEAT payload -> (seq, mission_state, visual_state)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def build_request_sync_frame(requested_state: int) -> bytes:
    """Build TYPE_REQUEST_SYNC frame."""
    return _build_frame(TYPE_REQUEST_SYNC, bytes([requested_state]))


def parse_request_sync_payload(payload: bytes) -> Optional[int]:
    """Parse TYPE_REQUEST_SYNC payload."""
    if len(payload) != 1:
        return None
    return payload[0]


def build_visual_servo_data_frame(
    error_x: int, error_y: int, distance_mm: int, state: int
) -> bytes:
    """
    Build TYPE_VISUAL_SERVO_DATA frame.
    All 16-bit values are signed little-endian.
    """
    payload = (
        error_x.to_bytes(2, byteorder='little', signed=True)
        + error_y.to_bytes(2, byteorder='little', signed=True)
        + distance_mm.to_bytes(2, byteorder='little', signed=True)
        + bytes([state])
    )
    return _build_frame(TYPE_VISUAL_SERVO_DATA, payload)


def parse_visual_servo_data_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_VISUAL_SERVO_DATA payload -> (error_x, error_y, distance_mm, state)."""
    if len(payload) != 7:
        return None
    error_x = int.from_bytes(payload[0:2], byteorder='little', signed=True)
    error_y = int.from_bytes(payload[2:4], byteorder='little', signed=True)
    distance_mm = int.from_bytes(payload[4:6], byteorder='little', signed=True)
    state = payload[6]
    return error_x, error_y, distance_mm, state


def build_emergency_stop_frame(reason: int) -> bytes:
    """Build TYPE_EMERGENCY_STOP frame."""
    return _build_frame(TYPE_EMERGENCY_STOP, bytes([reason]))


def parse_emergency_stop_payload(payload: bytes) -> Optional[int]:
    """Parse TYPE_EMERGENCY_STOP payload -> reason."""
    if len(payload) != 1:
        return None
    return payload[0]
