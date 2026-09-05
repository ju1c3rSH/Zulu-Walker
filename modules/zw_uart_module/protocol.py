# -*- coding: utf-8 -*-
"""
zw_uart_module - Protocol definitions

Defines protocol constants, frame building, and parsing functions for
UART communication between Orange Pi and STM32.

Protocol Frame Structure (v3.0, EOD2026):
| SOF1 | SOF2 | Length | Type | Payload | CRC16_HI | CRC16_LO |
|  1   |  1   |   1    |  1   |  0~252  |    1     |    1     |

- SOF1: Start of Frame byte 1, fixed 0xAA
- SOF2: Start of Frame byte 2, fixed 0x55
- Length: Bytes from Type to CRC16 (Type + Payload + CRC16)
- Type: Frame type identifier
- Payload: Variable length data
- CRC16: CRC16-CCITT (poly 0x1021, init 0xFFFF) of Type to Payload bytes,
  transmitted HIGH byte first (big-endian). Multi-byte payload fields are
  little-endian signed int16 unless noted.

EOD2026 data-type table (must stay in sync with panzer MCU docs/design.md §2.4):
| data_type | name        | sub_payload                                        |
|-----------|-------------|----------------------------------------------------|
| 0x01      | QR_TASK     | flags(1) + len(1) + ascii(18)                      |
| 0x02      | BOMB_DEV    | error_x(2) + error_y(2) + color_id(1) + flags(1)   |
| 0x03      | TARGET_DEV  | error_x(2) + error_y(2) + ring_id(1) + flags(1)    |
| 0x04      | HOSTAGE_DEV | error_x(2) + error_y(2) + shape_id(1) + flags(1)   |
| 0x05      | BUCKET_DEV  | error_x(2) + error_y(2) + flags(1)                 |
| 0x06      | HOME_DEV    | error_x(2) + error_y(2) + flags(1)                 |

CMD_REQUEST reserved byte (3rd payload byte) carries the expected task-code id:
digit1 for BOMB_DEV, digit2 for TARGET_DEV, digit3 for HOSTAGE_DEV, 0 otherwise.
Vision latches DEV_FLAG_MATCH only when detected id == expected id.
"""

from dataclasses import dataclass
from typing import Optional


# Protocol constants
SOF1 = 0xAA
SOF2 = 0x55

# Frame types
TYPE_ERROR = 0x01               # Bidirectional: error_type(1B) + error_value(2B, int16 LE)
TYPE_HEARTBEAT = 0x15           # [DEPRECATED] Bidirectional: seq + mission_state + visual_state
TYPE_VISUAL_SERVO_DATA = 0x17   # [DEPRECATED] Orange Pi -> MCU: error_x(2B) + error_y(2B) + flags(1B) + state(1B)
TYPE_EMERGENCY_STOP = 0x18      # Bidirectional: reason(1B)

# === Master-Slave protocol v3.0 ===
TYPE_CMD_REQUEST     = 0x20  # Master -> Slave: request data stream subscription
TYPE_CMD_ACK         = 0x21  # Slave -> Master: acknowledge subscription
TYPE_CMD_NACK        = 0x22  # Slave -> Master: reject subscription
TYPE_CMD_STOP        = 0x23  # Master -> Slave: stop data stream
TYPE_DATA_STREAM     = 0x24  # Slave -> Master: streaming data frame

# === Ping-Pong (link heartbeat) ===
TYPE_PING            = 0x25  # Master -> Slave: 保活探测
TYPE_PONG            = 0x26  # Slave -> Master: 保活应答

# NACK reasons
NACK_UNSUPPORTED_TYPE = 0x01
NACK_NOT_READY        = 0x02
NACK_BUSY             = 0x03

# === EOD2026 data stream types (for CMD_REQUEST.data_type) ===
DATA_QR_TASK     = 0x01   # QR task code: flags + len + ascii(18)
DATA_BOMB_DEV    = 0x02   # bomb block alignment (color per digit1)
DATA_TARGET_DEV  = 0x03   # laser target alignment (rings 8/9/10, HIT_CONFIRMED)
DATA_HOSTAGE_DEV = 0x04   # hostage alignment (shape per digit3)
DATA_BUCKET_DEV  = 0x05   # black EOD bucket alignment (bomb placement)
DATA_HOME_DEV    = 0x06   # return-zone alignment (hostage placement)

# Device flags bitmask (bit layout shared by all device streams)
DEV_FLAG_FOUND = 0x01   # bit0 TARGET_FOUND — vision-side latch; error/id fields are 0 when clear
DEV_FLAG_MATCH = 0x02   # bit1 COLOR_MATCH / SHAPE_MATCH — detected id == task-code id
DEV_FLAG_READY = 0x04   # bit2 READY_TO_PICK(0x02) / HIT_CONFIRMED(0x03) / READY_TO_PLACE(0x05/0x06)

# color_id / shape_id encodings (match the QR task-code digits)
ID_NONE = 0
COLOR_RED, COLOR_GREEN, COLOR_BLUE = 1, 2, 3
SHAPE_CYLINDER, SHAPE_CONE, SHAPE_WAIST_DRUM = 1, 2, 3
RING_NONE = 0   # ring_id: 0=no target, else 8/9/10

# Payload sizes for each data type (None = variable length)
# Total TYPE_DATA_STREAM payload size (seq + data_type + sub_payload)
DATA_PAYLOAD_SIZES = {
    DATA_QR_TASK:     20,
    DATA_BOMB_DEV:    6,
    DATA_TARGET_DEV:  6,
    DATA_HOSTAGE_DEV: 6,
    DATA_BUCKET_DEV:  5,
    DATA_HOME_DEV:    5,
}

SUPPORTED_DATA_TYPES = set(DATA_PAYLOAD_SIZES.keys())

DATA_TYPE_NAMES: dict[int, str] = {
    DATA_QR_TASK:     "QR_TASK",
    DATA_BOMB_DEV:    "BOMB_DEV",
    DATA_TARGET_DEV:  "TARGET_DEV",
    DATA_HOSTAGE_DEV: "HOSTAGE_DEV",
    DATA_BUCKET_DEV:  "BUCKET_DEV",
    DATA_HOME_DEV:    "HOME_DEV",
}

# Subscription -> align processor mode (coordinator gating)
DATA_TYPE_MODES: dict[int, str] = {
    DATA_BOMB_DEV:    "BOMB",
    DATA_TARGET_DEV:  "TARGET",
    DATA_HOSTAGE_DEV: "HOSTAGE",
    DATA_BUCKET_DEV:  "BUCKET",
    DATA_HOME_DEV:    "HOME",
}

ERROR_MIN = -5000
ERROR_MAX = 5000

NACK_REASON_NAMES: dict[int, str] = {
    NACK_UNSUPPORTED_TYPE: "UNSUPPORTED_TYPE",
    NACK_NOT_READY:        "NOT_READY",
    NACK_BUSY:             "BUSY",
}


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


# === Master-Slave protocol v3.0 builders ===

def build_cmd_request_frame(data_type: int, min_interval_ms: int = 0) -> bytes:
    """Build TYPE_CMD_REQUEST frame."""
    payload = bytes([data_type, min_interval_ms, 0x00])
    return _build_frame(TYPE_CMD_REQUEST, payload)


def build_cmd_ack_frame(data_type: int, max_freq_hz: int = 60, payload_size: int = 0) -> bytes:
    """Build TYPE_CMD_ACK frame."""
    payload = bytes([data_type, max_freq_hz, payload_size])
    return _build_frame(TYPE_CMD_ACK, payload)


def build_cmd_nack_frame(data_type: int, reason: int) -> bytes:
    """Build TYPE_CMD_NACK frame."""
    payload = bytes([data_type, reason])
    return _build_frame(TYPE_CMD_NACK, payload)


def build_cmd_stop_frame() -> bytes:
    """Build TYPE_CMD_STOP frame (empty payload)."""
    return _build_frame(TYPE_CMD_STOP, b'')


def build_data_stream_frame(seq: int, data_type: int, sub_payload: bytes) -> bytes:
    """Build TYPE_DATA_STREAM frame.

    Args:
        seq: Sequence counter (0-255, wraps)
        data_type: The data stream type being sent
        sub_payload: Type-specific payload bytes (excludes seq + data_type header)
    """
    inner = bytes([seq, data_type]) + sub_payload
    return _build_frame(TYPE_DATA_STREAM, inner)


# === Ping-Pong builders ===

def build_ping_frame(seq: int) -> bytes:
    """Build TYPE_PING frame. Payload = seq(1B)."""
    return _build_frame(TYPE_PING, bytes([seq & 0xFF]))


def build_pong_frame(seq: int) -> bytes:
    """Build TYPE_PONG frame. Payload = seq(1B)，原样回传 MCU。"""
    return _build_frame(TYPE_PONG, bytes([seq & 0xFF]))


# === Master-Slave protocol v3.0 parsers ===

def parse_cmd_request_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_CMD_REQUEST payload -> (data_type, min_interval_ms, reserved)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def parse_cmd_ack_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_CMD_ACK payload -> (data_type, max_freq_hz, payload_size)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]


def parse_cmd_nack_payload(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_CMD_NACK payload -> (data_type, reason)."""
    if len(payload) != 2:
        return None
    return payload[0], payload[1]


def parse_cmd_stop_payload(payload: bytes) -> bool:
    """Validate TYPE_CMD_STOP payload (must be empty)."""
    return len(payload) == 0


def parse_data_stream_header(payload: bytes) -> Optional[tuple]:
    """Parse TYPE_DATA_STREAM header -> (seq, data_type, sub_payload).

    Only extracts the header; the caller must interpret sub_payload
    based on data_type.
    """
    if len(payload) < 2:
        return None
    return payload[0], payload[1], payload[2:]


# === Ping-Pong parsers ===

def parse_ping_payload(payload: bytes) -> Optional[int]:
    """Parse TYPE_PING payload -> seq.  Returns None on invalid length."""
    if len(payload) != 1:
        return None
    return payload[0]


parse_pong_payload = parse_ping_payload  # same format


# === EOD2026 sub-payload builders/parsers ===
# sub_payload = DATA_STREAM payload minus the seq(1)+data_type(1) header.

def clamp_error(value: int) -> int:
    """Clamp a normalized pixel error to the ±5000 protocol range."""
    return max(ERROR_MIN, min(ERROR_MAX, int(value)))


def build_qr_task_sub_payload(decoded: bool, code: str) -> bytes:
    """QR_TASK sub_payload (20B): flags(1) + len(1) + ascii(18, zero-padded).

    flags bit0 = DECODED_VALID. When not decoded, code is ignored and the
    ascii field is all zeros (no-target frame semantics).
    """
    if not decoded or not code:
        return bytes([0, 0]) + b"\x00" * 18
    raw = code.encode("ascii")[:18]
    return bytes([DEV_FLAG_FOUND, len(raw)]) + raw.ljust(18, b"\x00")


def parse_qr_task_sub_payload(data: bytes) -> Optional[dict]:
    """Parse QR_TASK sub_payload -> {'decoded', 'code'}; None on bad length."""
    if len(data) != 20:
        return None
    flags, length = data[0], data[1]
    decoded = bool(flags & DEV_FLAG_FOUND)
    if not decoded or length == 0:
        return {"decoded": False, "code": ""}
    length = min(length, 18)
    return {"decoded": True, "code": data[2:2 + length].decode("ascii", errors="replace")}


def build_device_sub_payload(error_x: int, error_y: int, dev_id: int, flags: int) -> bytes:
    """BOMB/TARGET/HOSTAGE sub_payload (6B).

    error_x/error_y: normalized ±5000 int16 LE (0 when DEV_FLAG_FOUND clear).
    dev_id: color_id / ring_id / shape_id (0 = no target).
    flags: DEV_FLAG_* bitmask.
    """
    if not (flags & DEV_FLAG_FOUND):
        error_x = error_y = dev_id = 0
    return (
        clamp_error(error_x).to_bytes(2, "little", signed=True)
        + clamp_error(error_y).to_bytes(2, "little", signed=True)
        + bytes([dev_id & 0xFF, flags & 0xFF])
    )


def parse_device_sub_payload(data: bytes) -> Optional[dict]:
    """Parse device sub_payload (6B) -> {'error_x','error_y','id','flags'}."""
    if len(data) != 6:
        return None
    return {
        "error_x": int.from_bytes(data[0:2], "little", signed=True),
        "error_y": int.from_bytes(data[2:4], "little", signed=True),
        "id": data[4],
        "flags": data[5],
    }


def build_zone_sub_payload(error_x: int, error_y: int, flags: int) -> bytes:
    """BUCKET/HOME sub_payload (5B): error_x(2) + error_y(2) + flags(1)."""
    if not (flags & DEV_FLAG_FOUND):
        error_x = error_y = 0
    return (
        clamp_error(error_x).to_bytes(2, "little", signed=True)
        + clamp_error(error_y).to_bytes(2, "little", signed=True)
        + bytes([flags & 0xFF])
    )


def parse_zone_sub_payload(data: bytes) -> Optional[dict]:
    """Parse zone sub_payload (5B) -> {'error_x','error_y','flags'}."""
    if len(data) != 5:
        return None
    return {
        "error_x": int.from_bytes(data[0:2], "little", signed=True),
        "error_y": int.from_bytes(data[2:4], "little", signed=True),
        "flags": data[4],
    }


def build_device_stream_frame(seq: int, data_type: int, error_x: int, error_y: int,
                              dev_id: int, flags: int) -> bytes:
    """Convenience: full DATA_STREAM frame for a 6B device type."""
    return build_data_stream_frame(seq, data_type,
                                   build_device_sub_payload(error_x, error_y, dev_id, flags))


def build_zone_stream_frame(seq: int, data_type: int, error_x: int, error_y: int,
                            flags: int) -> bytes:
    """Convenience: full DATA_STREAM frame for a 5B zone type."""
    return build_data_stream_frame(seq, data_type,
                                   build_zone_sub_payload(error_x, error_y, flags))
