# -*- coding: utf-8 -*-
"""
zw_uart_module - 2026 工科赛协议定义 (panzer_links_protocol.md v2.1)

本分支 = 近端 MaixCAM 视觉从站（USART1，Brain 主站订阅制）。

帧结构（与 F:\\2026工科赛\\serial_protocol\\protocol.c 逐字节一致）:
| SOF1 | SOF2 | Length | Type | Payload | CRC16_HI | CRC16_LO |
|  1   |  1   |   1    |  1   |  0~252  |    1     |    1     |

- SOF1/SOF2: 0xAA / 0x55
- Length = Type + Payload + CRC16 = n + 3，有效范围 3~255
- CRC16-CCITT: poly 0x1021, init 0xFFFF, 不反射, 无终异或,
  计算 Type..Payload 末字节, 高字节在前（大端）发送
- Payload 内多字节字段一律小端（有符号为补码）

帧类型（四源核对：md v2.1 ↔ serial_protocol/protocol.c ↔ panzer/GetData
↔ panzer_Brain/Middlewares/serial_protocol + interrupt.c 实际用法）:
| 0x20 | CMD_REQUEST | 主→从 | [data_id, min_interval_ms, 爆色, 靶色, 形状] |
| 0x21 | CMD_ACK     | 从→主 | [data_id, max_freq_hz, payload_size]        |
| 0x22 | CMD_NACK    | 从→主 | [data_id, reason]                           |
| 0x23 | CMD_STOP    | 主→从 | 空                                          |
| 0x24 | DATA_STREAM | 从→主 | [seq, data_id, sub_payload(8B)]             |

订阅语义（USART1 v1.3 扩展）:
- 5B 完整订阅帧携带任务码（爆色/靶色/形状 随订阅下发）;
- 短帧(1~4B) = 重订阅，沿用上次任务码（幂等，不得清视觉锁存）;
- 换 data_id 或换任务码才允许清锁存（reset_task_latch）。
- ACK 字面 [data_id, 60, 10]（Brain on_cam1 不消费 ACK，照发保协议一致性）。

data_id 表（USART1 近端只支持 0x01~0x03，其余回 NACK [did, 0x01]）:
| 0x01 | DATA_EXPLOSIVE | [cx u16LE, pit 0~3, flags bit0=found, rsv x4]     |
| 0x02 | DATA_TARGET    | [err i16LE, flags bit0=found, rsv x5]             |
| 0x03 | DATA_RESCUE    | [pos 0~3, flags bit0=found, rsv x6]               |

0x02 err = horizontal_error_normalized_fixed = int((cx - w/2)/(w/2) * 10000)，
clamp ±1 后定点化（ Brain TRACKING_ERROR_FIXED_SCALE = 10000 透传底盘闭环）。
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# Protocol constants
SOF1 = 0xAA
SOF2 = 0x55

# === Frame types (v2.1, 无 PING/PONG) ===
TYPE_CMD_REQUEST  = 0x20  # Master -> Slave: 订阅/切任务
TYPE_CMD_ACK      = 0x21  # Slave -> Master: 订阅应答
TYPE_CMD_NACK     = 0x22  # Slave -> Master: 拒绝
TYPE_CMD_STOP     = 0x23  # Master -> Slave: 停止推送
TYPE_DATA_STREAM  = 0x24  # Slave -> Master: 数据流

# NACK reasons
NACK_UNSUPPORTED_TYPE = 0x01
NACK_NOT_READY        = 0x02

# === data_id 表（本端支持的三个视觉任务）===
DATA_EXPLOSIVE = 0x01   # 爆炸物坑位识别
DATA_TARGET    = 0x02   # 靶标横向误差
DATA_RESCUE    = 0x03   # 人质坑位识别

SUPPORTED_DATA_IDS = (DATA_EXPLOSIVE, DATA_TARGET, DATA_RESCUE)

DATA_ID_NAMES: dict[int, str] = {
    DATA_EXPLOSIVE: "EXPLOSIVE",
    DATA_TARGET:    "TARGET",
    DATA_RESCUE:    "RESCUE",
}

# data_id -> sub_payload 恒 8B; DATA_STREAM payload 恒 10B = seq + data_id + 8B
SUB_PAYLOAD_SIZE = 8
DATA_STREAM_PAYLOAD_LENGTH = 10

# ACK 字面值（main.py / 协议文档 §2: [did, 60, 10]）
ACK_MAX_FREQ_HZ = 60
ACK_PAYLOAD_SIZE = DATA_STREAM_PAYLOAD_LENGTH

# 0x02 误差定点系数（Brain interrupt.h TRACKING_ERROR_FIXED_SCALE）
TARGET_ERROR_SCALE = 10000
TARGET_ERROR_LIMIT = TARGET_ERROR_SCALE   # 归一化 clamp ±1

# 任务码编码（二维码任务码数字域）
COLOR_NONE = 0
COLOR_RED, COLOR_GREEN, COLOR_BLUE = 1, 2, 3
SHAPE_NONE = 0
SHAPE_CYLINDER, SHAPE_CONE, SHAPE_WAIST_DRUM = 1, 2, 3   # yuan_zhu / yuan_tai / yao_gu

# 订阅帧长度: 5 = 完整(带任务码), 1~4 = 重订阅(沿用旧任务码)
SUBSCRIBE_FULL_LEN = 5
MIN_INTERVAL_MS = 30   # push_ok 下限（main.py: max(interval, 30)）

# Frame size limits (protocol.h: PROTOCOL_MIN_LEN/MAX_PAYLOAD/MAX_FRAME)
MIN_FRAME_SIZE = 6          # SOF1 + SOF2 + Length + Type + CRC16 (空 Payload)
MAX_FRAME_SIZE = 258        # SOF(2) + Length(1) + Body(255)
MAX_PAYLOAD_SIZE = 252      # Length 上限 255: Type + Payload + CRC16


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
    """Build a standard SOF1/SOF2/Length/Type/Payload/CRC16 frame.

    Length = Type + Payload + CRC16 = n + 3 (protocol.c Protocol_EncodeFrame).
    """
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


# === Frame builders (slave -> master) ===

def build_cmd_ack_frame(data_id: int, max_freq_hz: int = ACK_MAX_FREQ_HZ,
                        payload_size: int = ACK_PAYLOAD_SIZE) -> bytes:
    """Build TYPE_CMD_ACK frame — v2.1 近端字面值 [did, 60, 10]。"""
    payload = bytes([data_id, max_freq_hz, payload_size])
    return _build_frame(TYPE_CMD_ACK, payload)


def build_cmd_nack_frame(data_id: int, reason: int) -> bytes:
    """Build TYPE_CMD_NACK frame [did, reason]。"""
    payload = bytes([data_id, reason])
    return _build_frame(TYPE_CMD_NACK, payload)


def build_cmd_stop_frame() -> bytes:
    """Build TYPE_CMD_STOP frame（空载荷; Brain Cam1_Stop 下发, 本端只收）。"""
    return _build_frame(TYPE_CMD_STOP, b"")


def build_data_stream_frame(seq: int, data_id: int, sub_payload: bytes) -> bytes:
    """Build TYPE_DATA_STREAM frame.

    Args:
        seq: Sequence counter (0-255, wraps)
        data_id: The data stream id being sent (0x01/0x02/0x03)
        sub_payload: Type-specific payload bytes (excludes seq + data_id header),
                     不足 8B 补零、超出截断（协议恒 8B 子载荷）。
    """
    sub = bytes(sub_payload)[:SUB_PAYLOAD_SIZE].ljust(SUB_PAYLOAD_SIZE, b"\x00")
    inner = bytes([seq & 0xFF, data_id]) + sub
    return _build_frame(TYPE_DATA_STREAM, inner)


# === Subscription frame parsing (master -> slave) ===

def parse_cmd_request_payload(payload: bytes) -> Optional[Tuple[int, int, Optional[Tuple[int, int, int]]]]:
    """Parse TYPE_CMD_REQUEST payload -> (data_id, min_interval_ms, codes|None).

    codes 为 (爆色, 靶色, 形状) 三元组，仅完整 5B 订阅帧携带；
    短帧(1~4B) 返回 codes=None 表示重订阅沿用旧任务码。
    空 payload 返回 None。
    """
    if not payload:
        return None
    data_id = payload[0]
    min_interval_ms = payload[1] if len(payload) >= 2 else 0
    if len(payload) >= SUBSCRIBE_FULL_LEN:
        codes = (payload[2], payload[3], payload[4])
    else:
        codes = None
    return data_id, min_interval_ms, codes


def parse_cmd_stop_payload(payload: bytes) -> bool:
    """Validate TYPE_CMD_STOP payload (must be empty)."""
    return len(payload) == 0


def parse_data_stream_header(payload: bytes) -> Optional[Tuple[int, int, bytes]]:
    """Parse TYPE_DATA_STREAM header -> (seq, data_id, sub_payload)."""
    if len(payload) < 2:
        return None
    return payload[0], payload[1], payload[2:]


# === Sub-payload builders (8B each, 与 Brain interrupt.c 消费偏移逐字段对齐) ===

def clamp_error_fixed(value: int) -> int:
    """Clamp a fixed-point normalized error to ±10000 (归一化 clamp ±1)。"""
    return max(-TARGET_ERROR_LIMIT, min(TARGET_ERROR_LIMIT, int(value)))


def _int16_le_bytes(value: int) -> bytes:
    """int16 两字节补码小端（main.py pack_signed16_le 等价实现）。"""
    return (value & 0xFFFF).to_bytes(2, byteorder="little")


def build_explosive_sub_payload(cx: int, pit: int, found: bool) -> bytes:
    """0x01 DATA_EXPLOSIVE sub_payload (8B)。

    cx: 爆炸物像素 x（Brain 仅监控不使用）; pit: 坑位 1~3, 0=未定位;
    found: bit0 TARGET_FOUND。未 found 时 cx/pit 归零（不透传假值）。
    """
    if not found:
        cx = 0
        pit = 0
    cx = max(0, min(0xFFFF, int(cx)))
    return _int16_le_bytes(cx) + bytes([pit & 0xFF, 0x01 if found else 0x00]) + b"\x00" * 4


def build_target_sub_payload(error_fixed: int, found: bool) -> bytes:
    """0x02 DATA_TARGET sub_payload (8B)。

    error_fixed: i16 LE = 归一化横向误差 × 10000（clamp ±1）;
    found=0 时误差字节为 0 占位（Brain 不透传假误差）。
    """
    if not found:
        error_fixed = 0
    error_fixed = clamp_error_fixed(error_fixed)
    return _int16_le_bytes(error_fixed) + bytes([0x01 if found else 0x00]) + b"\x00" * 5


def build_rescue_sub_payload(pos: int, found: bool) -> bytes:
    """0x03 DATA_RESCUE sub_payload (8B): pos 1~3, 0=未定位。"""
    if not found:
        pos = 0
    return bytes([pos & 0xFF, 0x01 if found else 0x00]) + b"\x00" * 6


# === Sub-payload parsers (PC 测试/对拍用) ===

def parse_explosive_sub_payload(data: bytes) -> Optional[dict]:
    """0x01 sub_payload -> {'cx', 'pit', 'found'}; Brain 取 pit@2, flags@3。"""
    if len(data) != SUB_PAYLOAD_SIZE:
        return None
    return {
        "cx": int.from_bytes(data[0:2], "little"),
        "pit": data[2],
        "found": bool(data[3] & 0x01),
    }


def parse_target_sub_payload(data: bytes) -> Optional[dict]:
    """0x02 sub_payload -> {'error_fixed', 'found'}; Brain 取 err@0-1 i16LE, flags@2。"""
    if len(data) != SUB_PAYLOAD_SIZE:
        return None
    return {
        "error_fixed": int.from_bytes(data[0:2], "little", signed=True),
        "found": bool(data[2] & 0x01),
    }


def parse_rescue_sub_payload(data: bytes) -> Optional[dict]:
    """0x03 sub_payload -> {'pos', 'found'}; Brain 取 pos@0, flags@1。"""
    if len(data) != SUB_PAYLOAD_SIZE:
        return None
    return {
        "pos": data[0],
        "found": bool(data[1] & 0x01),
    }


NACK_REASON_NAMES: dict[int, str] = {
    NACK_UNSUPPORTED_TYPE: "UNSUPPORTED_TYPE",
    NACK_NOT_READY:        "NOT_READY",
}
