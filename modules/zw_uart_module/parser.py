# -*- coding: utf-8 -*-
"""
zw_uart_module - C 协议解析器的忠实移植。

1:1 对应 F:\\2026工科赛\\serial_protocol\\protocol.c 的 Protocol_ParserFeed
四态状态机（panzer / panzer_Brain / 顶层 serial_protocol 三份 C 副本逐行一致）:

- WAIT_SOF1: 等 0xAA
- WAIT_SOF2: 等 0x55；收到非 0xAA 回 WAIT_SOF1，连续 0xAA 保持（AA AA 55 重同步）
- WAIT_LEN:  Length < MIN_LEN 或 > body_capacity → length_error + 复位
- WAIT_BODY: 攒满 Length 字节后校验 CRC（覆盖 body 前 len-2 字节，高字节在前）；
  成功回调 type/payload(body[1])/plen(len-3)；失败 crc_error + 整帧丢弃；
  两种结局都复位状态机（旧 Python 端的逐字节 pop 重扫不采用，以 C 为准）。

统计计数与 C Protocol_Stats_t 一致: rx_bytes / valid_frames / crc_errors / length_errors。
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional

from .protocol import (
    SOF1, SOF2,
    MIN_FRAME_SIZE,
    MAX_PAYLOAD_SIZE,
    FrameData,
    crc16_ccitt,
)

PROTOCOL_MIN_LEN = 3        # protocol.h PROTOCOL_MIN_LEN: Type + CRC16
PROTOCOL_MAX_BODY = 255     # protocol.h PROTOCOL_MAX_BODY: Length 字节上限
DEFAULT_BODY_CAPACITY = PROTOCOL_MAX_BODY


class ParserState(IntEnum):
    """Protocol_ParserState_t"""
    WAIT_SOF1 = 0
    WAIT_SOF2 = 1
    WAIT_LEN = 2
    WAIT_BODY = 3


@dataclass
class ParserStats:
    """Protocol_Stats_t"""
    rx_bytes: int = 0
    valid_frames: int = 0
    crc_errors: int = 0
    length_errors: int = 0


@dataclass
class _Accumulated:
    """坏帧保留原始 body 供上层日志（C 库无此需求，Python 端用于调试）。"""
    body: bytearray = field(default_factory=bytearray)


class ProtocolParser:
    """C Protocol_Parser_t 的 Python 等价物。

    on_frame(ctx 省略) 不用回调制——feed_byte/feed_block 返回解析出的
    FrameData 列表，便于 RX 线程批量分发; on_crc_error 可选保留坏帧日志。
    """

    def __init__(self, body_capacity: int = DEFAULT_BODY_CAPACITY,
                 on_crc_error: Optional[Callable[[bytes], None]] = None):
        if body_capacity < PROTOCOL_MIN_LEN:
            raise ValueError(f"body_capacity too small: {body_capacity}")
        self._body_capacity = min(body_capacity, PROTOCOL_MAX_BODY)
        self._on_crc_error = on_crc_error
        self.stats = ParserStats()
        self.state = ParserState.WAIT_SOF1
        self.len = 0
        self.idx = 0
        self._body = bytearray(self._body_capacity)
        self._pending = _Accumulated()

    # ---- lifecycle ----

    def reset(self) -> None:
        """Protocol_ParserReset"""
        self.state = ParserState.WAIT_SOF1
        self.len = 0
        self.idx = 0

    @property
    def body_capacity(self) -> int:
        return self._body_capacity

    # ---- feeding ----

    def feed_byte(self, byte: int) -> Optional[FrameData]:
        """Protocol_ParserFeed 单字节喂入; 校验通过返回 FrameData。"""
        self.stats.rx_bytes += 1

        if self.state == ParserState.WAIT_SOF1:
            if byte == SOF1:
                self.state = ParserState.WAIT_SOF2
            return None

        if self.state == ParserState.WAIT_SOF2:
            if byte == SOF2:
                self.state = ParserState.WAIT_LEN
            elif byte != SOF1:
                # 连续 0xAA 时保持等待 SOF2, 实现 AA AA 55 重同步
                self.state = ParserState.WAIT_SOF1
            return None

        if self.state == ParserState.WAIT_LEN:
            if byte < PROTOCOL_MIN_LEN or byte > self._body_capacity:
                self.stats.length_errors += 1
                self.reset()
                return None
            self.len = byte
            self.idx = 0
            self._pending.body.clear()
            self.state = ParserState.WAIT_BODY
            return None

        # WAIT_BODY
        self._body[self.idx] = byte & 0xFF
        self._pending.body.append(byte & 0xFF)
        self.idx += 1
        if self.idx < self.len:
            return None

        frame = self._finalize()
        self.reset()
        return frame

    def feed_block(self, data: bytes) -> List[FrameData]:
        """Protocol_ParserFeedBlock 整块喂入, 等价于逐字节 Feed。"""
        frames = []
        for byte in data:
            frame = self.feed_byte(byte)
            if frame is not None:
                frames.append(frame)
        return frames

    # ---- internals ----

    def _finalize(self) -> Optional[FrameData]:
        """CRC 校验（覆盖 body 前 len-2 字节, 高字节在前）并产出 FrameData。"""
        length = self.len
        calculated_crc = crc16_ccitt(bytes(self._body[:length - 2]))
        received_crc = (self._body[length - 2] << 8) | self._body[length - 1]

        if calculated_crc != received_crc:
            self.stats.crc_errors += 1
            if self._on_crc_error is not None:
                try:
                    self._on_crc_error(bytes(self._pending.body))
                except Exception:
                    pass
            return None

        self.stats.valid_frames += 1
        payload_length = length - 3
        return FrameData(
            frame_type=self._body[0],
            payload=bytes(self._body[1:1 + payload_length]),
        )


# === 便捷函数：从原始字节流中解析出全部合法帧（PC 测试用）===

def parse_stream(data: bytes, body_capacity: int = DEFAULT_BODY_CAPACITY) -> List[FrameData]:
    """把一段可能含噪声/多帧的字节流喂入解析器，返回全部合法帧。"""
    return ProtocolParser(body_capacity).feed_block(data)
