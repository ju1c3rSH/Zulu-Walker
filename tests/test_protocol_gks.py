# -*- coding: utf-8 -*-
"""Protocol layer tests — panzer_links_protocol.md v2.1 (USART1 近端视觉)。

向量来源:
- serial_protocol/test/test_protocol.c 的 C 自测向量（CRC/编码/重同步/错误/满载）,
  逐条移植 —— 任何改动破坏与 C 库 (panzer/panzer_Brain 副本) 的互操作都会在此失败;
- panzer_links_protocol.md §6 的 USART1 业务向量（订阅/ACK/数据/STOP）;
- Brain interrupt.c 消费语义的字段偏移对拍（pit@2/flags@3, err@0-1, pos@0/flags@1）。
"""
import pytest

from modules.zw_uart_module.protocol import (
    crc16_ccitt,
    parse_frame,
    build_cmd_ack_frame,
    build_cmd_nack_frame,
    build_cmd_stop_frame,
    build_cmd_laser_frame,
    build_data_stream_frame,
    build_explosive_sub_payload,
    build_target_sub_payload,
    build_rescue_sub_payload,
    parse_explosive_sub_payload,
    parse_target_sub_payload,
    parse_rescue_sub_payload,
    parse_cmd_request_payload,
    DATA_EXPLOSIVE, DATA_TARGET, DATA_RESCUE,
    SUB_PAYLOAD_SIZE, DATA_STREAM_PAYLOAD_LENGTH,
    NACK_UNSUPPORTED_TYPE, NACK_NOT_READY,
)
from modules.zw_uart_module.parser import ProtocolParser, ParserState, parse_stream


def hexs(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


# =========================================================================
# C 库 CRC 向量 (test_protocol.c TestCrcVectors)
# =========================================================================

def test_crc_vectors_from_c():
    assert crc16_ccitt(bytes([0x20, 0x07, 0x00, 0x00])) == 0x361E
    assert crc16_ccitt(bytes([0x20, 0x01, 0x00, 0x00])) == 0x84BE
    assert crc16_ccitt(bytes([0x23])) == 0xF5F1
    assert crc16_ccitt(bytes([0x24, 0x2A, 0x07, 0xDC, 0x05, 0x77, 0x01,
                              0x01, 0x00, 0x88, 0xFF])) == 0xB059


# =========================================================================
# C 库编码向量 (test_protocol.c TestEncode)
# =========================================================================

def test_encode_request_gold_from_c():
    # 3B 请求 (旧式短订阅) — C 库逐字节金向量: AA 55 06 20 07 00 00 36 1E
    from modules.zw_uart_module.protocol import _build_frame
    got = _build_frame(0x20, bytes([0x07, 0x00, 0x00]))
    assert hexs(got) == "AA 55 06 20 07 00 00 36 1E"


def test_encode_stop_gold_from_c():
    assert hexs(build_cmd_stop_frame()) == "AA 55 03 23 F5 F1"


# =========================================================================
# panzer_links_protocol.md §6 USART1 业务向量
# =========================================================================

def test_md6_subscribe_explosive():
    # Brain→CAM 订阅 0x01, 50ms: AA 55 06 20 01 32 00 E7 49
    from modules.zw_uart_module.protocol import _build_frame
    got = _build_frame(0x20, bytes([0x01, 0x32, 0x00]))
    assert hexs(got) == "AA 55 06 20 01 32 00 E7 49"


def test_md6_ack_explosive():
    assert hexs(build_cmd_ack_frame(DATA_EXPLOSIVE)) == "AA 55 06 21 01 3C 0A 13 B8"


def test_md6_data_stream_explosive():
    # CAM→Brain: seq=0x2A, cx=1234, pit=2, found=1
    sub = build_explosive_sub_payload(1234, 2, True)
    frame = build_data_stream_frame(0x2A, DATA_EXPLOSIVE, sub)
    assert hexs(frame) == "AA 55 0D 24 2A 01 D2 04 02 01 00 00 00 00 30 49"


def test_md6_stop():
    assert hexs(build_cmd_stop_frame()) == "AA 55 03 23 F5 F1"


# =========================================================================
# v2.1 子载荷语义 (Brain interrupt.c 消费偏移对拍)
# =========================================================================

def test_explosive_sub_payload_layout():
    # Brain: pit@2, flags@3 (bit0); cx u16LE @0-1; 未 found 全零
    sub = build_explosive_sub_payload(1234, 2, True)
    assert len(sub) == SUB_PAYLOAD_SIZE == 8
    assert sub[0:2] == b"\xD2\x04"           # 1234 LE
    assert sub[2] == 2 and sub[3] == 0x01
    assert sub[4:] == b"\x00" * 4
    zero = build_explosive_sub_payload(1234, 2, False)
    assert zero == b"\x00" * 8               # 不透传假值


def test_target_sub_payload_layout_and_clamp():
    # Brain: err i16LE@0-1, flags@2; 文档 UART4 同系数向量: +0.25 -> C4 09, -0.25 -> 3C F6
    assert build_target_sub_payload(2500, True)[0:2] == b"\xC4\x09"
    assert build_target_sub_payload(-2500, True)[0:2] == b"\x3C\xF6"
    assert build_target_sub_payload(20000, True)[0:2] == b"\x10\x27"    # clamp +10000
    assert build_target_sub_payload(-20000, True)[0:2] == b"\xF0\xD8"   # clamp -10000
    assert build_target_sub_payload(500, False) == b"\x00" * 8
    assert parse_target_sub_payload(build_target_sub_payload(-2500, True)) == \
        {"error_fixed": -2500, "found": True}


def test_rescue_sub_payload_layout():
    # Brain: pos@0, flags@1
    sub = build_rescue_sub_payload(3, True)
    assert sub[0] == 3 and sub[1] == 0x01 and sub[2:] == b"\x00" * 6
    assert build_rescue_sub_payload(3, False) == b"\x00" * 8


def test_data_stream_payload_always_10b():
    frame = build_data_stream_frame(0, DATA_RESCUE, b"")   # 短子载荷补零
    assert len(frame) == 2 + 1 + (1 + 10 + 2)              # SOF + LEN + body
    assert frame[4:4 + 2] == b"\x00\x03"
    parsed = parse_frame(frame)
    assert parsed is not None and len(parsed.payload) == DATA_STREAM_PAYLOAD_LENGTH


# =========================================================================
# 订阅帧解析 (5B 完整 / 短帧重订阅)
# =========================================================================

def test_subscription_full_frame_has_codes():
    did, interval, codes = parse_cmd_request_payload(bytes([0x01, 0x32, 0x02, 0x01, 0x03]))
    assert (did, interval) == (0x01, 0x32)
    assert codes == (2, 1, 3)


def test_subscription_short_frame_keeps_codes():
    did, interval, codes = parse_cmd_request_payload(bytes([0x02, 0x32, 0x00]))
    assert codes is None
    did, interval, codes = parse_cmd_request_payload(bytes([0x03, 0x1E]))
    assert codes is None and interval == 0x1E


def test_subscription_empty_payload_invalid():
    assert parse_cmd_request_payload(b"") is None


# =========================================================================
# C 状态机移植对拍 (test_protocol.c TestDecodeStream/TestResync/TestErrors/TestRoundTrip)
# =========================================================================

C_STREAM_FRAME = bytes.fromhex("AA55 0D24 2A 07DC 0577 0101 00 88FF B059".replace(" ", ""))


def test_c_stream_frame_block_feed():
    parser = ProtocolParser()
    frames = parser.feed_block(C_STREAM_FRAME)
    assert len(frames) == 1
    assert frames[0].frame_type == 0x24
    assert frames[0].payload == C_STREAM_FRAME[4:14]
    assert parser.stats.rx_bytes == len(C_STREAM_FRAME)
    assert parser.stats.valid_frames == 1


def test_c_stream_frame_byte_feed_and_le_fields():
    parser = ProtocolParser()
    frames = []
    for b in C_STREAM_FRAME:
        f = parser.feed_byte(b)
        if f is not None:
            frames.append(f)
    assert len(frames) == 1 and frames[0].payload == C_STREAM_FRAME[4:14]
    # C 库 ReadS16LE 对拍: percent_error_x=1500, ball_cm_x100=375, ball_val=-120
    payload = frames[0].payload
    assert int.from_bytes(payload[2:4], "little", signed=True) == 1500
    assert int.from_bytes(payload[4:6], "little", signed=True) == 375
    assert int.from_bytes(payload[8:10], "little", signed=True) == -120


def test_c_resync_garbage_and_double_sof():
    garbage_then_frame = bytes([0x00, 0x12, 0x55, 0xAA, 0x55, 0x03, 0x23, 0xF5, 0xF1])
    double_sof_frame = bytes([0xAA, 0xAA, 0x55, 0x03, 0x23, 0xF5, 0xF1])

    parser = ProtocolParser()
    frames = parser.feed_block(garbage_then_frame)
    assert len(frames) == 1 and frames[0].frame_type == 0x23

    parser = ProtocolParser()
    frames = parser.feed_block(double_sof_frame)
    assert len(frames) == 1 and frames[0].frame_type == 0x23


def test_c_crc_error_drops_and_recovers():
    frame = bytearray(build_cmd_stop_frame())
    parser = ProtocolParser()
    frame[4] ^= 0xFF   # 破坏 CRC 低字节
    assert parser.feed_block(bytes(frame)) == []
    assert parser.stats.crc_errors == 1
    frame[4] = 0xF5
    frames = parser.feed_block(bytes(frame))
    assert len(frames) == 1 and frames[0].frame_type == 0x23


def test_c_length_errors_and_capacity():
    parser = ProtocolParser(body_capacity=8)   # C 测试: 小容量只收 Length<=8
    assert parser.feed_block(b"\xAA\x55\x02\x23") == []
    assert parser.stats.length_errors == 1
    assert parser.feed_block(b"\xAA\x55\x09\x23") == []
    assert parser.stats.length_errors == 2
    # 复位后仍能正常收帧
    frames = parser.feed_block(build_cmd_stop_frame())
    assert len(frames) == 1


def test_c_252b_round_trip():
    payload = bytes(i & 0xFF for i in range(252))
    from modules.zw_uart_module.protocol import _build_frame
    frame = _build_frame(0x24, payload)
    assert len(frame) == 258 and frame[2] == 0xFF
    frames = parse_stream(frame)
    assert len(frames) == 1 and frames[0].payload == payload


def test_back_to_back_frames_and_noise():
    f1 = build_cmd_ack_frame(DATA_EXPLOSIVE)
    f2 = build_data_stream_frame(1, DATA_RESCUE, build_rescue_sub_payload(1, True))
    stream = b"\x99\x88" + f1 + b"\xAA\xAA\xAA" + f2 + f1
    frames = parse_stream(stream)
    assert [f.frame_type for f in frames] == [0x21, 0x24, 0x21]


def test_payload_containing_aa55():
    # payload 内含 AA 55 不应破坏定界（状态机按 Length 攒帧, 不扫内容）
    payload = bytes([0x01, 0xAA, 0x55, 0x03, 0x23, 0x02])
    from modules.zw_uart_module.protocol import _build_frame
    frame = _build_frame(0x24, payload)
    frames = parse_stream(frame + frame)
    assert len(frames) == 2
    assert all(f.payload == payload for f in frames)


def test_crc_error_callback_receives_body():
    seen = []
    parser = ProtocolParser(on_crc_error=seen.append)
    frame = bytearray(build_cmd_stop_frame())
    frame[4] ^= 0xFF
    parser.feed_block(bytes(frame))
    # body = Type + CRC16(2B), STOP 帧长 3
    assert len(seen) == 1
    assert len(seen[0]) == 3 and seen[0][0] == 0x23


# =========================================================================
# v2.3 附录 0x25 CMD_LASER (从→主, Brain 写 PA4, 无应答)
# =========================================================================

def test_laser_frame_gold_vectors():
    # 向量同 docs/competition/gks_laser_ctrl_addendum_v2.3.md §测试向量
    assert hexs(build_cmd_laser_frame(True)) == \
        "AA 55 0B 25 01 00 00 00 00 00 00 00 8F 91"
    assert hexs(build_cmd_laser_frame(False)) == \
        "AA 55 0B 25 00 00 00 00 00 00 00 00 C8 42"


def test_laser_frame_payload_layout():
    # payload 恒 8B: [on_off, rsv x7]; Brain 侧只读 offset 0
    for on in (True, False):
        frame = build_cmd_laser_frame(on)
        parsed = parse_frame(frame)
        assert parsed is not None and parsed.frame_type == 0x25
        assert parsed.payload == bytes([1 if on else 0]) + b"\x00" * 7
