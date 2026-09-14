from __future__ import annotations

from typing import Optional

from framework.hal import Machine

from .uart_driver import STM32UartInterface
from .parser import ProtocolParser, ParserState, ParserStats, parse_stream
from .protocol import (
    SOF1, SOF2,
    TYPE_CMD_REQUEST, TYPE_CMD_ACK, TYPE_CMD_NACK,
    TYPE_CMD_STOP, TYPE_DATA_STREAM,
    DATA_EXPLOSIVE, DATA_TARGET, DATA_RESCUE,
    SUPPORTED_DATA_IDS, DATA_ID_NAMES,
    SUB_PAYLOAD_SIZE, DATA_STREAM_PAYLOAD_LENGTH,
    ACK_MAX_FREQ_HZ, ACK_PAYLOAD_SIZE,
    TARGET_ERROR_SCALE, TARGET_ERROR_LIMIT,
    COLOR_NONE, COLOR_RED, COLOR_GREEN, COLOR_BLUE,
    SHAPE_NONE, SHAPE_CYLINDER, SHAPE_CONE, SHAPE_WAIST_DRUM,
    MIN_INTERVAL_MS, SUBSCRIBE_FULL_LEN,
    NACK_UNSUPPORTED_TYPE, NACK_NOT_READY, NACK_PARAM, NACK_UNKNOWNSUB,
    FrameData, parse_frame, crc16_ccitt,
    build_cmd_ack_frame, build_cmd_nack_frame,
    build_data_stream_frame,
    build_explosive_sub_payload, build_target_sub_payload,
    build_rescue_sub_payload,
    parse_explosive_sub_payload, parse_target_sub_payload,
    parse_rescue_sub_payload,
)
from .exceptions import (
    UartError, InvalidFrameError, ChecksumError, ParameterError,
)

_uart_interface: Optional[STM32UartInterface] = None
_running: bool = False


def init(machine: Machine, event_bus=None, **kwargs) -> None:
    global _uart_interface

    _uart_interface = STM32UartInterface(uart=machine.uart)
    if event_bus is not None:
        _uart_interface.set_event_bus(event_bus)


def start() -> bool:
    global _running
    if _uart_interface is None:
        return False
    if _running:
        return True
    if _uart_interface.start():
        _running = True
        return True
    return False


def loop():
    pass


def stop():
    global _running
    _running = False
    if _uart_interface is not None:
        _uart_interface.stop()


def get_interface():
    return _uart_interface
