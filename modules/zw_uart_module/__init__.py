from __future__ import annotations

from typing import Optional

from framework.hal import Machine

from .uart_driver import STM32UartInterface
from .protocol import (
    SOF1, SOF2,
    TYPE_ERROR,
    TYPE_CMD_REQUEST, TYPE_CMD_ACK, TYPE_CMD_NACK,
    TYPE_CMD_STOP, TYPE_DATA_STREAM,
    ERROR_TYPE_X, ERROR_TYPE_Y, ERROR_TYPE_Z, ERROR_TYPE_OTHER,
    DATA_LINE_POSITION, DATA_TARGET_POSITION,
    DATA_TARGET_COUNT, DATA_DETECTION_STATUS, DATA_ALL_TARGETS,
    NACK_UNSUPPORTED_TYPE, NACK_NOT_READY, NACK_BUSY,
    FrameData, build_error_frame, parse_frame,
    build_cmd_request_frame, build_cmd_ack_frame,
    build_cmd_nack_frame, build_cmd_stop_frame,
    build_data_stream_frame,
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
