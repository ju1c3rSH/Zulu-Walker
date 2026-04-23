# -*- coding: utf-8 -*-
"""
zw_uart_module - UART Communication Module for STM32

Provides binary frame-based UART communication between Orange Pi and STM32
microcontroller. Implements state synchronization and visual error feedback.

Key Features:
- Thread-safe state queries (zone information)
- Binary protocol with XOR checksum
- State machine frame parsing
- Background receive thread

Protocol:
| SOF | Length | Type | Payload | Checksum |
|  1  |   1    |  1   |  0~252  |    1     |

Frame Types:
- 0x01 EVENT_ERROR (TX): error_type + error_value
- 0x02 EVENT_ARRIVED_AT_ZONE (RX): zone_id
- 0x03 EVENT_PICK_AT_ZONE (RX): zone_id
- 0x04 EVENT_SET_ZONE (RX): zone_id
"""

import os
import sys

# Add module directory to path for internal imports
module_dir = os.path.dirname(__file__)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from .uart_driver import STM32UartInterface
from .protocol import (
    SOF,
    TYPE_ERROR, TYPE_ARRIVED, TYPE_PICK, TYPE_SET,
    ERROR_TYPE_X, ERROR_TYPE_Y, ERROR_TYPE_Z, ERROR_TYPE_OTHER,
    ORANGE_STATE_IDLE, ORANGE_STATE_SEARCH, ORANGE_STATE_TRACKING,
    ORANGE_STATE_RECOVERY, ORANGE_STATE_FAIL,
    FrameData, build_error_frame, parse_frame, build_orange_send_frame
)
from .exceptions import (
    UartError, InvalidFrameError, ChecksumError, ParameterError
)

# Module-level instance
_uart_interface: STM32UartInterface = None
_running: bool = False


def init():
    """Module initialization (called by ModuleManager)."""
    global _uart_interface

    print("[zw_uart_module] Initializing...")

    # Create STM32UartInterface instance
    _uart_interface = STM32UartInterface()

    print("[zw_uart_module] Initialized successfully")


def start() -> bool:
    """Module start (called by ModuleManager)."""
    global _uart_interface, _running

    if _uart_interface is None:
        print("[zw_uart_module] Error: Module not initialized")
        return False

    print("[zw_uart_module] Starting...")

    if _uart_interface.start():
        _running = True
        print("[zw_uart_module] Started successfully")
        return True
    else:
        print("[zw_uart_module] Failed to start")
        return False


def loop():
    """Module main loop (called by ModuleManager).

    This module is passive - state is updated by background receiver thread.
    Nothing to do in main loop.
    """
    pass


def stop():
    """Module stop (called by ModuleManager)."""
    global _uart_interface, _running

    print("[zw_uart_module] Stopping...")

    _running = False

    if _uart_interface is not None:
        _uart_interface.stop()

    print("[zw_uart_module] Stopped")


# === Direct access functions ===

def get_interface() -> STM32UartInterface:
    """Get the STM32UartInterface instance."""
    return _uart_interface


def get_current_zone() -> int:
    """Get current zone ID (convenience function)."""
    if _uart_interface:
        return _uart_interface.get_current_zone()
    return 0


def get_last_arrived_zone() -> int:
    """Get last arrived zone ID (convenience function)."""
    if _uart_interface:
        return _uart_interface.get_last_arrived_zone()
    return 0


def get_last_pick_zone() -> int:
    """Get last pick zone ID (convenience function)."""
    if _uart_interface:
        return _uart_interface.get_last_pick_zone()
    return 0


def send_error(error_type: int, error_value: int) -> bool:
    """Send error frame to STM32 (convenience function)."""
    if _uart_interface:
        return _uart_interface.send_error(error_type, error_value)
    return False


def send_orange_frame(state: int, deta_x: int, deta_y: int) -> bool:
    """Send orange frame to STM32 (convenience function)."""
    if _uart_interface:
        return _uart_interface.send_orange_frame(state, deta_x, deta_y)
    return False


def add_pick_callback(callback):
    """Add callback for PICK events (convenience function)."""
    if _uart_interface:
        _uart_interface.add_pick_callback(callback)


def remove_pick_callback(callback):
    """Remove PICK event callback (convenience function)."""
    if _uart_interface:
        _uart_interface.remove_pick_callback(callback)
