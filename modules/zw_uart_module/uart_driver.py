"""
zw_uart_module - Generic UART driver example.

Provides a SerialUartInterface that wraps a framework.hal.interface.Uart
protocol instance and demonstrates the module pattern (init/start/loop/stop).
Users should subclass or replace this with their own protocol.
"""

import logging
import threading
from typing import Optional

from framework.hal.interface import Uart
from .exceptions import UartError


class SerialUartInterface:
    def __init__(self, uart: Uart):
        self._uart = uart
        self._write_lock = threading.Lock()
        self._event_bus = None
        self._logger = logging.getLogger(__name__)

    @property
    def is_connected(self) -> bool:
        return self._uart.is_connected

    def set_event_bus(self, bus):
        self._event_bus = bus

    def start(self) -> bool:
        try:
            if not self._uart.connect():
                self._logger.error("Failed to connect to UART")
                return False
            self._uart.start_receiver(self._on_data_received)
            self._logger.info("SerialUartInterface started")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start UART interface: {e}")
            return False

    def stop(self):
        try:
            self._uart.stop_receiver()
            self._uart.disconnect()
            self._logger.info("SerialUartInterface stopped")
        except Exception as e:
            self._logger.error(f"Error stopping UART interface: {e}")

    def send_raw(self, frame: bytes) -> bool:
        try:
            with self._write_lock:
                if not self._uart.is_connected:
                    return False
                return self._uart.send(frame) == len(frame)
        except Exception as e:
            self._logger.error(f"Failed to send raw frame: {e}")
            return False

    def _on_data_received(self, data: bytes):
        try:
            self._logger.debug(f"Received: {data.hex()}")
            if self._event_bus:
                pass
        except Exception as e:
            self._logger.error(f"Error processing received data: {e}")

    def __enter__(self) -> "SerialUartInterface":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
