from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MockUart:
    def __init__(self, port: str = "mock", baudrate: int = 921600) -> None:
        self._port = port
        self._baudrate = baudrate
        self._connected = True
        self._tx_buffer = bytearray()
        self._rx_buffer = bytearray()
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

    @property
    def in_waiting(self) -> int:
        return len(self._rx_buffer)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockUart connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MockUart disconnected")

    def send(self, data: bytes) -> int:
        self._tx_buffer.extend(data)
        logger.debug("MockUart sent %d bytes", len(data))
        return len(data)

    def receive(self, size: int = 1) -> Optional[bytes]:
        if not self._rx_buffer:
            return None
        chunk = bytes(self._rx_buffer[:size])
        self._rx_buffer = self._rx_buffer[size:]
        return chunk

    def receive_all(self) -> Optional[bytes]:
        if not self._rx_buffer:
            return None
        chunk = bytes(self._rx_buffer)
        self._rx_buffer.clear()
        return chunk

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        self._receiver_callback = callback

    def stop_receiver(self) -> None:
        self._receiver_callback = None
