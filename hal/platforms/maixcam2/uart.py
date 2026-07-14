from __future__ import annotations

from typing import Callable, Optional

import maix.peripheral.uart


class MaixCam2Uart:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        self._port = port
        self._baudrate = baudrate
        self._uart: maix.peripheral.uart.UART = maix.peripheral.uart.UART()
        self._connected = False
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

    @property
    def in_waiting(self) -> int:
        if not self._connected:
            return 0
        try:
            n = self._uart.available(timeout=0)
            return max(n, 0)
        except Exception:
            return 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected:
            return True
        try:
            self._uart.set_port(self._port)
            self._uart.set_baudrate(self._baudrate)
            self._uart.open()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        try:
            self._uart.close()
        except Exception:
            pass

    def send(self, data: bytes) -> int:
        if not self._connected:
            return 0
        try:
            n = self._uart.write(data)
            return max(n, 0)
        except Exception:
            return 0

    def receive(self, size: int = 1) -> Optional[bytes]:
        if not self._connected:
            return None
        try:
            data = self._uart.read(len=size, timeout=0)
            return bytes(data) if data else None
        except Exception:
            return None

    def receive_all(self) -> Optional[bytes]:
        if not self._connected:
            return None
        try:
            data = self._uart.read(len=-1, timeout=0)
            return bytes(data) if data else None
        except Exception:
            return None

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        self._receiver_callback = callback

        def _wrapped(uart, data):
            if self._receiver_callback:
                try:
                    self._receiver_callback(bytes(data))
                except Exception:
                    pass

        try:
            self._uart.set_received_callback(_wrapped)
        except Exception:
            pass

    def stop_receiver(self) -> None:
        self._receiver_callback = None
        try:
            self._uart.set_received_callback(None)
        except Exception:
            pass
