from __future__ import annotations

from typing import Callable, Optional


class MaixCam2Uart:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        self._port = port
        self._baudrate = baudrate
        self._connected = False

    @property
    def in_waiting(self) -> int:
        return 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        return False

    def disconnect(self) -> None:
        self._connected = False

    def send(self, data: bytes) -> int:
        return 0

    def receive(self, size: int = 1) -> Optional[bytes]:
        return None

    def receive_all(self) -> Optional[bytes]:
        return None

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        pass

    def stop_receiver(self) -> None:
        pass
