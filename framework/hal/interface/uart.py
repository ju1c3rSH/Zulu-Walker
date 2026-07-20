from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Uart(Protocol):
    @property
    def in_waiting(self) -> int:
        ...

    @property
    def is_connected(self) -> bool:
        ...

    def connect(self) -> bool:
        ...

    def disconnect(self) -> None:
        ...

    def send(self, data: bytes) -> int:
        ...

    def receive(self, size: int = 1) -> Optional[bytes]:
        ...

    def receive_all(self) -> Optional[bytes]:
        ...

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        ...

    def stop_receiver(self) -> None:
        ...
