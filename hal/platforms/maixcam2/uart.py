from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import maix.peripheral.uart as _maix_uart
    _HAS_UART = True
except (ImportError, ModuleNotFoundError):
    try:
        from maix import uart as _maix_uart
        _HAS_UART = True
    except (ImportError, ModuleNotFoundError):
        _HAS_UART = False
        logger.warning("maix UART module not available, UART will be a stub")


class MaixCam2Uart:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        self._port = port
        self._baudrate = baudrate
        self._uart: Optional[_maix_uart.UART] = None
        self._connected = False
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

        if not _HAS_UART:
            return

        try:
            self._uart = _maix_uart.UART(port=port, baudrate=baudrate)
            self._connected = True
        except Exception as e:
            logger.warning("UART init failed (port=%s, baud=%d): %s", port, baudrate, e)
            self._uart = None
            self._connected = False

    @property
    def in_waiting(self) -> int:
        if not self._connected or self._uart is None:
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
        if not _HAS_UART:
            return False
        if self._uart is None:
            try:
                self._uart = _maix_uart.UART(port=self._port, baudrate=self._baudrate)
                self._connected = True
                return True
            except Exception:
                return False
        return False

    def disconnect(self) -> None:
        self._connected = False
        if self._uart is not None:
            try:
                self._uart.close()
            except Exception:
                pass

    def send(self, data: bytes) -> int:
        if not self._connected or self._uart is None:
            return 0
        try:
            n = self._uart.write(data)
            return max(n, 0)
        except Exception:
            return 0

    def receive(self, size: int = 1) -> Optional[bytes]:
        if not self._connected or self._uart is None:
            return None
        try:
            data = self._uart.read(len=size, timeout=0)
            return bytes(data) if data else None
        except Exception:
            return None

    def receive_all(self) -> Optional[bytes]:
        if not self._connected or self._uart is None:
            return None
        try:
            data = self._uart.read(len=-1, timeout=0)
            return bytes(data) if data else None
        except Exception:
            return None

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        if self._uart is None:
            return
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
        if self._uart is not None:
            try:
                self._uart.set_received_callback(None)
            except Exception:
                pass
