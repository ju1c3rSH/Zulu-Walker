from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import serial

logger = logging.getLogger(__name__)


class LinuxUart:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._receiver_thread: Optional[threading.Thread] = None
        self._receiver_callback: Optional[Callable[[bytes], None]] = None
        self._running = False
        self._connected = False

    @property
    def in_waiting(self) -> int:
        if self._serial and self._serial.is_open:
            return self._serial.in_waiting
        return 0

    @property
    def is_connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=0,
            )
            self._connected = True
            logger.info("UART connected: %s @ %d", self._port, self._baudrate)
            return True
        except serial.SerialException as e:
            logger.error("UART connect failed: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        self.stop_receiver()
        self._connected = False
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception as e:
                logger.error("UART close error: %s", e)

    def send(self, data: bytes) -> int:
        if not self.is_connected:
            return 0
        with self._lock:
            return self._serial.write(data)

    def receive(self, size: int = 1) -> Optional[bytes]:
        if not self.is_connected:
            return None
        try:
            return self._serial.read(size)
        except serial.SerialException:
            return None

    def receive_all(self) -> Optional[bytes]:
        if not self.is_connected:
            return None
        n = self.in_waiting
        if n == 0:
            return None
        try:
            return self._serial.read(n)
        except serial.SerialException:
            return None

    def start_receiver(self, callback: Callable[[bytes], None]) -> None:
        self._receiver_callback = callback
        if self._receiver_thread and self._receiver_thread.is_alive():
            return
        self._running = True
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

    def _receiver_loop(self) -> None:
        try:
            from utils.cpu_affinity import bind_current_thread
            bind_current_thread("uart_receiver")
        except ImportError:
            pass

        while self._running and self.is_connected:
            data = self.receive_all()
            if data and self._receiver_callback:
                try:
                    self._receiver_callback(data)
                except Exception as e:
                    logger.error("UART receiver callback error: %s", e)
            else:
                time.sleep(0.001)

    def stop_receiver(self) -> None:
        self._running = False
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1)
