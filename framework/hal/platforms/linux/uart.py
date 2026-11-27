from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Optional

import serial

logger = logging.getLogger(__name__)

# The literal that asks for the host's own serial port instead of a named one.
AUTO_PORT = "auto"

# The board serial devices of the MaixCAM platforms: they are not host ports
# and must not be picked by the automatic selection.
_EXCLUDED_LINUX_PORTS = frozenset({"/dev/ttyS3", "/dev/ttyS4"})


def resolve_serial_port(port: str = "") -> Optional[str]:
    """Resolve the serial port to open.

    An empty port or the literal "auto" selects the first usable host serial
    port; an explicit port is returned unchanged.  None means there is no
    port to open, which the caller treats as a non-fatal condition.
    """
    wanted = (port or "").strip()
    if wanted and wanted.lower() != AUTO_PORT:
        return wanted
    return _first_host_serial_port()


def _first_host_serial_port() -> Optional[str]:
    """Return the first host serial port, preferring USB devices."""
    try:
        from serial.tools import list_ports
    except Exception as exc:
        logger.info("Serial port enumeration unavailable: %s", exc)
        return None
    try:
        ports = list(list_ports.comports())
    except Exception as exc:
        logger.info("Serial port enumeration failed: %s", exc)
        return None
    usb: list[str] = []
    acm: list[str] = []
    other: list[str] = []
    for info in ports:
        device = str(getattr(info, "device", "") or "")
        if not device or device in _EXCLUDED_LINUX_PORTS:
            continue
        if device.startswith("/dev/ttyUSB"):
            usb.append(device)
        elif device.startswith("/dev/ttyACM"):
            acm.append(device)
        elif device.startswith("/dev/ttyS"):
            other.append(device)
        elif re.match(r"^COM\d+$", device, re.IGNORECASE):
            # COM1 is the classic system port on Windows; the robot link is
            # never on it.
            if int(device[3:]) >= 3:
                other.append(device)
    for group in (usb, acm, other):
        if group:
            chosen = sorted(group)[0]
            logger.info("Auto-selected host serial port: %s", chosen)
            return chosen
    logger.info("No host serial port found")
    return None


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
