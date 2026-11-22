from __future__ import annotations

import logging
from typing import Callable, Optional

from utils.boot_timing import boot_stage

logger = logging.getLogger(__name__)

# MaixCAM (cv181x) exposes UART as ``maix.uart``, not CAM2's
# ``maix.peripheral.uart``. Try both paths: CAM1 takes the first, and CAM2 can
# reuse this module via the second.
try:
    from maix import uart as _maix_uart
    _HAS_UART = True
except (ImportError, ModuleNotFoundError):
    try:
        import maix.peripheral.uart as _maix_uart
        _HAS_UART = True
    except (ImportError, ModuleNotFoundError):
        _maix_uart = None
        _HAS_UART = False
        logger.warning("maix UART module not available, UART will be a stub")


# Non-default pins must be muxed with set_pin_function or the port will not
# open or transfer data (called out in MaixPy docs/doc/zh/peripheral/uart.md).
# Set both TX and RX; setting only TX is a common oversight.
#
# ttyS0 deliberately has no pinmap: A16/A17 default to UART0_TX/UART0_RX
# (pinmap.md:42), so calling set_pin_function on already-default pins is
# redundant and some firmware returns non-ERR_NONE, causing false warnings.
# Only non-default ports need muxing (UART1 A19/A18, UART2 B0/B1).
_UART_PINMAP = {
    # UART0 by default; do not call set_pin_function (see above).
    "ttyS0": (),
    "ttyS1": (("A19", "UART1_TX"), ("A18", "UART1_RX")),
    "ttyS2": (("B0", "UART2_TX"), ("B1", "UART2_RX")),
}


def _ensure_pinmap(port: str) -> None:
    """Mux the port's pins to their UART function; warn on failure instead of
    raising.

    Return False when muxing could not be set so the caller can escalate the
    warning. Do not raise: some firmware/pin combinations are already muxed by
    default and a hard failure would be wrong.
    """
    key = ""
    for k in _UART_PINMAP:
        if port.endswith(k):
            key = k
            break
    if not key:
        logger.warning("UART port '%s' 不在已知引脚表，未设置 pinmap", port)
        return False
    pins = _UART_PINMAP[key]
    if not pins:
        # ttyS0: pins already serve UART by default, skip (not a failure).
        logger.warning("[CAM1] UART %s 使用出厂默认引脚功能(A16/A17)，跳过 pinmap", port)
        return True
    try:
        from maix import pinmap
    except (ImportError, ModuleNotFoundError):
        logger.warning("maix.pinmap 不可用：%s 可能未复用，串口将不通", port)
        return False
    ok = True
    for pin, func in pins:
        try:
            pinmap.set_pin_function(pin, func)
        except Exception as e:
            logger.warning("pinmap %s -> %s 失败: %s", pin, func, e)
            ok = False
    if not ok:
        logger.warning("UART %s pinmap 不完整，串口可能不通", port)
    return ok


class MaixCam1Uart:
    """MaixCAM UART link to the panzer_Brain USART1.

    Port map (MaixPy official pin table):

    | Port  | TX  | RX  | Device       | Notes                      |
    |-------|-----|-----|--------------|----------------------------|
    | UART0 | A16 | A17 | /dev/ttyS0   | used by this project       |
    | UART1 | A19 | A18 | /dev/ttyS1   | recommended for custom use |

    This project uses UART0 (/dev/ttyS0, A16/A17) to talk to the Brain with a
    fixed wiring. A16/A17 default to UART0_TX/UART0_RX (pinmap.md:42), so no
    pinmap is needed; see the empty ttyS0 entry in _ensure_pinmap.

    Three known UART0 limitations (MaixPy uart.md, "MaixCAM/MaixCAM-Pro
    serial notes"):

    1. Boot logs print until ``serial ready`` appears. The Brain must discard
       the boot segment or it may treat startup logs as protocol frames.
    2. TX (A16) is also the boot-mode detect pin; holding it low prevents
       booting. In a 3.3V<->5V level shifter, keep TX from being pulled low
       (leave it floating or use a level-shifter IC). If the board will not
       boot, check whether A16 is held low.
    3. UART0 is also the system's default maix protocol port: once claimed,
       the MaixVision IDE debug channel contends for the same lines. Use SSH
       (``ssh root@<device-ip>``) for bench debugging or disconnect the IDE.
       ``app.need_exit()`` travels over the maix protocol channel, so while
       UART0 is occupied the app may not stop normally from the IDE (use a
       physical power cut for an emergency stop).
    """

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._uart = None
        self._connected = False
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

        if not _HAS_UART:
            return
        # Mux non-default pins first (A16/A17 already default to UART0; the
        # helper skips them).
        if not _ensure_pinmap(port):
            logger.warning("UART %s 未能确认引脚复用，链路可能不通", port)
        # UART0 line-contention timing: the system Maix Comm Protocol owns
        # /dev/serial0. Once this project claims UART0, open kicks the system
        # service off and triggers an exit handshake before releasing the
        # port. That is firmware behavior and cannot be removed in code; it is
        # measured to decide whether switching to UART1 (A19/A18) is worth it.
        # Timing pinmap and open separately distinguishes handshake contention
        # from port configuration itself.
        with boot_stage("uart.pinmap"):
            pass  # _ensure_pinmap was called above; this only marks a time boundary
        try:
            with boot_stage("uart.open"):
                self._uart = _maix_uart.UART(port, baudrate)
            self._connected = True
        except TypeError:
            # Older firmware may accept only keywords.
            try:
                with boot_stage("uart.open(kw)"):
                    self._uart = _maix_uart.UART(port=port, baudrate=baudrate)
                self._connected = True
            except Exception as e:
                logger.warning("UART init failed (port=%s, baud=%d): %s",
                               port, baudrate, e)
                self._uart = None
                self._connected = False
        except Exception as e:
            logger.warning("UART init failed (port=%s, baud=%d): %s",
                           port, baudrate, e)
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
        if not _HAS_UART or self._uart is not None:
            return False
        try:
            self._uart = _maix_uart.UART(self._port, self._baudrate)
            self._connected = True
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        self._connected = False
        if self._uart is not None:
            try:
                self._uart.close()
            except Exception:
                pass
            self._uart = None

    def send(self, data: bytes) -> int:
        if not self._connected or self._uart is None:
            return 0
        try:
            n = self._uart.write(data)
            return max(n, 0) if n else 0
        except Exception:
            return 0

    def receive(self, size: int = 1) -> Optional[bytes]:
        if not self._connected or self._uart is None:
            return None
        # Mixing callback mode with read gives wrong results (MaixPy UART FAQ
        # item 6 warns: "do not use read once a callback is set"). Guard
        # against having both modes active at once.
        if self._receiver_callback is not None:
            return None
        try:
            data = self._uart.read(len=size, timeout=0)
            return bytes(data) if data else None
        except Exception:
            return None

    def receive_all(self) -> Optional[bytes]:
        if not self._connected or self._uart is None:
            return None
        if self._receiver_callback is not None:
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
            # MaixPy callback signature is (uart_instance, data).
            if self._receiver_callback:
                try:
                    self._receiver_callback(bytes(data))
                except Exception:
                    pass

        try:
            self._uart.set_received_callback(_wrapped)
        except Exception as e:
            logger.warning("set_received_callback failed: %s", e)

    def stop_receiver(self) -> None:
        self._receiver_callback = None
        if self._uart is not None:
            try:
                self._uart.set_received_callback(None)
            except Exception:
                pass
