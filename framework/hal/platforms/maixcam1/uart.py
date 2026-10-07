from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# MaixCAM (cv181x) 的串口模块是 `maix.uart`（不是 CAM2 的 `maix.peripheral.uart`）。
# 两条路径都试：CAM1 走第一条，CAM2 若复用本文件也能兜住。
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


# 非默认引脚必须先 set_pin_function 复用，否则打不开/无收发。
# 依据 MaixPy docs/doc/zh/peripheral/uart.md「常见代码错误」：不设引脚功能直接
# `uart.UART("/dev/ttyS1", 115200)` 是官方点名的错误写法。
#   UART1: A19=TX / A18=RX（出厂默认是 GPIOA19/PWM7，非 UART1 功能）
#   UART0: A16/A17 出厂默认已是 UART0_TX/RX，无需设置
# ⚠ TX 与 RX **都要设**，只设 TX 是常见漏项。
_UART_PINMAP = {
    "ttyS0": (("A16", "UART0_TX"), ("A17", "UART0_RX")),
    "ttyS1": (("A19", "UART1_TX"), ("A18", "UART1_RX")),
    "ttyS2": (("B0", "UART2_TX"), ("B1", "UART2_RX")),
}


def _ensure_pinmap(port: str) -> None:
    """把串口引脚复用为对应 UART 功能。失败只告警不抛。

    返回 False 表示复用没设成功（调用方可据此把警告升级）——不抛是因为
    部分固件/引脚组合本来就已默认复用，硬失败会误伤。
    """
    key = ""
    for k in _UART_PINMAP:
        if port.endswith(k):
            key = k
            break
    if not key:
        logger.warning("UART port '%s' 不在已知引脚表，未设置 pinmap", port)
        return False
    try:
        from maix import pinmap
    except (ImportError, ModuleNotFoundError):
        logger.warning("maix.pinmap 不可用：%s 可能未复用，串口将不通", port)
        return False
    ok = True
    for pin, func in _UART_PINMAP[key]:
        try:
            pinmap.set_pin_function(pin, func)
        except Exception as e:
            logger.warning("pinmap %s -> %s 失败: %s", pin, func, e)
            ok = False
    if not ok:
        logger.warning("UART %s pinmap 不完整，串口可能不通", port)
    return ok


class MaixCam1Uart:
    """MaixCAM 串口（对接 panzer_Brain USART1）。

    端口说明（MaixPy 官方引脚表，2026-10-07核对）:

    | 串口   | TX      | RX      | 设备         | 备注                |
    |--------|---------|---------|--------------|---------------------|
    | UART0  | A16     | A17     | /dev/ttyS0   | 系统终端/日志，勿占用 |
    | UART1  | A19     | A18     | /dev/ttyS1   | **推荐自定义通信**   |

    ⚠ UART0 是 maix protocol 与系统日志串口，接线会打断日志/应用退出通道。
      所以 CAM1 一律用 /dev/ttyS1 + A19(TX)/A18(RX)，与 CAM2 的 ttyS4 不同。
    ⚠ A19/A18 出厂默认是 GPIOA19/PWM7，**必须先 pinmap 复用**才通（见
      _ensure_pinmap）。CAM2 的 A21/A22 是板上默认引脚所以无需这一步，这正是
      回移植时极易漏掉的一环。
    ⚠ MaixPy 文档警告: 若做3.3V↔5V 电平转换，务必确保 TX 默认不拉低
      （CAM1 芯片特性，TX 拉低 = 无法开机）。
    """

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._uart = None
        self._connected = False
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

        if not _HAS_UART:
            return
        # 先复用引脚再开串口（A19/A18 出厂默认不是 UART1 功能）
        if not _ensure_pinmap(port):
            logger.warning("UART %s 未能确认引脚复用，链路可能不通", port)
        try:
            self._uart = _maix_uart.UART(port, baudrate)
            self._connected = True
        except TypeError:
            # 老固件可能只认关键字
            try:
                self._uart = _maix_uart.UART(port=port, baudrate=baudrate)
                self._connected = True
            except Exception as e:
                logger.warning("UART init failed (port=%s, baud=%d): %s",
                               port, baudrate, e)
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
        # 回调模式与 read 混用会读出错（MaixPy UART FAQ 第6 条明确警告
        # "设置了 callback 就不要再用 read"）。当前 uart_driver 只用回调，
        # 这里加守卫堵住"两个模式同时打开"的口子。
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
            # MaixPy 回调签名是 (uart_instance, data)
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
