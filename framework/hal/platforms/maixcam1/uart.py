from __future__ import annotations

import logging
from typing import Callable, Optional

from utils.boot_timing import boot_stage

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


# 非默认引脚必须先 set_pin_function 复用，否则打不开/无收发
# （MaixPy docs/doc/zh/peripheral/uart.md「常见代码错误」点名过这种写法）。
# TX 与 RX 都要设，只设 TX 是常见漏项。
#
# ttyS0 故意不设 pinmap：A16/A17 出厂默认就是 UART0_TX/UART0_RX
# （pinmap.md:42），对已是默认功能的引脚再调 set_pin_function 多此一举，
# 且部分固件会返回非 ERR_NONE 造成误告警。真正需要复用的只有非默认引脚的
# 串口（UART1 A19/A18、UART2 B0/B1）。
_UART_PINMAP = {
    # 出厂默认即UART0 功能，不要调 set_pin_function（见上）
    "ttyS0": (),
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
    pins = _UART_PINMAP[key]
    if not pins:
        # ttyS0: 出厂默认功能，跳过（不是"失败"）
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
    """MaixCAM 串口（对接 panzer_Brain USART1）。

    端口说明（MaixPy 官方引脚表）:

    | 串口   | TX      | RX      | 设备         | 备注                |
    |--------|---------|---------|--------------|---------------------|
    | UART0  | A16     | A17     | /dev/ttyS0   | 本项目在用     |
    | UART1  | A19     | A18     | /dev/ttyS1   | 官方推荐自定义通信   |

    本项目用 UART0（/dev/ttyS0, A16/A17）对接 Brain，接线已固定。A16/A17
    出厂默认即 UART0_TX/UART0_RX（pinmap.md:42），无需 pinmap —— 见
    _ensure_pinmap 里 ttyS0 故意留空的说明。

    UART0 的三个已知限制（MaixPy uart.md「MaixCAM/MaixCAM-Pro 串口使用注意点」）:

    1. 开机打印日志直到出现 `serial ready`。Brain 侧应丢弃开机段数据，
       否则可能把启动日志当协议帧。
    2. TX(A16) 是启动模式检测引脚，拉低 = 无法开机。做 3.3V↔5V 电平转换
       电路时务必确保 TX 默认不拉低（保持浮空或用电平转换芯片）。若上电起不来，
       先查 A16 是否被拉低。
    3. UART0 同时是系统默认 maix protocol 串口：占用后 MaixVision IDE 的调试
       通道会与本链路抢线，台架调试改用 SSH（`ssh root@<设备IP>`）或断开 IDE。
       注意 `app.need_exit()` 走的是 maix protocol 通道，占用 UART0 时应用可能
       无法通过 IDE 正常停止（急停可直接断电或用设备上的物理方式）。
    """

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._uart = None
        self._connected = False
        self._receiver_callback: Optional[Callable[[bytes], None]] = None

        if not _HAS_UART:
            return
        # 非默认引脚需先复用（A16/A17 出厂已是 UART0 功能，函数内会跳过）
        if not _ensure_pinmap(port):
            logger.warning("UART %s 未能确认引脚复用，链路可能不通", port)
        # UART0 抢线计时段：系统 Maix Comm Protocol 原本占着 /dev/serial0，
        #   本项目改用 UART0 后，这里 open 会把系统服务踢掉、触发一遍退出握手才把
        #   串口交回来 —— 这一段是固件行为, 代码侧消不掉, 只能测出来评估是否
        #   值得换 UART1(A19/A18)。分 pinmap / open 两段计时便于区分是抢线握手
        #   还是串口配置本身。
        with boot_stage("uart.pinmap"):
            pass  # _ensure_pinmap 已在上面同步调用, 这里只借它划出时间边界
        try:
            with boot_stage("uart.open"):
                self._uart = _maix_uart.UART(port, baudrate)
            self._connected = True
        except TypeError:
            # 老固件可能只认关键字
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
        # 回调模式与 read 混用会读出错（MaixPy UART FAQ 第6 条明确警告
        # "设置了 callback 就不要再用 read"）。这里加守卫堵住"两个模式同时打开"的口子。
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
