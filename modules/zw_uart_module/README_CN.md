# zw_uart_module

用于香橙派与STM32单片机通信的UART通信模块。

## 用途

提供基于二进制帧的UART通信功能：
- 接收来自STM32的区域事件（ARRIVED、PICK、SET）
- 向STM32发送视觉误差反馈

## 设计原则

**被动状态维护**：模块通过状态机解析STM32事件来维护内部状态。其他模块通过查询接口获取状态——不主动回调其他模块，实现松耦合。

## 依赖

- `pyserial` - 串口通信
- `logging` - 标准库
- `threading` - 标准库

## 使用方法

### 作为模块（通过ModuleManager调用）

在 `main.py` 的 `AUTO_START_MODULES` 中添加：

```python
AUTO_START_MODULES = [
    'zw_uart_module',
    # ... 其他模块
]
```

从其他模块查询状态：

```python
from modules.zw_uart_module import get_current_zone, get_last_arrived_zone, send_error

# 获取当前区域
zone = get_current_zone()

# 发送视觉误差
send_error(0, -3)  # X方向误差，值-3
```

### 直接使用

```python
from zw_uart_module import STM32UartInterface

with STM32UartInterface("/dev/ttyS4", baudrate=115200) as uart:
    uart.send_error(0, -3)  # 发送误差帧

    # 在主循环中查询状态
    while True:
        current = uart.get_current_zone()
        arrived = uart.get_last_arrived_zone()
        pick = uart.get_last_pick_zone()
        print(f"区域: 当前={current}, 到达={arrived}, 抓取={pick}")
        time.sleep(0.1)
```

## 通信协议

### 帧结构

| 字段 | 大小 | 描述 |
|------|------|------|
| SOF | 1 | 帧起始标志，固定 0xAA |
| Length | 1 | 从Type到Checksum的字节数 |
| Type | 1 | 帧类型标识 |
| Payload | 0~252 | 数据负载 |
| Checksum | 1 | Type到Payload所有字节的异或和 |

### 帧类型

| Type | 名称 | 方向 | 负载结构 |
|------|------|------|----------|
| 0x01 | EVENT_ERROR | 香橙派 → STM32 | error_type(1B) + error_value(2B, int16小端) |
| 0x02 | EVENT_ARRIVED_AT_ZONE | STM32 → 香橙派 | zone_id(1B) |
| 0x03 | EVENT_PICK_AT_ZONE | STM32 → 香橙派 | zone_id(1B) |
| 0x04 | EVENT_SET_ZONE | STM32 → 香橙派 | zone_id(1B) |

### 误差类型

| 值 | 描述 |
|----|------|
| 0 | X方向误差 |
| 1 | Y方向误差 |
| 2 | Z方向误差 |
| 3 | 其他误差 |

### 示例帧

```
# SET_ZONE 帧，区域ID为5
AA 03 04 05 01
# SOF=AA, Len=03, Type=04, Payload=05, Checksum=01
# Length = Type(1) + Payload(1) + Checksum(1) = 3

# ERROR 帧：类型=0(X方向)，值=-3
AA 05 01 00 FD FF 03
# SOF=AA, Len=05, Type=01, error_type=00, error_value=FDFF(-3), Checksum=03
# Length = Type(1) + Payload(3) + Checksum(1) = 5
```

## API参考

### STM32UartInterface类

| 方法 | 描述 |
|------|------|
| `start() -> bool` | 连接串口并启动接收线程 |
| `stop()` | 断开连接并清理资源 |
| `get_current_zone() -> int` | 获取当前区域ID |
| `get_last_arrived_zone() -> int` | 获取最近到达的区域ID |
| `get_last_pick_zone() -> int` | 获取最近抓取的区域ID |
| `send_error(type, value) -> bool` | 发送误差帧 |
| `set_log_level(level)` | 设置日志级别 |
| `set_debug_hex(enabled)` | 启用十六进制帧打印 |

### 模块函数

| 函数 | 描述 |
|------|------|
| `init()` | 模块初始化 |
| `start() -> bool` | 模块启动 |
| `loop()` | 模块主循环（空操作） |
| `stop()` | 模块停止 |
| `get_interface() -> STM32UartInterface` | 获取接口实例 |
| `get_current_zone() -> int` | 便捷函数 |
| `get_last_arrived_zone() -> int` | 便捷函数 |
| `get_last_pick_zone() -> int` | 便捷函数 |
| `send_error(type, value) -> bool` | 便捷函数 |

## 文件结构

```
zw_uart_module/
├── __init__.py      # 模块接口
├── uart_driver.py   # STM32UartInterface、FrameParser
├── protocol.py      # 协议常量、帧构造与解析
├── exceptions.py    # 自定义异常
├── README.md        # 英文文档
└── README_CN.md     # 中文文档
```

## 实现说明

### 状态机

帧解析器使用状态机逐字节处理数据：

```
WAITING_SOF → GOT_SOF → GOT_LEN → READING_DATA → (完成帧)
     ↑                                      |
     └──────────── 解析失败/无效帧 ──────────┘
```

### 缓冲区设计

Linux内核TTY子系统已在驱动层实现环形缓冲区。`SerialController.receive_all()`直接从内核缓冲区读取数据，无需在Python层额外实现缓冲区。

### 线程安全

- 状态变量使用 `threading.RLock` 保护
- 发送操作使用 `threading.Lock` 保护串口写入
- 接收线程持续运行，内部错误不会导致线程退出
