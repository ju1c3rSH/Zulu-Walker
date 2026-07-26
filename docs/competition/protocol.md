# 视觉-MCU 通信帧结构定义

> 适用项目：Zulu-Walker（Orange Pi 5B + STM32）
> 用途：香橙派（视觉/决策）与 STM32（运动/执行）的 UART 通信帧规范
>
> **内容定义**（帧类型、子命令、标志位等）见同目录 [`protocol_content.md`](protocol_content.md)

---

## 1. 物理层

| 参数 | 值 |
|---|---|
| 接口 | UART |
| 波特率 | 921600 |
| 数据位 | 8 |
| 校验位 | 无 |
| 停止位 | 1 |
| 流控制 | 无 |

---

## 2. 帧结构

```
| SOF1 | SOF2 | Length | Type | Payload  | CRC16_LO | CRC16_HI |
|  1   |  1   |   1    |  1   |  0~252   |    1     |    1     |

SOF1      = 0xAA (Start of Frame byte 1)
SOF2      = 0x55 (Start of Frame byte 2)
Length    = Type(1) + Payload(n) + CRC16(2)
CRC16     = CRC16-CCITT (poly 0x1021, init 0xFFFF)
            覆盖范围：Type 到 Payload，大端序发送
```

最小帧：6 字节（SOF1 + SOF2 + Length + Type + CRC16，Payload=0）
最大帧：258 字节

---

## 3. 校验算法

使用 CRC16-CCITT（多项式 `0x1021`，初始值 `0xFFFF`，无反射，无最终异或），覆盖范围从 Type 到 Payload，大端序发送。

```python
def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc
```

接收端校验：对 `data[3: -2]`（Type + Payload）计算 CRC16，与帧尾 2 字节比较。

---

## 4. 帧约束

| 参数 | 值 | 说明 |
|------|-----|------|
| SOF1 | `0xAA` | 起始字节 1，固定值 |
| SOF2 | `0x55` | 起始字节 2，固定值 |
| Length | 3~255 | Type + Payload + CRC16 总长 |
| Type | 0x00~0xFF | 帧类型标识（见 `protocol_content.md`） |
| Payload | 0~252 字节 | 数据载荷 |
| CRC16 | CRC16-CCITT | 覆盖 Type 到 Payload，大端序 |
| 最小帧长 | 6 字节 | Payload=0 时 |
| 最大帧长 | 258 字节 | 受 Length 字段约束 |

---

## 5. 通用帧构建/解析

### 5.1 构建帧

```python
def build_frame(frame_type: int, payload: bytes) -> bytes:
    content = bytes([frame_type]) + payload
    checksum = crc16_ccitt(content)
    length = len(content) + 2  # Type + Payload + CRC16(2)
    return bytes([SOF1, SOF2, length]) + content + checksum.to_bytes(2, 'big')
```

### 5.2 解析帧

```python
def parse_frame(data: bytes) -> Optional[FrameData]:
    if len(data) < 6 or data[0] != SOF1 or data[1] != SOF2:
        return None
    length = data[2]
    if len(data) != 3 + length:
        return None
    frame_type = data[3]
    payload = data[4:-2]
    if crc16_ccitt(data[3:-2]) != int.from_bytes(data[-2:], 'big'):
        return None
    return FrameData(frame_type=frame_type, payload=payload)
```

---

## 6. 代码映射（帧结构层）

| 协议元素 | Python 常量/函数 | 位置 |
|---|---|---|
| SOF1/SOF2 | `SOF1 = 0xAA`, `SOF2 = 0x55` | `protocol.py` |
| 校验函数 | `crc16_ccitt()` | `protocol.py` |
| 通用构建 | `_build_frame()` | `protocol.py` |
| 通用解析 | `parse_frame()` → `FrameData` | `protocol.py` |
| 帧数据模型 | `class FrameData` | `protocol.py` |

> 帧类型常量、载荷构建/解析函数见 `protocol_content.md` 的代码映射。

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v2.1** | 2026-07-26 | 加固：单字节 SOF → 双字节 SOF (0xAA 0x55)，XOR → CRC16-CCITT |
| **v2.0** | 2026-07-26 | 独立为帧结构定义，内容定义移至 `protocol_content.md` |
| **v1.3** | 2026-07-18 | 前序版本（内容与帧结构共存于本文档） |
