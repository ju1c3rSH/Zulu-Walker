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
| SOF  | Length | Type | Payload  | Checksum |
|  1   |   1    |  1   |  0~252   |    1     |

SOF       = 0xAA (Start of Frame)
Length    = Type(1) + Payload(n) + Checksum(1)
Checksum  = Type 到 Payload 所有字节的异或（XOR）
```

最小帧：4 字节（SOF + Length + Type + Checksum，Payload=0）
最大帧：255 字节

---

## 3. 校验算法

Checksum 为从 Type 到 Payload 所有字节的按位异或（XOR）。

```python
def xor_checksum(data: bytes) -> int:
    result = 0
    for byte in data:
        result ^= byte
    return result
```

接收端校验：对 `data[2: -1]`（Type + Payload）计算 XOR，与帧尾 Checksum 比较。

---

## 4. 帧约束

| 参数 | 值 | 说明 |
|------|-----|------|
| SOF | `0xAA` | 起始字节，固定值 |
| Length | 1~255 | Type + Payload + Checksum 总长 |
| Type | 0x00~0xFF | 帧类型标识（见 `protocol_content.md`） |
| Payload | 0~252 字节 | 数据载荷 |
| Checksum | XOR | 类型到载荷的 XOR |
| 最小帧长 | 4 字节 | Payload=0 时 |
| 最大帧长 | 255 字节 | 受 Length 字段约束 |

---

## 5. 通用帧构建/解析

### 5.1 构建帧

```python
def build_frame(frame_type: int, payload: bytes) -> bytes:
    content = bytes([frame_type]) + payload
    checksum = xor_checksum(content)
    length = len(content) + 1
    return bytes([SOF, length]) + content + bytes([checksum])
```

### 5.2 解析帧

```python
def parse_frame(data: bytes) -> Optional[FrameData]:
    if len(data) < 4 or data[0] != SOF:
        return None
    length = data[1]
    if len(data) != 2 + length:
        return None
    frame_type = data[2]
    payload = data[3:-1]
    if xor_checksum(data[2:-1]) != data[-1]:
        return None
    return FrameData(frame_type=frame_type, payload=payload)
```

---

## 6. 代码映射（帧结构层）

| 协议元素 | Python 常量/函数 | 位置 |
|---|---|---|
| SOF | `SOF = 0xAA` | `protocol.py` |
| 校验函数 | `xor_checksum()` | `protocol.py` |
| 通用构建 | `_build_frame()` | `protocol.py` |
| 通用解析 | `parse_frame()` → `FrameData` | `protocol.py` |
| 帧数据模型 | `class FrameData` | `protocol.py` |

> 帧类型常量、载荷构建/解析函数见 `protocol_content.md` 的代码映射。

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v2.0** | 2026-07-26 | 独立为帧结构定义，内容定义移至 `protocol_content.md` |
| **v1.3** | 2026-07-18 | 前序版本（内容与帧结构共存于本文档） |
