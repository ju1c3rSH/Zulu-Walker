# Zulu-Walker 主从机通信协议 v3.0

> 本文档定义 MaixCAM2（从机）与 MSPM0G3519（主机）之间的 UART 主从通信协议。
> 替代旧的"双向心跳 + 自主推送"模型（v2.1），采用类似 I²C 的主从订阅模式。
>
> 版本: 3.0-draft
> 最后更新: 2026-07-28
> 对应 Python 端实现: `modules/zw_uart_module/`

---

## 目录

1. [背景与设计动机](#1-背景与设计动机)
2. [帧结构（不变）](#2-帧结构不变)
3. [帧类型定义](#3-帧类型定义)
4. [数据流类型](#4-数据流类型)
5. [协议流详解](#5-协议流详解)
6. [逐帧 Payload 定义](#6-逐帧-payload-定义)
7. [MaixCAM2 从机端实现](#7-maixcam2-从机端实现)
8. [MSPM0 主机端实现指南](#8-mspm0-主机端实现指南)
9. [附录：完整帧例](#9-附录完整帧例)

---

## 1. 背景与设计动机

### 1.1 旧协议 v2.1 的问题

| 问题 | 说明 |
|------|------|
| 双向心跳无意义 | MaixCAM2 → MSPM0 发送的心跳帧中填充本机状态，但 MSPM0 **不需要**MaixCAM2 的状态来驱动自身逻辑——MSPM0 才是决策者 |
| 推送时机错位 | MaixCAM2 主动推送视觉伺服数据（`TYPE_VISUAL_SERVO_DATA`），但 MSPM0 可能并不需要该数据，导致带宽浪费 |
| 链路检测不可靠 | "必须定期收到对方心跳"才能确认链路存活 —— 若一方仅有数据发送而无心跳，链路状态会误报为断 |

### 1.2 新协议 v3.0 的设计原则

```
I²C 类比:
  MSPM0 (主机)      MaixCAM2 (从机)
       │                  │
       │──── CMD ────────>│    主机发起所有通信
       │<─── ACK ─────────│    从机只应答、不主动发
       │<─── DATA ────────│    订阅后持续推送（类似 I²C 连续读）
       │──── STOP ───────>│    主机取消订阅
```

1. **主机驱动**：所有通信由 MSPM0 发起。MaixCAM2 仅在收到 CMD 后应答或推送。
2. **订阅-推送**：主机发一次 `CMD_REQUEST` 注册兴趣，从机持续推送数据直至收到 `CMD_STOP`。
3. **帧结构不变**：`SOF + Length + Type + Payload + CRC16` 完全兼容 v2.1，MSPM0 现有解析代码零改动。
4. **向后兼容**：旧帧类型（`TYPE_HEARTBEAT`、`TYPE_VISUAL_SERVO_DATA`）代码保留但不发送，接收端解析不报错。

### 1.3 波特率

| 参数 | 值 |
|------|-----|
| 波特率 | **115200** |
| 数据位 | 8 |
| 校验 | 无 |
| 停止位 | 1 |

> 与 [`mspm0_protocol_impl.md`](./mspm0_protocol_impl.md) 的 UART4 配置完全一致。

### 1.4 旧帧类型（Legacy）处理

以下类型来自旧协议 v2.1 及更早版本（Orange Pi ↔ STM32 时代），在 v3.0 中的状态：

| 旧类型 | 值 | 原用途 | v3.0 状态 |
|--------|-----|-------|-----------|
| `TYPE_ARRIVED` | 0x02 | MCU→OP 到达区域 | **移除，不再使用** |
| `TYPE_PICK` | 0x03 | MCU→OP 拾取请求 | **移除，不再使用** |
| `TYPE_SET` | 0x04 | MCU→OP 设置 | **移除，不再使用** |
| `TYPE_CMD_FROM_MCU` | 0x10 | MCU→OP 命令 | **移除，不再使用** |
| `TYPE_STATUS_FROM_VISION` | 0x11 | OP→MCU 视觉状态 | **移除**（被 DATA_STREAM + DATA_DETECTION_STATUS 替代） |
| `TYPE_QR_RESULT` | 0x12 | OP→MCU QR 结果 | **移除** |
| `TYPE_COLOR_RESULT` | 0x13 | OP→MCU 颜色结果 | **移除** |
| `TYPE_ACTION_DONE` | 0x14 | MCU→OP 动作完成 | **移除，不再使用** |
| `TYPE_HEARTBEAT` | 0x15 | 双向心跳 | **DEPRECATED**，接收端保留解析，不再发送 |
| `TYPE_REQUEST_SYNC` | 0x16 | 同步请求 | **移除** |
| `TYPE_VISUAL_SERVO_DATA` | 0x17 | OP→MCU 伺服数据 | **DEPRECATED**，被 TYPE_DATA_STREAM 替代 |
| `TYPE_EMERGENCY_STOP` | 0x18 | 双向紧急停止 | **保持** |

> 对 MaixCAM2（从机）侧：旧类型的构建函数代码保留但不再调用，解析函数保留向后兼容。
> 对 MSPM0（主机）侧：不应再发送/解析已移除的类型，详见 [§8](#8-mspm0-主机端实现指南)。

### 1.5 关键设计决策

| 决策 | 理由 |
|------|------|
| 去掉 `TYPE_HEARTBEAT` | 链路存活由"定期 CMD"隐含保证。若主机 5s 不发任何 CMD，从机视为断链 |
| `TYPE_VISUAL_SERVO_DATA` 标记为 DEPRECATED | 被通用 `TYPE_DATA_STREAM` + `DATA_LINE_POSITION` 替代 |
| 流模式允许重复帧 | 伺服控制需要同一误差值持续发送，不做去重 |
| 主循环 @60Hz 推送 | 主循环 `MAIN_LOOP_DELAY=0.016`，每次迭代最多发一帧，天然限速 60fps |
| 收到新 `CMD_REQUEST` 隐式切换 | 无需先发 `CMD_STOP`，减少一次往返 → 主机可快速"换挡" |
| 空数据帧带 valid=0 | MSPM0 通过 valid 位区分"无检测"和"断链" |
| 超时只由"最后 CMD"计时 | `_last_cmd_time` 仅在收到 CMD_REQUEST/CMD_STOP 时刷新，**不在流送 DATA_STREAM 时刷新**——否则流送会阻止超时触发 |

---

## 2. 帧结构（不变）

```
| SOF1(0xAA) | SOF2(0x55) | Length(1B) | Type(1B) | Payload(0~252B) | CRC16_HI(1B) | CRC16_LO(1B) |
```

- **Length** = Type(1) + Payload(n) + CRC16(2) = n + 3
- **CRC16** = CRC16-CCITT(poly=0x1021, init=0xFFFF) of [Type + Payload]，大端序
- 最小帧 = 6 字节，最大帧 = 258 字节
- 最大 Payload = 252 字节（Length ≤ 255）

> CRC16 的 Python 参考实现见 `modules/zw_uart_module/protocol.py:crc16_ccitt()`；
> C 参考实现见 [`mspm0_protocol_impl.md`](./mspm0_protocol_impl.md#3-crc16-ccitt-实现)。

---

## 3. 帧类型定义

### 3.1 类型汇总

| 常量 | 值 | 方向 | 说明 | 状态 |
|------|-----|------|------|------|
| `TYPE_ERROR` | `0x01` | 双向 | 错误报告 | **保持** |
| `TYPE_HEARTBEAT` | `0x15` | — | 双向心跳 | **DEPRECATED**，不再发送 |
| `TYPE_VISUAL_SERVO_DATA` | `0x17` | — | 旧视觉伺服数据 | **DEPRECATED**，被 0x24 替代 |
| `TYPE_EMERGENCY_STOP` | `0x18` | 双向 | 紧急停止 | **保持** |
| `TYPE_CMD_REQUEST` | **`0x20`** | Master→Slave | 请求订阅数据流 | **新增** |
| `TYPE_CMD_ACK` | **`0x21`** | Slave→Master | 确认收妥/准备就绪 | **新增** |
| `TYPE_CMD_NACK` | **`0x22`** | Slave→Master | 拒绝/无法执行 | **新增** |
| `TYPE_CMD_STOP` | **`0x23`** | Master→Slave | 停止数据流 | **新增** |
| `TYPE_DATA_STREAM` | **`0x24`** | Slave→Master | 流数据帧 | **新增** |

> 对于 DEPRECATED 类型：Python 端保留解析函数，发送侧不再构建；C 端同样保留解析、不再发送。这样做是保证接入旧固件时不会误报 CRC 错误。

### 3.2 各帧 Payload 长度速查

| Type | Payload 最小 | Payload 最大 | 固定/可变 |
|------|-------------|-------------|-----------|
| `0x01` ERROR | 3 | 3 | 固定 |
| `0x18` EMERGENCY_STOP | 1 | 1 | 固定 |
| `0x20` CMD_REQUEST | 3 | 3 | 固定 |
| `0x21` CMD_ACK | 3 | 3 | 固定 |
| `0x22` CMD_NACK | 2 | 2 | 固定 |
| `0x23` CMD_STOP | 0 | 0 | 固定（空 payload） |
| `0x24` DATA_STREAM | 3 | 252 | 可变（取决于 data_type） |

> DATA_STREAM 的 payload = `seq(1) + data_type(1) + sub_payload(0~250)`，合计 ≤ 252。子 payload 上限 250。详见 §6.5。

---

## 4. 数据流类型

当主机发送 `CMD_REQUEST` 时，通过 `data_type` 字段指定需要订阅的数据类型。

### 4.1 类型定义

```python
# 在 protocol.py 中定义为常量
DATA_LINE_POSITION    = 0x01   # 循迹误差 pe_x/pe_y
DATA_TARGET_POSITION  = 0x02   # 目标（钢珠/钢板）中心坐标
DATA_TARGET_COUNT     = 0x03   # 目标数量
DATA_DETECTION_STATUS = 0x04   # 综合检测状态
DATA_ALL_TARGETS      = 0x05   # 全部目标 bbox 列表
```

### 4.2 Payload 大小与预期帧率

| data_type | 每个 DATA_STREAM payload 大小 | 典型推送频率 |
|-----------|------------------------------|-------------|
| `0x01` | 8 字节 | 60Hz（主循环速率） |
| `0x02` | 8 字节 | 60Hz |
| `0x03` | 3 字节 | 60Hz |
| `0x04` | 5 字节 | 60Hz |
| `0x05` | 可变（3 + N×5，N≤16） | 60Hz |

> **推送频率限制**：MaixCAM2 主循环为 60Hz，因此最大推送速率为 60fps。实际受视觉管道 FPS 制约（30~120fps）。`CMD_ACK.max_freq_hz` 字段将返回 60。

### 4.3 扩展新数据流类型

如需增加新的数据流类型，请按以下步骤操作：

**Step 1: 分配常量**
- 使用 `0x10` ~ `0x1F` 范围的取值（避开当前帧类型区间 `0x01`~`0x24`，为未来帧类型扩展预留空间）
- 在 `protocol.py` 中添加常量，格式：
  ```python
  DATA_YOUR_TYPE = 0x10
  ```
- 避免与帧类型（特别是 `0x20`~`0x24`）冲突

**Step 2: 注册 Payload 大小**
- 在 `protocol.py` 的 `DATA_PAYLOAD_SIZES` 字典中添加条目：
  ```python
  DATA_PAYLOAD_SIZES = {
      DATA_LINE_POSITION: 8,
      DATA_TARGET_POSITION: 8,
      DATA_TARGET_COUNT: 3,
      DATA_DETECTION_STATUS: 5,
      DATA_ALL_TARGETS: None,  # None = 可变长度
      DATA_YOUR_TYPE: N,       # 固定大小或 None
  }
  ```
  `CMD_ACK.payload_size` 会自动从此字典读取。

**Step 3: 构建 payload 函数**
- 在 `protocol.py`/`coordinator.py` 中增加对应 payload 构建函数
- 函数签名统一为 `_build_{name}_payload(latest_data: dict) -> Optional[bytes]`
- payload 长度必须与 `DATA_PAYLOAD_SIZES` 注册一致

**Step 4: 更新本文档**
- 在第 4.1 节和 6.5 节中添加新类型的描述和帧格式表格
- 在第 7.2 节的分发器中添加新分支

**注意事项：**
- 若新增数据依赖于新的视觉管道（processor），需先在 `vision_config.yaml` 中注册该 processor
- payload 最大 252 字节。若变长类型，确保最坏情况不超过此限制
- 向后兼容：主机请求未知类型 → 从机回复 `CMD_NACK(reason=UNSUPPORTED_TYPE)`

---

## 5. 协议流详解

### 5.1 正常订阅-推送流

```
MSPM0 (Master)                MaixCAM2 (Slave: IDLE)
    │                                │
    │  CMD_REQUEST(DATA_LINE_POSITION) │
    │ ─────────────────────────────> │  data_type=0x01
    │                                │  保存 streaming_type，进入 STREAMING
    │  CMD_ACK(0x01, 60, 8)          │  data_type, max_freq_hz, payload_size
    │ <───────────────────────────── │
    │                                │
    │  DATA_STREAM(seq=0, type=0x01, pe_x=...) │  ← 主循环每 tick 发 1 帧
    │ <───────────────────────────── │
    │  DATA_STREAM(seq=1, type=0x01, pe_x=...) │
    │ <───────────────────────────── │
    │  DATA_STREAM(seq=2, type=0x01, pe_x=...) │
    │ <───────────────────────────── │
    │          ...                   │
    │                                │
    │  CMD_STOP                      │
    │ ─────────────────────────────> │  streaming_type=0，进入 IDLE
│  CMD_ACK(0x00, 0, 0)           │  data_type=0x00 表示"已停止"
│ <───────────────────────────── │  主机不等待此 ACK，纯通知性质
```

### 5.2 隐式切换（换挡）

```
    │  正在流送 DATA_LINE_POSITION    │
    │          ...                   │
    │                                │
    │  CMD_REQUEST(DATA_TARGET_POSITION) │  ← 不先发 STOP
    │ ─────────────────────────────> │  隐式停止旧流 → 开始新流
    │  CMD_ACK(0x02, 60, 8)          │
    │ <───────────────────────────── │
    │  DATA_STREAM(seq=0, type=0x02, x=..., y=...) │
    │ <───────────────────────────── │
```

> **理由**：减少一次往返延迟。若主机频繁切换数据需求（例如线循迹中短暂切换查看前方目标），隐式切换更高效。

### 5.3 异常流程

**A. 请求不支持的数据类型**
```
MSPM0 → CMD_REQUEST(0xFF)
MaixCAM2 → CMD_NACK(data_type=0xFF, reason=0x01 UNSUPPORTED_TYPE)
```

**B. 流送中收到 EMERGENCY_STOP**
```
MSPM0 → EMERGENCY_STOP(reason=1)
MaixCAM2 → 立即停止流送 → 进入 IDLE → 设置 SM 错误状态
MSPM0 侧应稍后发送 CMD_REQUEST 重新建立通信
```

**C. 主机发送 CMD_STOP 时从机已 IDLE**
```
MSPM0 → CMD_STOP
MaixCAM2（已在 IDLE，streaming_type == 0）
  → 行为：仍然回复 CMD_ACK(0x00, 0, 0)，不影响任何状态
  → 主机侧：收到 ACK 即认为 STOP 生效，无害
```

**D. ERROR 帧**
```
TYPE_ERROR (0x01) 保持双向能力，用途：
  MaixCAM2 → MSPM0：视觉管道严重故障时发送（例如模型加载失败）
  MSPM0 → MaixCAM2：传感器自检失败时发送
两者均非通信协议常态，仅作调试辅助。从机收到 ERROR 帧后记录日志，不改变流送状态。
```

**E. 从机链路超时**
```
MSPM0（故障/断线）  MaixCAM2
    │                   │
    │   ...5秒沉默...    │
    │                   │  超时检测触发
    │                   │  streaming_type=0, master_linked=False
    │                   │  不再发送任何帧
    │                   │  （等待主机恢复后重新 CMD_REQUEST）
```

> 超时时间：5 秒。在 `coordinator.py` 的 `_CMD_TIMEOUT = 5.0` 定义。
> 重连方式：MSPM0 发送新 `CMD_REQUEST`，从机收到后刷新 `_last_cmd_time` 即可重新连上。

---

## 6. 逐帧 Payload 定义

### 6.1 TYPE_CMD_REQUEST (0x20) — Master → Slave

请求订阅指定数据流。

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | 1 | `data_type` | uint8 | 请求的数据流类型（见 §4） |
| 1 | 1 | `min_interval_ms` | uint8 | 最小帧间隔（0=不限速；建议填 0，由视觉管道 FPS 决定） |
| 2 | 1 | `reserved` | uint8 | 保留（填 0x00） |

总 payload 长度：**3 字节**

```c
// C struct 注释
typedef struct __attribute__((packed)) {
    uint8_t data_type;        // 0x01~0x05
    uint8_t min_interval_ms;  // 0=no limit
    uint8_t reserved;         // must be 0x00
} cmd_request_t;
```

### 6.2 TYPE_CMD_ACK (0x21) — Slave → Master

确认已收到请求，准备开始推送数据流。

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | 1 | `data_type` | uint8 | 确认的数据类型；对 CMD_STOP 回复 0x00 |
| 1 | 1 | `max_freq_hz` | uint8 | 预估最大推送频率（固定 60，匹配主循环速率） |
| 2 | 1 | `payload_size` | uint8 | 每个 DATA_STREAM 帧的 payload 字节数（供 MSPM0 分配接收缓冲区） |

总 payload 长度：**3 字节**

```c
typedef struct __attribute__((packed)) {
    uint8_t data_type;       // 0x01~0x05, or 0x00 for STOP ack
    uint8_t max_freq_hz;     // 60
    uint8_t payload_size;    // bytes per DATA_STREAM frame
} cmd_ack_t;
```

### 6.3 TYPE_CMD_NACK (0x22) — Slave → Master

拒绝请求。

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | 1 | `data_type` | uint8 | 被拒绝的数据类型 |
| 1 | 1 | `reason` | uint8 | 拒绝原因码 |

总 payload 长度：**2 字节**

**拒绝原因码：**

| 值 | 常量 | 说明 |
|----|------|------|
| `0x01` | `NACK_UNSUPPORTED_TYPE` | 不支持所请求的数据类型 |
| `0x02` | `NACK_NOT_READY` | 视觉管道未就绪 |
| `0x03` | `NACK_BUSY` | 从机忙（不可中断） |

```c
typedef struct __attribute__((packed)) {
    uint8_t data_type;
    uint8_t reason;  // 0x01=unsupported, 0x02=not ready, 0x03=busy
} cmd_nack_t;
```

### 6.4 TYPE_CMD_STOP (0x23) — Master → Slave

停止当前数据流。

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| — | 0 | — | — | 空 payload |

Payload 长度：**0 字节**（整帧 = 6 字节）

### 6.5 TYPE_DATA_STREAM (0x24) — Slave → Master

流送帧。通用头 + 类型相关子 payload。

**通用头：**

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 0 | 1 | `seq` | uint8 | 序列号（0→255→0 循环） |
| 1 | 1 | `data_type` | uint8 | 与 CMD_REQUEST 一致的 data_type |

#### 6.5.1 DATA_LINE_POSITION (0x01) 子格式

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 2 | 2 | `pe_x` | int16 LE | 横向误差（±32767） |
| 4 | 2 | `pe_y` | int16 LE | 纵向误差 |
| 6 | 1 | `flags` | uint8 | 视觉标志位 |
| 7 | 1 | `state` | uint8 | 视觉状态机状态 |

整帧 payload = **8 字节**。CMD_ACK.payload_size = 8。

```c
typedef struct __attribute__((packed)) {
    uint8_t  seq;
    uint8_t  data_type;        // 0x01
    int16_t  pe_x;             // LE
    int16_t  pe_y;             // LE
    uint8_t  flags;
    uint8_t  state;
} data_line_position_t;
```

> 与旧 `TYPE_VISUAL_SERVO_DATA`（0x17）的 payload 字段完全一致，增加了一个 seq 头。

#### 6.5.2 DATA_TARGET_POSITION (0x02) 子格式

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 2 | 2 | `x` | int16 LE | 目标中心 X 坐标（像素；0=无效） |
| 4 | 2 | `y` | int16 LE | 目标中心 Y 坐标（像素） |
| 6 | 1 | `confidence` | uint8 | 置信度（0~255，255=1.0） |
| 7 | 1 | `flags` | uint8 | Bit0: valid（1=有效检测）；Bit1~7: 保留 |

整帧 payload = **8 字节**。CMD_ACK.payload_size = 8。

**valid 位语义：**
- `valid=1` → x, y, confidence 有效
- `valid=0` → 视觉管道无检测结果（但链路正常）。MSPM0 应丢弃此帧数据，继续等待有效帧
- 即使 valid=0 也发送，是为了让 MSPM0 确认链路仍存活

```c
#define FLAG_VALID  (1 << 0)

typedef struct __attribute__((packed)) {
    uint8_t  seq;
    uint8_t  data_type;        // 0x02
    int16_t  x;                // LE, pixel
    int16_t  y;                // LE, pixel
    uint8_t  confidence;       // 0-255
    uint8_t  flags;            // bit0=valid
} data_target_position_t;
```

#### 6.5.3 DATA_TARGET_COUNT (0x03) 子格式

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 2 | 1 | `count` | uint8 | 当前帧检测到的目标数（0~255） |

整帧 payload = **3 字节**。CMD_ACK.payload_size = 3。

```c
typedef struct __attribute__((packed)) {
    uint8_t seq;
    uint8_t data_type;   // 0x03
    uint8_t count;       // 0-255
} data_target_count_t;
```

#### 6.5.4 DATA_DETECTION_STATUS (0x04) 子格式

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 2 | 1 | `visual_state` | uint8 | 视觉状态机状态：0=IDLE, 1=SEARCH, 2=TRACKING, 3=RECOVERY, 4=FAIL |
| 3 | 1 | `visual_flags` | uint8 | 视觉标志位 bitfield |
| 4 | 2 | `count` | uint16 LE | 目标计数 |

 整帧 payload = **6 字节**。CMD_ACK.payload_size = 6。

```c
typedef struct __attribute__((packed)) {
    uint8_t  seq;
    uint8_t  data_type;      // 0x04
    uint8_t  visual_state;   // 0-4
    uint8_t  visual_flags;
    uint16_t count;          // LE
} data_detection_status_t;
```

#### 6.5.5 DATA_ALL_TARGETS (0x05) 子格式

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|------|------|------|------|------|
| 2 | 1 | `count` | uint8 | 检测到的目标数 N |
| 3 | N×5 | `targets[]` | 数组 | 每个目标 {x(int16 LE), y(int16 LE), class(uint8)} |

整帧 payload = **3 + N×5** 字节。N 上限 = 16。

> **可变长度注意**：此类型 payload 可变。CMD_ACK 返回 `payload_size=0`。
> MSPM0 端须在解析时通过 `count` 字段动态计算实际长度，无法预先分配固定缓冲区。
> 参见 [§8.4](#84-data_stream-解析) 的解析代码。

```c
typedef struct __attribute__((packed)) {
    int16_t x;       // LE, pixel
    int16_t y;       // LE, pixel
    uint8_t class;   // object class ID
} target_entry_t;

typedef struct __attribute__((packed)) {
    uint8_t        seq;
    uint8_t        data_type;   // 0x05
    uint8_t        count;       // N (0~16)
    target_entry_t targets[];   // N entries
} data_all_targets_t;
```

---

## 7. MaixCAM2 从机端实现

### 7.1 改动文件清单

| 文件 | 改动类型 | 关键内容 |
|------|---------|---------|
| `modules/zw_uart_module/protocol.py` | 修改 | 新增常量、构建器、解析器 |
| `modules/zw_uart_module/events.py` | 修改 | 新增 CmdRequestEvent, CmdStopEvent |
| `modules/zw_uart_module/uart_driver.py` | 修改 | 新增帧类型分发 |
| `modules/zw_uart_module/__init__.py` | 修改 | 导出新增常量 |
| `app/coordinator.py` | 重构 | 删除心跳线程，实现 CMD 处理 + 流送 |

### 7.2 protocol.py 改动

**新增常量：**

```python
# === Master-Slave protocol v3.0 ===
# Frame types
TYPE_CMD_REQUEST     = 0x20
TYPE_CMD_ACK         = 0x21
TYPE_CMD_NACK        = 0x22
TYPE_CMD_STOP        = 0x23
TYPE_DATA_STREAM     = 0x24

# NACK reasons
NACK_UNSUPPORTED_TYPE = 0x01
NACK_NOT_READY        = 0x02
NACK_BUSY             = 0x03

# Data stream types
DATA_LINE_POSITION    = 0x01
DATA_TARGET_POSITION  = 0x02
DATA_TARGET_COUNT     = 0x03
DATA_DETECTION_STATUS = 0x04
DATA_ALL_TARGETS      = 0x05

# Payload sizes for each data type (None = variable length)
# This maps data_type -> sub_payload bytes (seq + data_type excluded).
# Used by CMD_ACK to inform MSPM0 of expected frame size.
# For variable-length types (None), MSPM0 must parse dynamically.
DATA_PAYLOAD_SIZES = {
    DATA_LINE_POSITION: 8,
    DATA_TARGET_POSITION: 8,
    DATA_TARGET_COUNT: 3,
    DATA_DETECTION_STATUS: 5,
    DATA_ALL_TARGETS: None,
}

# Supported data types for validation
SUPPORTED_DATA_TYPES = set(DATA_PAYLOAD_SIZES.keys())

# Total DATA_STREAM frame payload size = 2 (seq + data_type) + sub_payload
# CMD_ACK.payload_size reports sub_payload size only (not including the 2-byte header)
# so that MSPM0 buffer calculation is: total = 2 + payload_size
```

**新增构建器：**

```python
def build_cmd_request_frame(data_type: int, min_interval_ms: int = 0) -> bytes:
    payload = bytes([data_type, min_interval_ms, 0x00])
    return _build_frame(TYPE_CMD_REQUEST, payload)

def build_cmd_ack_frame(data_type: int, max_freq_hz: int, payload_size: int) -> bytes:
    payload = bytes([data_type, max_freq_hz, payload_size])
    return _build_frame(TYPE_CMD_ACK, payload)

def build_cmd_nack_frame(data_type: int, reason: int) -> bytes:
    payload = bytes([data_type, reason])
    return _build_frame(TYPE_CMD_NACK, payload)

def build_cmd_stop_frame() -> bytes:
    return _build_frame(TYPE_CMD_STOP, b'')

def build_data_stream_frame(seq: int, data_type: int, payload: bytes) -> bytes:
    inner = bytes([seq, data_type]) + payload
    return _build_frame(TYPE_DATA_STREAM, inner)
```

**新增解析器：**

```python
def parse_cmd_request_payload(payload: bytes) -> Optional[tuple]:
    """Parse CMD_REQUEST payload -> (data_type, min_interval_ms, reserved)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]

def parse_cmd_ack_payload(payload: bytes) -> Optional[tuple]:
    """Parse CMD_ACK payload -> (data_type, max_freq_hz, payload_size)."""
    if len(payload) != 3:
        return None
    return payload[0], payload[1], payload[2]

def parse_cmd_nack_payload(payload: bytes) -> Optional[tuple]:
    """Parse CMD_NACK payload -> (data_type, reason)."""
    if len(payload) != 2:
        return None
    return payload[0], payload[1]

def parse_cmd_stop_payload(payload: bytes) -> bool:
    """Validate CMD_STOP payload (should be empty)."""
    return len(payload) == 0

def parse_data_stream_payload(payload: bytes) -> Optional[tuple]:
    """Parse DATA_STREAM payload -> (seq, data_type, sub_payload)."""
    if len(payload) < 2:
        return None
    return payload[0], payload[1], payload[2:]
```

### 7.3 events.py 改动

```python
@dataclass
class CmdRequestEvent:
    data_type: int
    min_interval_ms: int

@dataclass
class CmdStopEvent:
    pass  # 无参数，STOP 就是 STOP
```

### 7.4 uart_driver.py 改动

在 `_handle_frame()` 的帧分发中增加新类型的处理分支。同时保留 TYPE_HEARTBEAT 的静默解析（丢弃 payload 但不报错），避免旧固件发来的心跳触发 "Unknown frame type" 警告：

```python
def _handle_frame(self, frame: FrameData):
    if frame.frame_type == TYPE_CMD_REQUEST:
        parsed = parse_cmd_request_payload(frame.payload)
        if parsed is not None and self._event_bus:
            self._event_bus.publish(CmdRequestEvent(parsed[0], parsed[1]))
    elif frame.frame_type == TYPE_CMD_STOP:
        if parse_cmd_stop_payload(frame.payload) and self._event_bus:
            self._event_bus.publish(CmdStopEvent())
    elif frame.frame_type == TYPE_CMD_ACK:
        # 从机不期望收到 CMD_ACK —— 若收到可能说明角色混淆，记录警告
        self._logger.warning(f"Unexpected CMD_ACK received (slave role)")
    elif frame.frame_type == TYPE_CMD_NACK:
        # 从机不期望收到 CMD_NACK —— 同样记录
        self._logger.warning(f"Unexpected CMD_NACK received (slave role)")
    elif frame.frame_type == TYPE_DATA_STREAM:
        # 从机不期望收到 DATA_STREAM —— 记录
        self._logger.warning(f"Unexpected DATA_STREAM received (slave role)")
    elif frame.frame_type == TYPE_HEARTBEAT:
        # 静默丢弃——从机不再处理心跳。保持解析以维持 stats 准确
        pass
    elif frame.frame_type == TYPE_EMERGENCY_STOP:
        parsed = parse_emergency_stop_payload(frame.payload)
        if parsed is not None:
            self._logger.error(f"EMERGENCY_STOP reason={parsed}")
            if self._event_bus:
                self._event_bus.publish(EmergencyStopEvent(parsed))
    else:
        self._logger.warning(f"Unknown frame type: 0x{frame.frame_type:02X}")
```

`_handle_frame_with_stats` 中的过滤简化（所有已知类型直接分发，无需单独列出）：

```python
def _handle_frame_with_stats(self, frame: FrameData):
    self._rx_frames_ok += 1
    self._handle_frame(frame)
    # 注意：_handle_frame 内部对未知类型会 +1 _rx_frames_unknown 并警告
    # ...stats logging...
```

### 7.5 coordinator.py 重构（核心改动）

**删除：**
- `_heartbeat_thread`, `_heartbeat_seq`, `_heartbeat_lock`
- `_last_mcu_heartbeat`, `_is_linked`
- `_HEARTBEAT_INTERVAL`, `_HEARTBEAT_TIMEOUT`
- `_heartbeat_loop()` 方法
- `_send_initial_status()` 方法

**新增/修改：**

```python
# 新常量
_CMD_TIMEOUT = 5.0  # 5s 无 CMD 视为断链

class LineFollowCoordinator:
    def __init__(self, event_bus: EventBus):
        # ...原有初始化（移除心跳相关字段）...
        
        # Master-slave streaming state
        self._streaming_type = 0      # 0 = IDLE / not streaming
        self._stream_seq = 0
        self._last_cmd_time = 0.0
        self._master_linked = False
        
        # ⚠ 并发保护：_streaming_type / _last_cmd_time / _master_linked
        # 由 EventBus 线程写入，主线程读取
        self._cmd_lock = threading.Lock()
        
        # Vision result cache
        self._latest_line: dict = {}
        self._latest_ai: dict = {}

    def start(self) -> None:
        self._running = True
        self._wire_events()
        self._last_cmd_time = time.monotonic()
        # 不再启动心跳线程

    def stop(self) -> None:
        self._running = False
        # 心跳线程已删除，无需 join

    def get_info(self) -> dict:
        with self._cmd_lock:
            linked = self._master_linked
        detections = self._latest_ai.get("detections", [])
        return {
            "state": self.state_machine.current_state,
            "state_id": self.state_machine.current_state_id,
            "link_active": linked,
            "det_count": len(detections),
            "fps": self._last_fps,
        }

    def _wire_events(self) -> None:
        self.event_bus.subscribe(CmdRequestEvent, self._on_cmd_request)
        self.event_bus.subscribe(CmdStopEvent, self._on_cmd_stop)
        self.event_bus.subscribe(EmergencyStopEvent, self._on_emergency)

    def loop(self) -> None:
        # 1. Drain vision results → 更新缓存
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

        # 2. 处理状态机事件队列
        with self._sm_lock:
            while self._sm_queue:
                self._sm_queue.popleft()()
        self.state_machine.run_to_completion()

        # 3. 链路超时检测 + 流送
        now = time.monotonic()
        with self._cmd_lock:
            streaming = self._streaming_type
            if streaming != 0:
                if now - self._last_cmd_time > _CMD_TIMEOUT:
                    self._streaming_type = 0
                    self._master_linked = False
                    streaming = 0  # 超时后不再发送
                else:
                    payload = self._build_stream_payload(streaming)
                    if payload is not None and len(payload) > 0:
                        frame = build_data_stream_frame(
                            self._stream_seq, streaming, payload)
                        self._send(frame)
                        self._stream_seq = (self._stream_seq + 1) & 0xFF
        
        # 4. WDT 喂狗（冗余，主循环也喂了）
        self._wdt_feed()

    # 多线程安全：CMD 事件处理器
    def _on_cmd_request(self, event: CmdRequestEvent) -> None:
        data_type = event.data_type
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._master_linked = True

        if data_type not in SUPPORTED_DATA_TYPES:
            frame = build_cmd_nack_frame(data_type, NACK_UNSUPPORTED_TYPE)
            self._send(frame)
            return

        with self._cmd_lock:
            # 切换流送状态（隐式停止旧流）
            self._streaming_type = data_type
            self._stream_seq = 0
        payload_size = DATA_PAYLOAD_SIZES.get(data_type, 0)
        if payload_size is None:
            payload_size = 0
        frame = build_cmd_ack_frame(data_type, 60, payload_size)
        self._send(frame)

    def _on_cmd_stop(self, event: CmdStopEvent) -> None:
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._streaming_type = 0
            self._stream_seq = 0
        frame = build_cmd_ack_frame(0x00, 0, 0)
        self._send(frame)
```

> `_on_cmd_request`、`_on_cmd_stop` 的实现在上方的主代码块中（带 `_cmd_lock` 保护），此处不重复列出。

**视觉结果缓存与流送 payload 构建：**

```python
def _process_vision_results(self, all_results: dict) -> None:
    for pipeline_id, results in all_results.items():
        for task_name, vision_result in results.items():
            if not isinstance(vision_result, VisionResult):
                continue
            data = vision_result.result_data if vision_result.success else {}
            if task_name == "line_follow":
                self._latest_line = data
            elif task_name == "ai_inference":
                self._latest_ai = data

def _build_stream_payload(self, data_type: int) -> Optional[bytes]:
    if data_type == DATA_LINE_POSITION:
        return self._build_line_position_payload()
    elif data_type == DATA_TARGET_POSITION:
        return self._build_target_position_payload()
    elif data_type == DATA_TARGET_COUNT:
        return self._build_target_count_payload()
    elif data_type == DATA_DETECTION_STATUS:
        return self._build_detection_status_payload()
    elif data_type == DATA_ALL_TARGETS:
        return self._build_all_targets_payload()
    return None

def _build_line_position_payload(self) -> Optional[bytes]:
    data = self._latest_line
    pe_x = data.get("percent_error_x", 0)
    # 从视觉数据中获取或使用默认值
    pe_y = data.get("percent_error_y", 0)
    target_found = data.get("target_found", False)
    flags = 1 if target_found else 0
    state = self.state_machine.current_state_id
    return (pe_x.to_bytes(2, 'little', signed=True) +
            pe_y.to_bytes(2, 'little', signed=True) +
            bytes([flags, state]))

def _build_target_position_payload(self) -> Optional[bytes]:
    data = self._latest_ai
    detections = data.get("detections", [])
    if detections:
        # Detection 是 dataclass，使用属性访问（非 dict .get()）
        best = max(detections, key=lambda d: d.score)
        x = int(best.x)
        y = int(best.y)
        conf = max(0, min(255, int(best.score * 255)))
        flags = 0x01  # valid
    else:
        x, y, conf = 0, 0, 0
        flags = 0x00  # not valid
    return (x.to_bytes(2, 'little', signed=True) +
            y.to_bytes(2, 'little', signed=True) +
            bytes([conf, flags]))

def _build_target_count_payload(self) -> Optional[bytes]:
    data = self._latest_ai
    detections = data.get("detections", [])
    return bytes([len(detections)])

def _build_detection_status_payload(self) -> Optional[bytes]:
    data = self._latest_ai
    detections = data.get("detections", [])
    count = len(detections)
    visual_state = 0  # TODO: 从 VisualStateMachine 获取
    visual_flags = 0
    return (bytes([visual_state, visual_flags]) +
            count.to_bytes(2, 'little'))

def _build_all_targets_payload(self) -> Optional[bytes]:
    data = self._latest_ai
    detections = data.get("detections", [])
    count = min(len(detections), 16)
    payload = bytes([count])
    for d in detections[:count]:
        payload += (int(d.x).to_bytes(2, 'little', signed=True) +
                    int(d.y).to_bytes(2, 'little', signed=True) +
                    bytes([d.class_id]))
    return payload
```

### 7.6 调度说明

1. **主循环 @60Hz**：`module_manager.run_main_loop()` 每 16.7ms 调用一次 `coordinator.loop()`。每次 loop 最多发送 1 帧 DATA_STREAM。
2. **视觉管道异步**：Camera Process 线程独立运行，将检测结果入 deque。主循环在每次 loop 中排空 deque，只保留最新值。
3. **链路超时**：若 5 秒内未收到任何 CMD（REQUEST/STOP），从机自动停止流送并标记断链。
   - ⚠ `_last_cmd_time` **仅在** `_on_cmd_request` / `_on_cmd_stop`（即 EventBus 回调）中更新。
   - 流送中发送 DATA_STREAM **不会**刷新 `_last_cmd_time`。这意味着若主机发 CMD_REQUEST 后对 UART 总线一言不发，即使从机仍在以 60fps 流送，`_last_cmd_time` 也不会更新→5s 后超时触发→从机停止流送。
   - 这符合"主机不发 CMD 即视为断链"的设计原则。
4. **恢复方式**：收到新 CMD_REQUEST 即重新连接。
5. **WDT**：ModuleManager 主循环已每 16.7ms 喂一次狗（超时 3s）。Coordinator 额外喂狗作为冗余。
6. **`min_interval_ms` 当前被忽略**：该字段保留供未来使用，当前从机总是以最高速率（60Hz）推送。

### 7.7 迁移注意事项（从 v2.1 到 v3.0）

从旧版 coordinator 迁移时需同步执行以下清理：

| 改动点 | 操作 | 原因 |
|--------|------|------|
| `_heartbeat_thread` / `_heartbeat_loop()` | **删除** | 不再需要心跳线程 |
| `_heartbeat_seq` / `_last_mcu_heartbeat` / `_is_linked` | **删除** | 被 `_streaming_type` / `_last_cmd_time` / `_master_linked` 替代 |
| `_HEARTBEAT_INTERVAL` / `_HEARTBEAT_TIMEOUT` | **删除** | 不再使用 |
| `_heartbeat_lock` | **删除** | 被 `_cmd_lock` 替代 |
| `_on_heartbeat()` 方法 | **删除** | 不再订阅 HeartbeatEvent |
| `is_link_active()` 方法 | **删除或重写** | 改用 `get_info()["link_active"]` |
| `_send_initial_status()` 调用 | **删除** | 不再需要启动时发心跳帧 |
| 导入 `HeartbeatEvent` | **删除** | 不再使用 |
| 导入 `build_heartbeat_frame` / `build_visual_servo_data_frame` | **删除** | 被 DATA_STREAM 构建器替代 |
| `_handle_line_follow_result()` 直接发送帧 | **改为缓存** | 发送由流送机制统一控制 |
| `_last_det_count` | **改为动态获取** | 从 `len(self._latest_ai.get("detections", []))` 动态计算 |

---

## 8. MSPM0 主机端实现指南

本节为另一 Agent 实现 MSPM0 端提供指引。[`mspm0_protocol_impl.md`](./mspm0_protocol_impl.md) 已提供帧解析/构建的基础设施，本节仅说明**扩展部分**。

### 8.1 回调注册

现有 `op_uart_register_callback()` 机制足够。需注册的回调与应移除的旧回调：

```c
// ===== 注册新回调 =====
op_uart_register_callback(0x21, on_cmd_ack);      // TYPE_CMD_ACK   — 确认请求
op_uart_register_callback(0x22, on_cmd_nack);     // TYPE_CMD_NACK  — 拒绝请求
op_uart_register_callback(0x24, on_data_stream);  // TYPE_DATA_STREAM — 流数据

// ===== 保留的旧回调 =====
op_uart_register_callback(0x18, on_emergency_stop); // EMERGENCY_STOP — 双向紧急停止

// ===== 应移除的旧回调 =====
// 以下类型在 v3.0 中已移除/废弃，不再注册回调：
//   0x02 (TYPE_ARRIVED), 0x03 (TYPE_PICK), 0x04 (TYPE_SET)
//   0x10 (TYPE_CMD_FROM_MCU), 0x14 (TYPE_ACTION_DONE)
//   0x15 (TYPE_HEARTBEAT)       — 不再使用
//   0x17 (TYPE_VISUAL_SERVO_DATA) — 被 DATA_STREAM 替代

// 不需要注册接收回调的帧（主机发送，主机不需要收）:
//   0x20 (TYPE_CMD_REQUEST), 0x23 (TYPE_CMD_STOP)
```

> 若旧代码中注册了上列"应移除"的回调函数，建议注释或删除对应的注册行，避免误导后续维护者。

### 8.2 帧构建新增内容

```c
// op_uart_send() 已支持任意 type。主机端只需使用正确参数调用即可：

// 请求订阅 line position 数据
uint8_t req_payload[3] = {0x01, 0x00, 0x00};  // data_type, interval, reserved
op_uart_send(0x20, req_payload, 3);            // TYPE_CMD_REQUEST

// 停止当前数据流
op_uart_send(0x23, NULL, 0);                   // TYPE_CMD_STOP (空payload)

// 紧急停止（保留）
op_uart_send(0x18, &reason, 1);               // TYPE_EMERGENCY_STOP
```

### 8.3 主循环状态机

MSPM0 主循环的角色从"等待并处理接收数据"变为"主动控制通信"：

```
// 伪代码
typedef enum {
    IDLE,           // 无活跃订阅
    WAITING_ACK,    // 已发 CMD_REQUEST，等待 ACK/NACK
    STREAMING,      // 正在接收 DATA_STREAM
    ERROR,          // 通信异常
} comm_state_t;

comm_state_t comm_state = IDLE;
uint8_t current_data_type = 0;

// 中断回调设置以下标志（使用 volatile）
volatile uint8_t rx_frame_type = 0;   // 最近收到的帧类型
volatile uint8_t rx_data_type = 0;    // 最近收到的 data_type
volatile uint8_t rx_reason = 0;       // NACK 原因
volatile uint8_t rx_pending = 0;      // 1 = 有新帧待处理

// on_cmd_ack 回调
void on_cmd_ack(const uint8_t *payload, uint8_t len) {
    if (len < 3) return;
    rx_frame_type = TYPE_CMD_ACK;
    rx_data_type = payload[0];
    // payload[1] = max_freq_hz (ignored)
    // payload[2] = payload_size (for buffer allocation)
    rx_pending = 1;
}

// on_cmd_nack 回调
void on_cmd_nack(const uint8_t *payload, uint8_t len) {
    if (len < 2) return;
    rx_frame_type = TYPE_CMD_NACK;
    rx_data_type = payload[0];
    rx_reason = payload[1];
    rx_pending = 1;
}

// on_data_stream 回调 (IRQ context)
void on_data_stream(const uint8_t *payload, uint8_t len) {
    // 轻量拷贝，供主循环使用
    // 见 §8.4
}

void main_loop(void) {
    wdt_feed();
    uint32_t cmd_timeout = 0;
    
    switch (comm_state) {
    case IDLE:
        if (need_line_follow()) {
            uint8_t req[3] = {0x01, 0, 0};
            op_uart_send(0x20, req, 3);          // CMD_REQUEST
            current_data_type = 0x01;
            comm_state = WAITING_ACK;
            cmd_timeout = get_tick_ms() + 10;     // 10ms（@115200 约 1ms 即可收完 ACK）
        }
        break;
        
    case WAITING_ACK:
        if (rx_pending) {
            rx_pending = 0;
            if (rx_frame_type == TYPE_CMD_ACK && rx_data_type == current_data_type) {
                // ACK 匹配 → 开始流送
                comm_state = STREAMING;
            } else if (rx_frame_type == TYPE_CMD_NACK) {
                // NACK → 退避后重试
                comm_state = IDLE;
                delay_ms(50);
            } else if (rx_frame_type == TYPE_CMD_ACK && rx_data_type != current_data_type) {
                // ACK 但类型不匹配 → 退避
                comm_state = IDLE;
                delay_ms(10);
            }
        } else if (get_tick_ms() > cmd_timeout) {
            // 超时 → 重试
            comm_state = IDLE;
        }
        break;
        
    case STREAMING:
        // on_data_stream 回调已更新 shared_data
        // 控制算法读取 shared_data.pe_x 等
        if (need_stop()) {
            op_uart_send(0x23, NULL, 0);          // CMD_STOP
            // 不等待 ACK，直接进入 IDLE
            comm_state = IDLE;
        }
        break;
        
    case ERROR:
        // 错误处理逻辑
        break;
    }
}
```

### 8.4 DATA_STREAM 解析

⚠ **中断上下文警告**：`on_data_stream` 回调在 UART4 RX 中断上下文中执行。必须保持 <5µs，禁止阻塞操作、printf、动态分配。

```c
// ⚠ 运行在 IRQ 上下文 — 仅做轻量赋值，业务逻辑在主循环处理
static void on_data_stream(const uint8_t *payload, uint8_t len) {
    if (len < 2) return;  // 最少需要 seq(1) + data_type(1)
    
    uint8_t seq      = payload[0];
    uint8_t data_type = payload[1];
    
    // ── Cortex-M0+ 注意 ──
    // 勿使用 *(int16_t*)&payload[2]！ARMv6-M 不支持未对齐半字加载，
    // 若 payload 指针未 2 字节对齐则触发 HardFault。
    // 使用位运算：payload[2] | (payload[3] << 8)（如下）安全可移植。
    
    if (data_type == 0x01 && len >= 8) {
        // DATA_LINE_POSITION
        int16_t pe_x = (int16_t)(payload[2] | (payload[3] << 8));
        int16_t pe_y = (int16_t)(payload[4] | (payload[5] << 8));
        uint8_t flags = payload[6];
        uint8_t state = payload[7];
        // 更新全局变量供主循环使用
    }
    else if (data_type == 0x02 && len >= 8) {
        // DATA_TARGET_POSITION
        int16_t x = (int16_t)(payload[2] | (payload[3] << 8));
        int16_t y = (int16_t)(payload[4] | (payload[5] << 8));
        uint8_t conf = payload[6];
        uint8_t flags = payload[7];
        if (flags & 0x01) {
            // valid=1: 更新目标坐标（使用全局变量或乒乓缓冲）
            shared_target_x = x;
            shared_target_y = y;
            shared_target_conf = conf;
            shared_target_valid = 1;
        } else {
            // valid=0: 保留上一次有效值，通知主循环"无新检测"
            shared_target_valid = 0;
        }
    }
    else if (data_type == 0x03 && len >= 3) {
        // DATA_TARGET_COUNT
        uint8_t count = payload[2];
        shared_target_count = count;
    }
    else if (data_type == 0x04 && len >= 6) {
        // DATA_DETECTION_STATUS
        uint8_t  visual_state = payload[2];
        uint8_t  visual_flags = payload[3];
        uint16_t count = (uint16_t)(payload[4] | (payload[5] << 8));
        shared_visual_state = visual_state;
        shared_visual_flags = visual_flags;
        shared_detection_count = count;
    }
    else if (data_type == 0x05 && len >= 3) {
        // DATA_ALL_TARGETS
        uint8_t count = payload[2];
        if (count == 0) return;  // 无目标，无需处理
        if (len < (uint8_t)(3 + count * 5)) return;
        for (uint8_t i = 0; i < count && i < 16; i++) {
            uint8_t *entry = payload + 3 + i * 5;
            int16_t x = (int16_t)(entry[0] | (entry[1] << 8));
            int16_t y = (int16_t)(entry[2] | (entry[3] << 8));
            uint8_t class_id = entry[4];
            // 更新全局检测数组
        }
        shared_target_count = count;
    }
    // 忽略未知 data_type
}
```

### 8.5 UART 配置注意事项（MSPM0G3519）

**RX FIFO 阈值：** 建议设置为 **4 或 8 字节**（而非阈值=1）。
- 115200 波特率下每字节需 86.8µs（8N1），推荐 4~8 字节以减少中断次数。
- 在 FIFO 中断服务函数中使用循环读取，一次性排空 FIFO 中所有字节。

**IRQ 优先级：** 建议 `NVIC_SetPriority(UART4_INT_IRQn, 1)`。
- 优先级 0 保留给电机步进等硬实时中断。
- 优先级 1 确保 UART4 能在 TOF/UART0-1 等 ISR 之前得到响应。

**引脚冲突验证：** 实施前确认 UART4 GPIO 不与 UART0/1/3/7 及电机驱动 GPIO 冲突。

### 8.6 实现步骤

```
Step 1 — 依据 mspm0_protocol_impl.md 完成 UART4 基础设施
         注意：忽略 §9 中的旧回调注册示例，使用 §8.1 的新回调列表
Step 2 — 新增 protocol.h 中的帧类型常量
           #define TYPE_CMD_REQUEST     0x20
           #define TYPE_CMD_ACK         0x21
           #define TYPE_CMD_NACK        0x22
           #define TYPE_CMD_STOP        0x23
           #define TYPE_DATA_STREAM     0x24
Step 3 — 注册新回调（参考 §8.1）
           op_uart_register_callback(0x21, on_cmd_ack);
           op_uart_register_callback(0x22, on_cmd_nack);
           op_uart_register_callback(0x24, on_data_stream);
           op_uart_register_callback(0x18, on_emergency_stop);
Step 4 — 实现主循环通信状态机（IDLE/WAITING_ACK/STREAMING，参考 §8.3）
Step 5 — 实现 on_data_stream 解析器（参考 §8.4）
Step 6 — 端到端测试：MD5 → CMD_REQUEST → MaixCAM2 ACK → DATA_STREAM
```

---

## 9. 附录：完整帧例

以下为十六进制示例，可用作测试向量。

### CMD_REQUEST (DATA_LINE_POSITION)

```
AA 55 06 20 01 00 00 84 BE
│   │  │  │  │  │  │  │  │
│   │  │  │  │  │  │  │  └─ CRC16_LO
│   │  │  │  │  │  │  └──── CRC16_HI
│   │  │  │  │  │  └─────── reserved=0x00
│   │  │  │  │  └────────── min_interval_ms=0x00
│   │  │  │  └───────────── data_type=0x01
│   │  │  └──────────────── Type=0x20 (CMD_REQUEST)
│   │  └─────────────────── Length=0x06
│   └────────────────────── SOF2=0x55
└────────────────────────── SOF1=0xAA
```

**字段分解：**
- Type + Payload = `20 01 00 00`
- CRC16 = CRC16-CCITT(`20 01 00 00`) = `0x84BE` → 大端 `84 BE`

### CMD_ACK (DATA_LINE_POSITION)

```
AA 55 06 21 01 3C 08 33 FA
│   │  │  │  │  │  │  │  │
│   │  │  │  │  │  │  │  └─ CRC16_LO
│   │  │  │  │  │  │  └──── CRC16_HI
│   │  │  │  │  │  └─────── payload_size=0x08 (8 bytes per DATA_STREAM)
│   │  │  │  │  └────────── max_freq_hz=0x3C (60 Hz)
│   │  │  │  └───────────── data_type=0x01
│   │  │  └──────────────── Type=0x21 (CMD_ACK)
│   │  └─────────────────── Length=0x06
│   └────────────────────── SOF2=0x55
└────────────────────────── SOF1=0xAA
```

### CMD_NACK (UNSUPPORTED_TYPE)

```
AA 55 05 22 FF 01 37 E4
│   │  │  │  │  │  │  │
│   │  │  │  │  │  │  └─ CRC16_LO
│   │  │  │  │  │  └──── CRC16_HI
│   │  │  │  │  └─────── reason=0x01 (UNSUPPORTED_TYPE)
│   │  │  │  └────────── data_type=0xFF
│   │  │  └───────────── Type=0x22 (CMD_NACK)
│   │  └──────────────── Length=0x05
│   └─────────────────── SOF2=0x55
└─────────────────────── SOF1=0xAA
```

### CMD_STOP

```
AA 55 03 23 F5 F1
│   │  │  │  │  │
│   │  │  │  │  └─ CRC16_LO
│   │  │  │  └──── CRC16_HI
│   │  │  └─────── Type=0x23 (CMD_STOP)
│   │  └────────── Length=0x03 (空 payload: Type=1 + payload=0 + CRC16=2)
│   └───────────── SOF2=0x55
└───────────────── SOF1=0xAA
```

### DATA_STREAM (DATA_LINE_POSITION)

```
AA 55 0B 24 00 01 02 00 00 00 01 00 23 71
│   │  │  │  │  │  │  │  │  │  │  │  │  │
│   │  │  │  │  │  │  │  │  │  │  │  │  └─ CRC16_LO
│   │  │  │  │  │  │  │  │  │  │  │  └──── CRC16_HI
│   │  │  │  │  │  │  │  │  │  │  └─────── state=0x00
│   │  │  │  │  │  │  │  │  │  └────────── flags=0x01 (target found)
│   │  │  │  │  │  │  │  │  └───────────── pe_y=0x0000
│   │  │  │  │  │  │  │  └──────────────── pe_x=0x0002 (2 pixels right)
│   │  │  │  │  │  │  └─────────────────── data_type=0x01
│   │  │  │  │  │  └────────────────────── seq=0x00
│   │  │  │  │  └───────────────────────── Type=0x24 (DATA_STREAM)
│   │  │  │  └──────────────────────────── Length=0x0B (11 = 1+8+2)
│   │  │  └─────────────────────────────── SOF2=0x55
│   │  └────────────────────────────────── SOF1=0xAA
```

**字段分解：**
- Type + Payload = `24 00 01 02 00 00 00 01 00`
- CRC16 = CRC16-CCITT(`24 00 01 02 00 00 00 01 00`) = `0x2371` → 大端 `23 71`
- seq=0x00, data_type=0x01, pe_x=0x0002, pe_y=0x0000, flags=0x01, state=0x00

---

> 本文档与 [`mspm0_protocol_impl.md`](./mspm0_protocol_impl.md) 配合阅读，
> 后者提供 CRC16 查表实现和帧解析状态机的 C 代码详细参考。
