# Ping-Pong 保活协议

> 适用项目：Zulu-Walker（MaixCAM2 + MSPM0）
> 用途：MSPM0（Master）与 MaixCAM2（Slave）之间的链路存活检测
>
> 帧结构、CRC16 校验算法见 [`protocol.md`](protocol.md)

---

## 1. 帧类型

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| **0x25** | `TYPE_PING` | MSPM0 → MaixCAM2 | `seq(1B)` | 保活探测 |
| **0x26** | `TYPE_PONG` | MaixCAM2 → MSPM0 | `seq(1B)` | 保活应答，seq 原样回传 |

## 2. Payload

```
PING / PONG:
  Byte 0: seq (u8, 0~255，递增回环)
```

MaixCAM2 收到 PING 后立即构建 PONG，seq 原样拷贝，不做修改。

## 3. 行为规范

### 3.1 MSPM0（Master）侧

```
每 T_ping ms 发一次 PING(seq++)。
每次发 PING 后启动超时计时器：
  若 T_timeout ms 内收到匹配 seq 的 PONG → 链路正常
  若超时 → 连续失败计数 +1
    连续失败 ≥ N_max → 判定断线
      动作：停止运动、切 ERROR、报警
```

**推荐参数：**

| 参数 | 建议值 | 说明 |
|------|--------|------|
| T_ping | 200 ms | 探测间隔 |
| T_timeout | 100 ms | 单次超时 |
| N_max | 3 | 连续超时次数阈值 |

### 3.2 MaixCAM2（Slave）侧

- **被动应答**：收到 PING → 1ms 内回 PONG，不做超时检测
- **日志打印**：每次收到 PING 和发出 PONG 都打印一行日志
- **容错**：MaixCAM2 不因 PING 丢失而改变自身状态

### 3.3 异常处理

- MaixCAM2 收到格式错误的 PING（payload 长度 != 1）→ 丢弃，不回 PONG
- MSPM0 收到 seq 不匹配的 PONG → 忽略，按超时处理
- MSPM0 收到非预期的 PONG → MaixCAM2 不会主动发 PONG，可忽略

## 4. 时序示例

```
MSPM0                        MaixCAM2
  |                              |
  |── PING(seq=0x01) ──────────→|  PING seq=01 日志
  |                              |  构建 PONG(seq=0x01)
  |←─ PONG(seq=0x01) ──────────┤  PONG seq=01 日志
  |  链路OK ✓                    |
  |                              |
  |── PING(seq=0x02) ──────────→|  PING seq=02 日志
  |         (超时)               |
  |         fail_count = 1       |
  |                              |
  |── PING(seq=0x03) ──────────→|  PING seq=03 日志
  |←─ PONG(seq=0x03) ──────────┤  PONG seq=03 日志
  |  链路恢复 ✓                   |
  |  fail_count = 0              |
  |                              |
  |── PING(seq=0x04) ──────────→|
  |── PING(seq=0x05) ──────────→|  (串口物理断开)
  |── PING(seq=0x06) ──────────→|
  |  fail_count = 3 → 断线!     |
  |  → EMERGENCY_STOP            |
```

## 5. 与旧心跳（TYPE_HEARTBEAT）的关系

`TYPE_HEARTBEAT (0x15)` 已废弃。Ping-Pong 替代其链路检测功能，更简洁（seq 1 字节 vs 3 字节），职责单一（纯保活，不含业务状态同步）。

MaixCAM2 收到 HEARTBEAT 帧时静默忽略，不做任何动作。

## 6. 代码映射

| 协议元素 | Python 常量/函数 | 位置 |
|---|---|---|
| TYPE_PING | `TYPE_PING = 0x25` | `protocol.py` |
| TYPE_PONG | `TYPE_PONG = 0x26` | `protocol.py` |
| 构建 PING | `build_ping_frame(seq)` | `protocol.py` |
| 构建 PONG | `build_pong_frame(seq)` | `protocol.py` |
| 解析 | `parse_ping_payload(payload)` | `protocol.py` |
| 处理逻辑 | `STM32UartInterface._handle_frame()` | `uart_driver.py` |
