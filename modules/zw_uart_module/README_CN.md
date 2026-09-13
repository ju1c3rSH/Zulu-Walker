# zw_uart_module

2026 工科赛近端 MaixCAM 视觉从站的 UART 协议模块（**panzer_links_protocol v2.1**，对接 panzer_Brain USART1 主站）。

> 协议四源核对记录: `docs/competition/gks_protocol_crosscheck.md`；
> 完整规范: `docs/competition/gks_panzer_links_protocol_v2.1.md`。

## 帧结构（与 serial_protocol C 库逐字节一致）

```
| AA | 55 | Length | Type | Payload[0..252] | CRC16_HI | CRC16_LO |
Length = Type + Payload + CRC16 = n + 3（有效 3~255）
CRC16-CCITT: poly 0x1021, init 0xFFFF, 不反射, 无终异或, 高字节在前
Payload 内多字节一律小端（补码）
```

## 帧型与 data_id

| Type | 名称 | 方向 | Payload |
|---|---|---|---|
| 0x20 | CMD_REQUEST | 主→从 | `[data_id, min_interval_ms, 爆色, 靶色, 形状]`（5B 完整订阅; 1~4B = 幂等重订阅沿用旧码） |
| 0x21 | CMD_ACK | 从→主 | `[data_id, 60, 10]` |
| 0x22 | CMD_NACK | 从→主 | `[data_id, reason]`（0x01 不支持 / 0x02 未就绪） |
| 0x23 | CMD_STOP | 主→从 | 空 |
| 0x24 | DATA_STREAM | 从→主 | `[seq, data_id, sub_payload(8B)]`，payload 恒 10B |

| data_id | 名称 | 子载荷（8B） |
|---|---|---|
| 0x01 | DATA_EXPLOSIVE | `[cx u16LE, pit 0~3, flags bit0=found, rsv×4]` |
| 0x02 | DATA_TARGET | `[err i16LE = norm×10000 clamp±1, flags bit0, rsv×5]` |
| 0x03 | DATA_RESCUE | `[pos 0~3, flags bit0, rsv×6]` |

## 模块结构

| 文件 | 职责 |
|---|---|
| `protocol.py` | 帧编解码（CRC 查表）、v2.1 语义（订阅解析/子载荷 builders/parsers） |
| `parser.py` | C `Protocol_ParserFeed` 四态状态机 1:1 移植：AA-AA-55 重同步、坏 CRC 整帧丢弃、Length 校验、统计计数（rx_bytes/valid_frames/crc_errors/length_errors） |
| `uart_driver.py` | HAL UART 包装 + 从站帧分发（CMD_REQUEST/STOP → EventBus） |
| `events.py` | `CmdRequestEvent`（含任务码）/ `CmdStopEvent` |

## 关键语义（对照 Brain interrupt.c 实际消费逻辑）

- 订阅期**零值心跳不断流**：缺阈值/无结果也按 `min_interval`（下限 30ms）推全零帧，
  否则 Brain 1s stale 判定 + 1s 看门狗重订阅。
- 同 data_id 同任务码的重订阅**不清视觉锁存**；换 id / 换码才清（`GksTaskProcessor.set_task` 判重）。
- ACK/NACK 照发保协议一致性（Brain 只消费 0x24）。
- 未知 data_id（如远端的 0x10）回 NACK `[did, 0x01]`。

## 测试

`python -m pytest tests/test_protocol_gks.py` — C 库自测向量移植（CRC/编码/重同步/错误/满载）+ md §6 USART1 业务向量逐字节断言 + Brain 字段偏移对拍。
