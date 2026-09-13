# 协议四源核对记录（2026-09-13）

> 分支 `EOD2026-NearCam` 协议重建依据。结论：**四源一致，无冲突**。
> 冲突决议规则：md 文档与 C 代码不一致时，以 C 代码 + 实测向量为准并回报。

## 1. 核对对象

| # | 来源 | 角色 |
|---|------|------|
| ① | `F:\2026工科赛\serial_protocol\panzer_links_protocol.md`（v2.1, 2026-09-12） | 语义规范 |
| ② | `F:\2026工科赛\serial_protocol\protocol.c/h` | C 库正本（帧格式/解析器） |
| ③ | `F:\2026工科赛\panzer\GetData\protocol.c/h` | 底盘副本 |
| ④ | `F:\2026工科赛\panzer_Brain\Middlewares\serial_protocol\protocol.c/h` + `Interrupt/interrupt.c/h` | 近端对端（USART1 主站）实际用法 |

- ②③④ 三份 C 库副本**逐行一致**（protocol.h 逐字节比对通过）。
- 帧格式正本 = protocol.h 头注释 + protocol.c 实现；md 引用的 `protocol.md` 文件不存在（悬空引用，审查 P2，不影响）。

## 2. 帧格式核对（① ↔ ②，一致）

- `AA 55 | Length | Type | Payload[0..252] | CRC_HI CRC_LO`，`Length = plen+3`（3~255）。
- CRC16-CCITT：poly 0x1021、init 0xFFFF、不反射、无终异或、覆盖 Type..Payload、**高字节在前**。
- Payload 内多字节一律小端（补码）。
- 编码器帧长 `plen+6`；Python 端 `_build_frame` 与 C `Protocol_EncodeFrame` 逐字节等价。

## 3. 以 C 代码为准的两处差异（Python 端按 C 重建）

1. **解析器 LEN 上限 = body_capacity**（C 端由集成方自选；panzer_Brain USART1 用 24B，只收 LEN≤24 的帧）。Python 端 `ProtocolParser` 默认 capacity=255（protocol.h `PROTOCOL_MAX_BODY`）；本端发出的 DATA_STREAM(LEN=13)/ACK(LEN=6) 均在 Brain 24B 限内。
2. **坏 CRC 恢复**：C 状态机**整帧丢弃**并复位 WAIT_SOF1（`AA AA 55` 重同步在 WAIT_SOF2 态实现：连续 0xAA 保持等待）。旧 MaixCAM1 Python 端的逐字节 pop 重扫且无 AA-AA-55 重同步——不采用，已按 C 四态状态机 1:1 移植（`modules/zw_uart_module/parser.py`）。

## 4. USART1 近端语义核对（① ↔ ④ interrupt.c 实际用法，一致）

| 语义 | Brain 侧实现 | 本端对应 |
|---|---|---|
| 订阅 | `Cam1_Subscribe` 发 `0x20` 5B `[data_id, interval_ms, 爆色, 靶色, 形状]` | `parse_cmd_request_payload`：5B 带 codes，1~4B = 重订阅沿用旧码 |
| 看门狗 | `Comm_Task`(10ms)：CAM 静默 >1s 同参重订阅（幂等） | 同码重订不清锁存（`set_task` 判重；§5.3） |
| STOP | `Cam1_Stop` 发 `0x23` 空载荷 | 停推 + 门控 idle + 处理器 reset |
| 数据消费 | `on_cam1` **只消费 0x24**，payload 恒 10B，did∈[0x01,0x03]，拷 8B 子载荷 | DATA_STREAM payload 恒 10B |
| ACK/NACK | **不消费**（on_cam1 无 0x21/0x22 分支） | 照发保协议一致性：ACK `[did,60,10]`、NACK `[did,0x01/0x02]` |
| 0x01 字段 | `pit@2, flags@3`（`Camera_GetPitPosition`） | `[cx u16LE, pit, flags, rsv×4]` |
| 0x02 字段 | `err i16LE@0-1, flags@2`（`Camera_GetTargetError`，系数 10000） | `[err i16LE, flags, rsv×5]`，clamp ±1 |
| 0x03 字段 | `pos@0, flags@1`（`Camera_GetPitPosition`） | `[pos, flags, rsv×6]` |
| 帧龄 | `CAMERA_FRAME_TIMEOUT_MS = 1000`：>1s 判 stale 拒用 | **零值心跳不断流**（缺阈值/无结果也按 interval 推全零帧） |
| 有效性 | `found=1 且 pit∈[1,3]` 才被采纳 | 未锁存推 `found=0`、`pit/pos=0`、不透传假 cx/err |

- data_id 全表（interrupt.h `DataId_t`）：0x01 EXPLOSIVE / 0x02 TARGET / 0x03 RESCUE / 0x10 QR_CODE（远端）/ 0x21 CAR_CTRL / 0x22 CAR_EVENT（底盘）。近端只支持 0x01~0x03。

## 5. 测试锚定

- `tests/test_protocol_gks.py`：C 自测（test_protocol.c）CRC/编码/重同步/错误/满载向量逐条移植 + md §6 USART1 向量逐字节断言 + Brain 字段偏移对拍。
- `tests/test_coordinator_gks.py`：订阅幂等/零值心跳/NACK/STOP/静默 idle 语义。
- `tests/test_gks_task_processor.py`：照搬逻辑（两帧锁存/镜像/定点化/形状映射）行为等价。
