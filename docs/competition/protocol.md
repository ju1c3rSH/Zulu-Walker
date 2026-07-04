# 视觉-MCU 通信协议

> 适用项目：Zulu-Walker（Orange Pi 5B + STM32）
> 用途：香橙派（视觉/决策）与 STM32（运动/执行）的 UART 通信规范

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

## 3. 帧类型定义

### 3.1 旧区域事件帧（保留兼容）

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| 0x01 | `TYPE_ERROR` | OP → MCU | `error_type(1B) + error_value(2B LE)` | 视觉伺服误差（X=0, Y=1, Z=2, Other=3） |
| 0x02 | `TYPE_ARRIVED` | MCU → OP | `zone_id(1B)` | 到达某区域（START=0, QR_BOARD=1, RAW=2, ROUGH=3, TEMP=4） |
| 0x03 | `TYPE_PICK` | MCU → OP | `zone_id(1B)` | 请求在某区抓取 |
| 0x04 | `TYPE_SET` | MCU → OP | `zone_id(1B)` | 设置当前区域 |

### 3.2 任务同步帧

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| **0x10** | `TYPE_CMD_FROM_MCU` | MCU → OP | `cmd_id(1B) + args` | MCU 请求视觉服务 |
| **0x11** | `TYPE_STATUS_FROM_VISION` | OP → MCU | `mission_state(1B) + visual_state(1B) + flags(1B) + cargo_count(1B)` | 视觉上报综合状态 |
| **0x12** | `TYPE_QR_RESULT` | OP → MCU | `len(1B) + ascii[len]` | 二维码任务码 |
| **0x13** | `TYPE_COLOR_RESULT` | OP → MCU | `color_id(1B) + confidence(1B)` | 当前物料颜色识别结果 |
| **0x14** | `TYPE_ACTION_DONE` | MCU → OP | `action_id(1B) + result(1B)` | MCU 动作完成 |
| **0x15** | `TYPE_HEARTBEAT` | 双向 | `seq(1B) + mission_state(1B) + visual_state(1B)` | 心跳 + 状态快照。MCU → OP 时 `visual_state` 恒为 0 |
| **0x16** | `TYPE_REQUEST_SYNC` | 双向 | `requested_state(1B)` | 请求对方强制同步到某状态 |
| **0x17** | `TYPE_VISUAL_SERVO_DATA` | OP → MCU | `error_x(2B LE) + error_y(2B LE) + distance(2B LE) + state(1B)` | 高频视觉伺服数据 |
| **0x18** | `TYPE_EMERGENCY_STOP` | 双向 | `reason(1B)` | 任意一方急停 |

---

## 4. 子命令（TYPE_CMD_FROM_MCU）

| cmd_id | 名称 | 参数 | 说明 |
|:---|:---|:---|:---|
| 0x01 | `CMD_START_QR` | 无 | 一键启动，开启 QR 检测 |
| 0x06 | `CMD_STOP_VISUAL` | 无 | 紧急停止所有视觉任务 |

> **注意**：视觉任务的日常启停由 Orange Pi 根据 MissionSM 状态变化自动管理，MCU 不需要发送 CMD 来开关任务。详见 `docs/architecture/state_machine.md` §2 区域自动控制。

---

## 5. VisualFlags（STATUS_FROM_VISION 的 flags 字节）

```
bit 0: TARGET_FOUND      当前帧检测到目标
bit 1: READY_TO_PICK     偏差足够小，可以抓取
bit 2: READY_TO_PLACE    偏差足够小，可以放置
bit 3: VISUAL_FAIL       视觉异常（超时/丢失）
bit 4: QR_OK             二维码已识别
bit 5: CARGO_CONFIRMED   视觉确认载台有物料
bit 6: COLOR_MISMATCH    检测到的颜色与预期不符
bit 7: 保留
```

---

## 6. action_id 映射（TYPE_ACTION_DONE）

每完成一个动作，MCU 发一次 `ACTION_DONE`，`action_id` 标识动作类型：

| action_id | 对应动作 | 触发事件 | 每批次循环次数 |
|:---|:---|:---|:---|
| 1 | `PICK_RAW` — 原料区取料 | `PICK_DONE` | 3（每次取 1 个物料） |
| 2 | `PLACE_ROUGH` — 粗加工区放料 | `place_action_done` 标志 | 3（每次放 1 个物料） |
| 3 | `PLACE_TEMP` — 暂存区放料/码垛 | `place_action_done` 标志 | 3（每次放 1 个物料） |
| 4 | `PICK_ROUGH` — 粗加工区取料（回收入机器人） | `PICK_DONE` | 3（每次取 1 个物料） |

> **注意**：`action_id=2/3`（PLACE）不触发事件转换，而是通过 `place_action_done` 标志让状态机的 `on_execute` 决定下一步。详见 `docs/architecture/state_machine.md` §2 混合模型。

## 8. 结果码

| 值 | 名称 | 含义 |
|:---|:---|:---|
| 0x00 | `ACTION_OK` | 成功 |
| 0x01 | `ACTION_BUSY` | 执行中 |
| 0x02 | `ACTION_TIMEOUT` | 超时 |
| 0x03 | `ACTION_FAIL` | 失败 |
| 0x04 | `ACTION_NO_CARGO` | 未检测到物料 |

---

## 9. 典型通信时序

### 7.1 标准流程（区域自动控制）

```
OP 上电 → STATUS_FROM_VISION state=IDLE

MCU 按键 → CMD_START_QR
OP 开启 qr_detect → STATUS_FROM_VISION state=READ_QR
OP 解码成功 → QR_RESULT "123+231"
         → STATUS_FROM_VISION state=NAV_TO_RAW

MCU 到达 RAW → TYPE_ARRIVED zone=2
OP 切换到 ALIGN_RAW
OP 自动开启 track_cargo
  → VISUAL_SERVO_DATA (持续发送，MCU PID 微调)
  → STATUS_FROM_VISION flags=ready_to_pick
MCU 抓取 → ACTION_DONE action_id=1, result=OK
OP 推进 step，继续下一轮 ...

MCU 到达 ROUGH → TYPE_ARRIVED zone=3
OP 自动切换到 ring_track
  → VISUAL_SERVO_DATA
  → STATUS_FROM_VISION flags=ready_to_place
MCU 放置 → ACTION_DONE action_id=2, result=OK
OP 重复 ALIGN+PLACE 直到 cargo_count=0
OP 切换到 PICK 阶段（picking_from_rough=True）

MCU 到达 ROUGH 色环前（取回阶段）
OP ALIGN_ROUGH → VISUAL_SERVO_DATA
  → STATUS_FROM_VISION flags=ready_to_pick
MCU 取回 → ACTION_DONE action_id=4, result=OK
OP 重复 ALIGN+PICK 直到 cargo_count=3

MCU 到达 TEMP → TYPE_ARRIVED zone=4
OP 自动切换到 ring_track
  → VISUAL_SERVO_DATA
  → STATUS_FROM_VISION flags=ready_to_place
MCU 放置 → ACTION_DONE action_id=3, result=OK
OP 重复 ALIGN+PLACE 直到 cargo_count=0

OP → RETURN_HOME → FINISHED
```

关键要点：
- **MCU 不需要发 CMD 来开关视觉任务** — OP 自动根据状态切换
- **MCU 不需要发 CMD 来传递颜色** — OP 从 `batch_order` 自动获取
- MCU 只需发 `TYPE_ARRIVED` 告知到达区域，发 `ACTION_DONE` 告知动作完成
- **`ARRIVED_*` 同步职责**：MCU 决定前往某区域后，更新自身 MissionSM + 发 `TYPE_ARRIVED`；
  OP 收到后更新镜像 MissionSM。双方状态机保持同步。

### 9.2 视觉伺服数据流

```
OP 进入 TRACKING 状态后，每帧发送 VISUAL_SERVO_DATA：
  error_x: int16 LE（归一化 -5000~5000，0=画面中心）
  error_y: int16 LE（同上）
  distance_mm: int16 LE（目标距离，毫米）
  state: 1B（当前 visual_state: 0=IDLE, 1=SEARCH, 2=TRACKING, 3=RECOVERY, 4=FAIL）

MCU 根据 error_x/error_y 做 PID 微调，直到视觉上报 ready_to_pick/ready_to_place
```

### 9.3 心跳与超时

```
双方每 100ms 互发 HEARTBEAT：
  seq: 1B（递增，0~255 循环）
  mission_state: 1B（发送方的当前任务状态 ID）
  visual_state: 1B（发送方当前视觉状态 ID，MCU→OP 时恒为 0）

如果一方连续 3 次（300ms）未收到对方心跳：
  → 本地状态机切 ERROR
  → 发 EMERGENCY_STOP reason=HEARTBEAT_LOST
  → 停车，等待人工 RESET
```

### 9.4 错误恢复

```
MCU/OP 发现异常 → 切 ERROR
  → 发 STATUS_FROM_VISION state=ERROR, flags=visual_fail
  → 发 REQUEST_SYNC state=ERROR
对方收到后也切 ERROR

人工修复后 → 发 RESET（或 CMD_STOP_VISUAL 后重新 CMD_START_QR）
  → 状态机回到 WAIT_START
```

---

## 8. 同步规则

1. **谁触发转换，谁负责通知对方**。
   - MCU 触发的转换（到达区域、执行完成）→ MCU 发 `TYPE_ARRIVED` / `TYPE_ACTION_DONE`
   - OP 触发的转换（QR 识别成功、目标丢失、可放置）→ OP 发 `TYPE_STATUS_FROM_VISION`

2. **双方都维护一份 MissionSM**。

3. **定期心跳 `TYPE_HEARTBEAT`**。丢失 3 个心跳认为通信中断 → `ERROR`。

4. **状态不一致时 `REQUEST_SYNC`**。双方切 `ERROR` 并停车。

---

## 9. 代码映射

| 协议元素 | Python 常量 | 位置 |
|---|---|---|
| 帧类型 | `TYPE_ERROR` ~ `TYPE_EMERGENCY_STOP` | `protocol.py` |
| 子命令 | `CMD_START_QR`, `CMD_STOP_VISUAL` | `protocol.py` |
| 标志位 | `class VisualFlags` | `protocol.py` |
| 结果码 | `ACTION_OK` ~ `ACTION_NO_CARGO` | `protocol.py` |
| 帧构建 | `build_*_frame()` 系列函数 | `protocol.py` |
| 帧解析 | `parse_*_payload()` 系列函数 | `protocol.py` |

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v1.2** | 2026-07-04 | ① 删除 `CMD_START_COLOR_DETECT(0x02)`、`CMD_TRACK_TARGET(0x03)`、`CMD_TRACK_TOP(0x05)` — 视觉任务启停改为区域自动控制；② `CMD_TRACK_RING(0x04)` 不再需要，一并删除；③ `COLOR_RESULT` 确认 payload 为 `color_id + confidence`（v1.1 已改）；④ 新增 §7.2~7.4 时序补充；⑤ 新增 §9 代码映射 |
| **v1.1** | 2026-07-03 | ① `COLOR_RESULT` 移除 `slot_idx`；② `CMD_START_COLOR_DETECT` 移除 `slot_idx`；③ 移除 `CMD_SET_EXPOSURE`；④ 时序 B 新增颜色检测步骤；⑤ `HEARTBEAT` 增加 MCU→OP 时 `visual_state=0` |
| **v1.0** | — | 初版协议定义 |
