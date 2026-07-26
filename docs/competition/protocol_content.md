# 视觉-MCU 通信协议 — 内容定义

> 适用项目：Zulu-Walker（Orange Pi 5B + STM32）
> 用途：定义帧结构中的 Type、Payload、标志位等业务内容
>
> **帧结构定义**（物理层、帧格式、校验算法）见同目录 [`protocol.md`](protocol.md)

---

## 1. 帧类型定义

### 1.1 旧区域事件帧（保留兼容）

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| 0x01 | `TYPE_ERROR` | OP → MCU | `error_type(1B) + error_value(2B LE)` | 视觉伺服误差（X=0, Y=1, Z=2, Other=3） |
| 0x02 | `TYPE_ARRIVED` | MCU → OP | `zone_id(1B)` | 到达某区域（START=0, QR_BOARD=1, RAW=2, ROUGH=3, TEMP=4） |
| 0x03 | `TYPE_PICK` | MCU → OP | `zone_id(1B)` | 请求在某区抓取 |
| 0x04 | `TYPE_SET` | MCU → OP | `zone_id(1B)` | 设置当前区域 |

### 1.2 任务同步帧

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| **0x10** | `TYPE_CMD_FROM_MCU` | MCU → OP | `cmd_id(1B) + args` | MCU 请求视觉服务 |
| **0x11** | `TYPE_STATUS_FROM_VISION` | OP → MCU | `mission_state(1B) + visual_state(1B) + flags(1B) + cargo_count(1B)` | 视觉上报综合状态 |
| **0x12** | `TYPE_QR_RESULT` | OP → MCU | `len(1B) + ascii[len]` | 二维码任务码 |
| **0x13** | `TYPE_COLOR_RESULT` | OP → MCU | `color_id(1B) + confidence(1B)` | 当前物料颜色识别结果 |
| **0x14** | `TYPE_ACTION_DONE` | MCU → OP | `action_id(1B) + result(1B)` | MCU 动作完成 |
| **0x15** | `TYPE_HEARTBEAT` | 双向 | `seq(1B) + mission_state(1B) + visual_state(1B)` | 心跳 + 状态快照。MCU → OP 时 `visual_state` 恒为 0 |
| **0x16** | `TYPE_REQUEST_SYNC` | 双向 | `requested_state(1B)` | 请求对方强制同步到某状态 |
| **0x17** | `TYPE_VISUAL_SERVO_DATA` | OP → MCU | `error_x(2B LE) + error_y(2B LE) + flags(1B) + state(1B)` | 高频视觉伺服数据（每帧必发） |
| **0x18** | `TYPE_EMERGENCY_STOP` | 双向 | `reason(1B)` | 任意一方急停 |

---

## 2. 子命令（TYPE_CMD_FROM_MCU）

| cmd_id | 名称 | 参数 | 说明 |
|:---|:---|:---|:---|
| 0x01 | `CMD_START_QR` | 无 | 一键启动，开启 QR 检测 |
| 0x06 | `CMD_STOP_VISUAL` | 无 | 紧急停止所有视觉任务 |
| 0x07 | `CMD_START_RING_DISCOVERY` | `color_id(1B)` | 启动色环发现，指定目标颜色 |
| 0x08 | `CMD_DISCOVERY_DONE` | 无 | 三色映射完成，退出发现阶段 |
| 0x09 | `CMD_START_CARGO_STACKING_DISCOVERY` | `color_id(1B)` | 启动码垛货物发现，指定目标颜色。MCU → OP，用法与 CMD_START_RING_DISCOVERY 相同 |

> **注意**：视觉任务的日常启停由 Orange Pi 根据 MissionSM 状态变化自动管理，MCU 不需要发送 CMD 来开关任务。详见 `docs/architecture/state_machine.md` §2 区域自动控制。

---

## 3. VisualFlags（STATUS_FROM_VISION / VISUAL_SERVO_DATA 共用的 flags 字节）

```
bit 0: TARGET_FOUND      当前帧检测到目标
bit 1: READY_TO_PICK     偏差足够小，可以抓取
bit 2: READY_TO_PLACE    偏差足够小，可以放置
bit 3: VISUAL_FAIL       视觉异常（超时/丢失）
bit 4: QR_OK             二维码已识别
bit 5: CARGO_CONFIRMED   视觉确认载台有物料
bit 6: COLOR_MISMATCH    检测到的颜色与预期不符
bit 7: RING_CENTERED      色环已居中（发现阶段使用）
```

---

## 4. action_id 映射（TYPE_ACTION_DONE）

每完成一个动作，MCU 发一次 `ACTION_DONE`，`action_id` 标识动作类型（代码层使用 `ActionId` 枚举，见 `protocol.py`）：

| action_id | 枚举名 | 对应动作 | 触发事件 | 每批次循环次数 |
|:---|:---|:---|:---|:---|
| 1 | `ActionId.PICK_RAW` | `PICK_RAW` — 原料区取料 | `PICK_DONE` | 3（每次取 1 个物料） |
| 2 | `ActionId.PLACE_ROUGH` | `PLACE_ROUGH` — 粗加工区放料 | `place_action_done` 标志 | 3（每次放 1 个物料） |
| 3 | `ActionId.PICK_ROUGH` | `PICK_ROUGH` — 粗加工区取料（回收入机器人） | `PICK_DONE` | 3（每次取 1 个物料） |
| 4 | `ActionId.PLACE_TEMP` | `PLACE_TEMP` — 暂存区放料/码垛 | `place_action_done` 标志 | 3（每次放 1 个物料） |

> **注意**：`action_id=2/3`（PLACE）不触发事件转换，而是通过 `place_action_done` 标志让状态机的 `on_execute` 决定下一步。详见 `docs/architecture/state_machine.md` §2 混合模型。

---

## 5. 结果码

| 值 | 名称 | 含义 |
|:---|:---|:---|
| 0x00 | `ACTION_OK` | 成功 |
| 0x01 | `ACTION_BUSY` | 执行中 |
| 0x02 | `ACTION_TIMEOUT` | 超时 |
| 0x03 | `ACTION_FAIL` | 失败 |
| 0x04 | `ACTION_NO_CARGO` | 未检测到物料 |

---

## 6. 典型通信时序

### 6.1 标准流程（区域自动控制）

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
  → VISUAL_SERVO_DATA flags=ready_to_pick
MCU 抓取 → ACTION_DONE action_id=ActionId.PICK_RAW (1), result=OK
OP 推进 step，继续下一轮 ...

MCU 到达 ROUGH → TYPE_ARRIVED zone=3
OP 自动进入 RING_DISCOVERY
  → MCU 移动爪到中央参考位置
  → 对 R/G/B 各发一次 CMD_START_RING_DISCOVERY
  → OP 输出 VISUAL_SERVO_DATA，MCU PID 微调
  → 环居中后 OP 发送 TYPE_COLOR_RESULT，MCU 记录电机位置
  → MCU 完成三色映射 → CMD_DISCOVERY_DONE
OP: RING_DISCOVERY → ALIGN_ROUGH → PLACE_ROUGH（瞬时级联，无视觉伺服）
MCU 靠 mapping + 惯导到达目标位置 → ACTION_DONE action_id=ActionId.PLACE_ROUGH (2), result=OK
OP 重复 ALIGN+PLACE 直到 cargo_count=0
OP 切换到 PICK 阶段（picking_from_rough=True）

MCU 到达 ROUGH 色环前（取回阶段，空载 crane）
OP ALIGN_ROUGH → PICK_ROUGH（瞬时级联，无视觉伺服）
MCU 靠 mapping + 惯导到达目标位置 → ACTION_DONE action_id=ActionId.PICK_ROUGH (3), result=OK
OP 重复 ALIGN+PICK 直到 cargo_count=3

MCU 到达 TEMP → TYPE_ARRIVED zone=4
OP 自动进入 RING_DISCOVERY（流程同 ROUGH）
  → 发现完成 → ALIGN_TEMP → PLACE_TEMP（瞬时级联，无视觉伺服）
MCU 靠 mapping + 惯导到达目标位置 → ACTION_DONE action_id=ActionId.PLACE_TEMP (4), result=OK
OP 重复 ALIGN+PLACE 直到 cargo_count=0

OP → RETURN_HOME → FINISHED
```

关键要点：
- **MCU 不需要发 CMD 来开关视觉任务** — OP 自动根据状态切换
- **MCU 不需要发 CMD 来传递颜色** — OP 从 `batch_order` 自动获取
- MCU 只需发 `TYPE_ARRIVED` 告知到达区域，发 `ACTION_DONE` 告知动作完成
- **`ARRIVED_*` 同步职责**：MCU 决定前往某区域后，更新自身 MissionSM + 发 `TYPE_ARRIVED`；
  OP 收到后更新镜像 MissionSM。双方状态机保持同步。

### 6.2 视觉伺服数据流

```
OP 每帧发送 VISUAL_SERVO_DATA（无论是否检测到目标）：
  error_x: int16 LE（归一化 -5000~5000，0=画面中心。未检出时为 0）
    └─ OP 内部: 处理器直接输出 int[-5000, 5000]，不经转换直接编码发送
  error_y: int16 LE（同上）
  flags: 1B（同 VisualFlags 位定义，包含 TARGET_FOUND / READY_TO_PICK / READY_TO_PLACE 等）
  state: 1B（当前 visual_state: 0=IDLE, 1=SEARCH, 2=TRACKING, 3=RECOVERY, 4=FAIL）

MCU 根据每帧 flags 直接决策：
  flags & READY_TO_PICK  → 执行抓取
  flags & READY_TO_PLACE → 执行放置
  flags & TARGET_FOUND   → PID 微调（使用 error_x/error_y）
  否则                   → 保持/回退
```

> **注意**：不再通过 STATUS_FROM_VISION 单独发送 READY 信号。每帧 SERVO_DATA 自带 flags，MCU 仅需解析一个帧即可做出所有决策，消除时序竞争。

#### READY 标志锁存

OP 进入 stable tracking（连续 N 帧稳定检测到目标）后，锁存 `READY_TO_PICK` 或 `READY_TO_PLACE` 标志。锁存期间**每帧持续发送**该标志，不受视觉噪声干扰（目标短暂丢失不解除）。

锁存仅在以下条件清除：
| 清除条件 | 说明 |
|---|---|
| MCU 回应 `ACTION_DONE`（result=OK） | 动作完成，解除锁存 |
| OP 激活新视觉任务（`_activate_task`） | 切换到新的跟踪目标 |
| OP 停用视觉（`_deactivate_all_visual`） | 进入导航 / 错误状态 |

> **RAW 原料区风险**：圆盘旋转供料，MCU 收到 READY 后若机械臂动作期间目标随圆盘转走，可能抓空。此时 MCU 应自行判断失败 → ERROR 恢复，或在下一次圆盘停顿时重新锁定。

### 6.3 心跳（纯监控信号，不参与决策）

```
双方每 100ms 互发 HEARTBEAT：
  seq: 1B（递增，0~255 循环）
  mission_state: 1B（发送方的当前任务状态 ID）
  visual_state: 1B（发送方当前视觉状态 ID，MCU→OP 时恒为 0）

心跳仅用于维护 is_linked 标记（调试/日志用途）：
  - 收到任何有效心跳 → is_linked = true
  - 连续 3 次（300ms）未收到对方心跳 → is_linked = false
  - 链路恢复后自动重置为 true
  - 不会触发 EMERGENCY_STOP、切 ERROR 或任何业务动作

两侧均提供 Vision_Protocol_IsLinkActive() 供调试面板使用。
```

### 6.4 错误恢复

```
MCU/OP 发现异常 → 切 ERROR
  → 发 STATUS_FROM_VISION state=ERROR, flags=visual_fail
  → 发 REQUEST_SYNC state=ERROR
对方收到后也切 ERROR

人工修复后 → 发 RESET（或 CMD_STOP_VISUAL 后重新 CMD_START_QR）
   → 状态机回到 WAIT_START
```

### 6.5 色环发现流程（RING_DISCOVERY）

到达 ROUGH/TEMP 区后执行，用于建立颜色→世界坐标映射。

```
MCU 到达 ROUGH/TEMP 区 → TYPE_ARRIVED zone=ROUGH/TEMP
OP 自动进入 RING_DISCOVERY 状态

MCU 移动爪到中央参考位置（该位置 camera 画面中可见全部三个色环）

FOR each color (R → G → B):
  MCU → OP: CMD_START_RING_DISCOVERY color=X
  OP 激活 ring_discovery 任务，锁定 target_color=X
  OP 每帧 VISUAL_SERVO_DATA(error_x, error_y, TARGET_FOUND)
  MCU PID 微调爪位置，error → 0
  OP 连续 10 帧稳定 → RING_CENTERED flag
  OP → MCU: TYPE_COLOR_RESULT(color_id=X, confidence)
  MCU 记录 mapping[X] = 当前电机位置
  MCU 回到中央参考位置  ← 必须回参考位置后换下一颜色

MCU → OP: CMD_DISCOVERY_DONE
OP: RING_DISCOVERY → ALIGN_ROUGH/TEMP → PLACE_ROUGH/TEMP（瞬时级联）

后续 PLACE/PICK 全部靠 MCU mapping + 惯导，OP 不做视觉伺服。
```

> **约束**：
> - 每次进入 ROUGH 或 TEMP 区（包括 Batch 2）都必须重新发现。同一个 batch/zone 内不需要重复。
> - 换颜色前 MCU 必须回到中央参考位置。
> - TEMP Batch 2：色环被 Batch 1 货物遮挡，OP 检测货物底部轮廓（与色环中心同位置）。
> - 发现超时 30s → ERROR(50)。

---

## 7. 同步规则

1. **谁触发转换，谁负责通知对方**。
   - MCU 触发的转换（到达区域、执行完成）→ MCU 发 `TYPE_ARRIVED` / `TYPE_ACTION_DONE`
   - OP 触发的转换（QR 识别成功、目标丢失、可放置）→ OP 发 `TYPE_STATUS_FROM_VISION`

2. **双方都维护一份 MissionSM**。

3. **定期心跳 `TYPE_HEARTBEAT`**。丢失 3 个心跳认为通信中断 → `ERROR`。

4. **状态不一致时 `REQUEST_SYNC`**。双方切 `ERROR` 并停车。

---

## 8. 代码映射

| 协议元素 | Python 常量 | 位置 |
|---|---|---|
| 帧类型 | `TYPE_ERROR` ~ `TYPE_EMERGENCY_STOP` | `protocol.py` |
| 子命令 | `CMD_START_QR`, `CMD_STOP_VISUAL` | `protocol.py` |
| 标志位 | `class VisualFlags` | `protocol.py` |
| 结果码 | `ACTION_OK` ~ `ACTION_NO_CARGO` | `protocol.py` |
| 帧构建 | `build_*_frame()` 系列函数 | `protocol.py` |
| 帧解析 | `parse_*_payload()` 系列函数 | `protocol.py` |

> 通用帧结构（SOF/Length/Checksum）的构建解析见 [`protocol.md`](protocol.md) §5。

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v2.0** | 2026-07-26 | 从 `protocol.md` 独立为内容定义文档 |
| **v1.3** | 2026-07-18 | 新增 `CMD_START_CARGO_STACKING_DISCOVERY(0x09)` |
| **v1.2** | 2026-07-04 | 删除多组不再使用的 CMD，时序补充 |
| **v1.1** | 2026-07-03 | COLOR_RESULT/CMD 参数调整 |
| **v1.0** | — | 初版内容定义 |
