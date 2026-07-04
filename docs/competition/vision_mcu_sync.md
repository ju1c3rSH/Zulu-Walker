# 工创赛智能物流搬运：视觉-电控同步状态机与通信协议

> ⚠️ **本文件已拆分**
>
> - **通信协议** → 参见 [`docs/competition/protocol.md`](protocol.md)（更新中）
> - **状态机设计** → 参见 [`docs/architecture/state_machine.md`](../architecture/state_machine.md)（更新中）
>
> 本文档保留作为历史参考和完整追溯，协议和状态机定义以新文档为准。

> 适用项目：Zulu-Walker（Orange Pi 5B + STM32）
> 目标：让香橙派（视觉/决策）与 STM32（运动/执行）在赛场上“步调一致”

---

## 一、任务逻辑梳理

### 1.1 比赛核心流程（初赛）

```
一键启动
    │
    ▼
[读二维码] ──► 获取任务码，例如 "123+231"
    │
    ▼
[第一批：原料区 → 粗加工区]
    │   按第一组三位顺序（如 1红→2绿→3蓝）依次抓取 3 个物料
    │   每次最多装 3 个，必须全部装载到机器人上才能离开原料区
    ▼
[粗加工区放置]
    │   按颜色放到对应色环区域
    ▼
[转运：粗加工区 → 暂存区]
    │   按“第一批被抓取的顺序”把 3 个物料搬到暂存区对应颜色区域
    ▼
[第二批：原料区 → 粗加工区]
    │   按第二组三位顺序依次抓取 3 个物料
    ▼
[粗加工区放置]
    │
    ▼
[码垛：粗加工区 → 暂存区]
    │   按第二批顺序，把物料码垛到暂存区“同颜色、已正确放置”的物料上
    ▼
[返回启停区]
```

### 1.2 关键约束与边界条件

| 约束 | 对状态机/协议的影响 |
|:---|:---|
| 完全自主，禁止赛中通信 | 所有决策必须在赛前写入；协议只用于板间同步，不接受外部指令 |
| 物料必须放在机器人上运输 | 状态机必须区分“抓取中 / 已装载 / 放置中”，只有 `LOADED` 才能触发区域切换 |
| 每次装载 ≤ 3 个 | 计数器 `cargo_count` 在视觉/电控间同步 |
| 二维码决定顺序 | 视觉读完 QR 后，把两组任务码发给电控，电控据此生成任务队列 |
| 颜色识别在转盘上完成 | 转盘停顿时 MCU 通过 `CMD_START_COLOR_DETECT` 请求识别当前物料颜色；MCU 自行跟踪转盘位置，视觉不报告槽位 |
| Cargo 摄像头视角中物料呈现为圆形（顶面） | 检测算法以圆/椭圆/四边形检测为主（复用 `CircleTargetDetector`）；跟踪阶段输出圆心坐标 + 偏差 + 距离 |
| 色环放置精度决定得分 | 视觉在放置前给出精细偏差（视觉伺服），电控微调 |
| 机器人有 3 个物料槽位，每槽固定对应一种颜色 | Slot 0=RED, Slot 1=GREEN, Slot 2=BLUE；抓取时视觉确认颜色与目标槽位匹配（`COLOR_MISMATCH` 检测），装载时按 `slot_index` 存放；模型层 `CargoItem.slot_index` 根据颜色自动计算 |

### 1.3 视觉与电控的职责划分

| 职责 | Orange Pi（视觉+高层决策） | STM32（电控+执行） |
|:---|:---|:---|
| **定位/导航** | 识别二维码板、色环、物料；输出偏差 | 里程计、PID、路径规划、底盘运动 |
| **抓取/放置** | 识别目标颜色/位置，输出机械臂偏差 | 控制机械臂、吸盘/夹爪、升降 |
| **任务顺序** | 解析 QR，把任务码回传 | 维护任务队列，决定下一步去哪个区 |
| **状态同步** | 维护视觉子状态机，上报识别结果 | 维护任务主状态机，请求视觉服务 |
| **安全/超时** | 视觉丢失、识别失败告警 | 运动超时、碰撞检测、急停 |

**核心原则**：
- **STM32 是任务主状态机的“主控”**（它知道走到哪里了、执行机构是否完成）。
- **Orange Pi 是视觉子状态机的“主控”**（它决定何时找到目标、何时可放置）。
- 双方通过“命令-响应”协议协商推进，但任何一方发现危险都可以把状态机切到 `ERROR`。

---

## 二、双层状态机设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Mission State Machine                    │
│                     (任务级，STM32 主导)                      │
│  IDLE -> WAIT_START -> READ_QR -> NAV_TO_RAW -> ALIGN_RAW ->  │
│  PICK -> CHECK_LOAD -> NAV_TO_ROUGH -> ALIGN_ROUGH -> PLACE ->│
│  ... -> FINISHED / ERROR                                      │
└─────────────────────────────────────────────────────────────┘
                              │ 请求视觉服务 / 接收结果
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Visual Servo State Machine                 │
│                   (视觉级，Orange Pi 主导)                    │
│       IDLE -> SEARCH -> TRACKING -> RECOVERY -> FAIL          │
│       + QR_READ / COLOR_DETECT / PLACE_GUIDE 等扩展视觉任务   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Mission State（任务主状态）

状态名尽量做到“一看就懂当前在干什么”。

| 状态名 | 含义 | 视觉侧行为 | 电控侧行为 |
|:---|:---|:---|:---|
| `IDLE` | 上电初始化 | 等待启动 | 等待启动 |
| `WAIT_START` | 一键启动前就绪 | 可显示状态码 | 蜂鸣/灯提示，等待按键 |
| `READ_QR` | 读取二维码 | **QR_READ** 任务，识别到后发送 `QR_RESULT` | 移动到二维码板前，等待结果 |
| `NAV_TO_RAW` | 前往原料区 | 进入 `SEARCH` 搜索物料/转盘 | 按里程计行驶 |
| `ALIGN_RAW` | 在原料区对准 | `TRACKING` 输出色环/物料偏差 | 根据偏差微调 |
| `PICK_RAW` | 抓取物料 | 识别当前被抓取物颜色，确认对准 | 控制机械臂抓取 |
| `CHECK_LOAD` | 确认已装载 | 检测机器人载台是否有物料 | 读取传感器/称重 |
| `NAV_TO_ROUGH` | 前往粗加工区 | 搜索色环 | 按路径行驶 |
| `ALIGN_ROUGH` | 对准粗加工区色环 | `TRACKING` 输出色环偏差 | 微调 |
| `PLACE_ROUGH` | 放置物料 | 确认释放后物料位置 | 控制机械臂放置 |
| `NAV_TO_TEMP` | 前往暂存区 | 搜索色环 | 按路径行驶 |
| `ALIGN_TEMP` | 对准暂存区色环 | `TRACKING` 输出色环偏差 | 微调 |
| `PLACE_TEMP` | 放置/码垛 | 确认同颜色目标、输出码垛偏差 | 放置 |
| `RETURN_HOME` | 返回启停区 | 可关闭视觉省电 | 返回 |
| `FINISHED` | 任务完成 | 显示完成 | 停机、蜂鸣 |
| `ERROR` | 异常 | 记录错误码 | 停止或执行恢复策略 |

### 2.3 状态如何同步？——“请求-服务”模型

```
STM32 侧                    Orange Pi 侧
─────────                   ────────────
 Mission SM                  Mission SM（镜像）
     │                            │
     │  1. 发送 CMD:START_QR       │
     │────────────────────────────>│
     │                            ▼
     │                    Visual SM: IDLE -> QR_READ
     │                            │
     │  2. 发送 QR_RESULT          │
     │<────────────────────────────│
     │                            ▼
     ▼                    Mission SM 进入 NAV_TO_RAW
```

**同步规则**：
1. **谁触发转换，谁负责通知对方**。
   - 电控触发的转换（到达区域、执行完成）→ 电控发 `STATUS_FROM_MCU`。
   - 视觉触发的转换（QR 识别成功、目标丢失、可放置）→ 视觉发 `STATUS_FROM_VISION`。
2. **双方都维护一份 Mission SM**。收到对方的状态报文后，如果本地状态落后或冲突，以“更靠后”的状态为准（除非跳到 `ERROR`）。
3. **定期心跳 `HEARTBEAT`**。丢失 3 个心跳（默认 300ms 周期）即认为通信中断，进入 `ERROR`。
4. **状态不一致时进入 `ERROR`**。例如电控在 `PICK_RAW`，视觉报告 `VISUAL_FAIL`，双方切 `ERROR` 并停车。

### 2.4 Visual Servo State（视觉子状态）

复用项目已有的 `VisualStateMachine`（`IDLE / SEARCH / TRACKING / RECOVERY / FAIL`），但在 Mission SM 的不同阶段启用不同视觉任务：

| Mission 阶段 | 视觉任务 | Visual SM 状态 |
|:---|:---|:---|
| `READ_QR` | `qr_detect` | `SEARCH` → QR 成功 → `IDLE` |
| `ALIGN_RAW` | `color_detect` + `target_track` | `SEARCH` → `TRACKING` |
| `PICK_RAW` | `target_track`（确认在夹爪中心） | `TRACKING` |
| `ALIGN_ROUGH` / `ALIGN_TEMP` | `color_ring_track` | `SEARCH` → `TRACKING` |
| `PLACE_TEMP`（码垛） | `top_target_track` | `SEARCH` → `TRACKING` |

---

## 三、视觉-电控双向通信协议

### 3.1 帧格式（保留现有二进制帧）

沿用 `modules/zw_uart_module/protocol.py` 已有的格式：

```
| SOF | Length | Type | Payload | Checksum |
|  1  |   1    |  1   |  0~252  |    1     |

SOF       = 0xAA
Length    = Type(1) + Payload(n) + Checksum(1)
Checksum  = Type 到 Payload 所有字节的异或（XOR）
```

### 3.2 帧类型定义

保留旧类型，新增任务同步类型。

| Type | 名称 | 方向 | Payload | 说明 |
|:---|:---|:---|:---|:---|
| 0x01 | `TYPE_ERROR` | OP → MCU | `error_type(1B) + error_value(2B LE)` | 保留，视觉伺服误差 |
| 0x02 | `TYPE_ARRIVED` | MCU → OP | `zone_id(1B)` | 保留，到达某区域 |
| 0x03 | `TYPE_PICK` | MCU → OP | `zone_id(1B)` | 保留，请求在某区抓取 |
| 0x04 | `TYPE_SET` | MCU → OP | `zone_id(1B)` | 保留，设置当前区域 |
| **0x10** | `CMD_FROM_MCU` | MCU → OP | `cmd_id(1B) + args` | 电控请求视觉服务 |
| **0x11** | `STATUS_FROM_VISION` | OP → MCU | `mission_state(1B) + visual_state(1B) + flags(1B) + cargo_count(1B)` | 视觉上报综合状态 |
| **0x12** | `QR_RESULT` | OP → MCU | `len(1B) + ascii[len]` | 二维码任务码 |
| **0x13** | `COLOR_RESULT` | OP → MCU | `color_id(1B) + confidence(1B)` | 当前物料颜色识别结果 |
| **0x14** | `ACTION_DONE` | MCU → OP | `action_id(1B) + result(1B)` | 电控动作完成 |
| **0x15** | `HEARTBEAT` | 双向 | `seq(1B) + mission_state(1B) + visual_state(1B)` | 心跳 + 状态快照。MCU → OP 时 `visual_state` 恒为 0 |
| **0x16** | `REQUEST_SYNC` | 双向 | `requested_state(1B)` | 请求对方强制同步到某状态 |
| **0x17** | `VISUAL_SERVO_DATA` | OP → MCU | `error_x(2B LE) + error_y(2B LE) + distance(2B LE) + state(1B)` | 高频视觉伺服数据 |
| **0x18** | `EMERGENCY_STOP` | 双向 | `reason(1B)` | 任意一方急停 |

### 3.3 命令子类型（`CMD_FROM_MCU` 的 `cmd_id`）

| cmd_id | 名称 | 参数 | 视觉响应 |
|:---|:---|:---|:---|
| 0x01 | `CMD_START_QR` | 无 | 启动 QR 检测，成功后回 `QR_RESULT` |
| 0x02 | `CMD_START_COLOR_DETECT` | 无 | 识别当前所在位置的物料颜色，回 `COLOR_RESULT` |
| 0x03 | `CMD_TRACK_TARGET` | `color_id(1B)` | 进入 `SEARCH/TRACKING` 跟踪指定颜色目标 |
| 0x04 | `CMD_TRACK_RING` | `color_id(1B)` | 跟踪指定颜色色环 |
| 0x05 | `CMD_TRACK_TOP` | `color_id(1B)` | 码垛时跟踪顶层目标 |
| 0x06 | `CMD_STOP_VISUAL` | 无 | 视觉进入 `IDLE` |

### 3.4 `STATUS_FROM_VISION` 标志位（`flags`）

```
bit 0: target_found      当前帧检测到目标
bit 1: ready_to_pick     偏差足够小，可以抓取
bit 2: ready_to_place    偏差足够小，可以放置
bit 3: visual_fail       视觉异常（超时/丢失）
bit 4: qr_ok             二维码已识别
bit 5: cargo_confirmed   视觉确认载台有物料
bit 6: color_mismatch    颜色与预期不符
bit 7: reserved
```

### 3.5 错误/结果码

| 值 | `action_id`（`ACTION_DONE`） | `result` |
|:---|:---|:---|
| 0x00 | `ACTION_OK` | 成功 |
| 0x01 | `ACTION_BUSY` | 执行中 |
| 0x02 | `ACTION_TIMEOUT` | 超时 |
| 0x03 | `ACTION_FAIL` | 失败 |
| 0x04 | `ACTION_NO_CARGO` | 未检测到物料 |

### 3.6 典型通信时序

#### 时序 A：读取二维码

```
MCU:  CMD_FROM_MCU  CMD_START_QR
OP :  STATUS_FROM_VISION  state=READ_QR, flags=qr_ok
OP :  QR_RESULT  "123+231"
MCU:  ACTION_DONE  action_id=QR_READ, result=OK
```

#### 时序 B：抓取第一个物料（红色）

```
MCU:  CMD_FROM_MCU  CMD_START_COLOR_DETECT
OP :  COLOR_RESULT  color_id=RED, confidence=95
MCU:  CMD_FROM_MCU  CMD_TRACK_TARGET color_id=RED
OP :  STATUS_FROM_VISION  state=ALIGN_RAW, visual_state=SEARCH
OP :  VISUAL_SERVO_DATA  error_x=..., error_y=..., state=TRACKING
...（持续发送，电控 PID 微调）...
OP :  STATUS_FROM_VISION  flags=ready_to_pick
MCU:  ACTION_DONE  action_id=PICK, result=OK     # MCU 自动放入对应 SLOT
MCU:  HEARTBEAT  seq=..., mission_state=CHECK_LOAD, visual_state=0
OP :  STATUS_FROM_VISION  flags=cargo_confirmed
MCU:  HEARTBEAT  seq=..., mission_state=NAV_TO_ROUGH, visual_state=0
```

#### 时序 C：心跳丢失 -> 错误恢复

```
MCU 连续 3 次未收到 HEARTBEAT
MCU -> 切 ERROR，停车、松开夹爪
MCU 发送 REQUEST_SYNC state=ERROR
OP  收到后切 ERROR，等待人工 RESET
```

---

## 四、代码实现建议

### 4.1 文件组织

```
context/
    event_bus.py                 # 已建：类型化 pub/sub
    mission_state_machine.py     # 已建：任务级状态机（从 utils/ 搬出）
    mission_context.py           # 已建：中枢协调器

utils/state_machine/
    base.py                      # 已有：通用状态机基类
    visual_state_machine.py      # 已有：视觉跟踪状态机

modules/zw_opencv_module/models/
    color.py                     # 已建：Color 枚举
    cargo.py                     # 已建：CargoItem, CargoSet, CargoZone

modules/zw_uart_module/
    protocol.py                  # 帧类型 + VisualFlags + build/parse 函数
    uart_driver.py               # UART 收发 + FrameParser
```

### 4.2 MissionStateMachine 关键接口

```python
class MissionStateMachine(BaseStateMachine):
    class States:
        IDLE = 0
        WAIT_START = 1
        READ_QR = 2
        NAV_TO_RAW = 3
        ALIGN_RAW = 4
        PICK_RAW = 5
        CHECK_LOAD = 6
        NAV_TO_ROUGH = 7
        ALIGN_ROUGH = 8
        PLACE_ROUGH = 9
        NAV_TO_TEMP = 10
        ALIGN_TEMP = 11
        PLACE_TEMP = 12
        NAV_TO_RAW_SECOND = 13
        # ... 第二批类似，可参数化
        RETURN_HOME = 20
        FINISHED = 21
        ERROR = 22

    def on_qr_result(self, qr_str: str): ...
    def on_arrived(self, zone_id: int): ...
    def on_action_done(self, action_id: int, result: int): ...
    def on_visual_status(self, state: int, flags: int): ...
```

### 4.3 双方启动流程

1. **上电**：
   - Orange Pi 启动相机、UART、任务管理器，进入 `MissionState.IDLE` / `VisualState.IDLE`。
   - STM32 初始化底盘、机械臂、UART，进入 `MissionState.IDLE`。
2. **握手**：
   - 双方互发 `HEARTBEAT`，确认 alive。
   - Orange Pi 发送一次 `STATUS_FROM_VISION state=IDLE`。
3. **一键启动**：
   - 队员按键 → STM32 切 `WAIT_START` → `READ_QR`，发 `CMD_START_QR`。
4. **运行**：按 Mission SM 推进。

### 4.4 调试技巧

- 在 Orange Pi 上把每次状态转换和 UART 收发打印到日志 `/tmp/zw_mission.log`。
- 用 `run.py debug` 时叠加显示当前 `MissionState` 和 `VisualState`。
- 赛场上把 `HEARTBEAT` 周期设 100ms，视觉伺服数据 30~50ms 一帧。

---

## 五、常见问题速查

| 问题 | 建议做法 |
|:---|:---|
| 二维码读不出来 | 在 `READ_QR` 状态让 STM32 做小幅左右摆动；视觉多帧确认后再发 `QR_RESULT` |
| 转盘颜色识别错 | 赛前在场地实际灯光下标定 `uv_params.yaml`，用 `COLOR_RESULT.confidence` 做阈值过滤 |
| 色环对准偏差大 | 把色环跟踪和物料跟踪分开：粗对准用色环中心，精放置用物料底部投影 |
| 码垛倒塌 | 在 `PLACE_TEMP` 用视觉确认“目标正上方无遮挡、同颜色”后再下降 |
| 通信偶发丢包 | 所有关键命令要求 `ACTION_DONE` 确认；状态报文带序列号，重复只处理一次 |
| 状态机跑偏 | 任何一方收到 `REQUEST_SYNC` 必须回复当前状态；不一致时双方切 `ERROR` |

---

## 六、版本与维护

### 版本历史

| 版本 | 日期 | 变更 |
|:---|:---|:---|
| **v1.1** | 2026-07-03 | ① `COLOR_RESULT` 移除 `slot_idx`（MCU 自行跟踪转盘位置）；② `CMD_START_COLOR_DETECT` 移除 `slot_idx` 参数；③ 移除 `CMD_SET_EXPOSURE`（OP 自主管理相机）；④ 修复时序 B：新增颜色检测步骤，`STATUS_FROM_MCU` → `HEARTBEAT`；⑤ `HEARTBEAT` 增加 MCU→OP 时 `visual_state=0`；⑥ 新增机器人 3 槽位固定颜色约束（Slot 0=R, 1=G, 2=B）；⑦ 新增 `CargoItem`/`CargoSet`/`CargoZone` 模型层；⑧ 删除 `utils/state.py` 废弃代码 |
| **v1.0** | — | 初版协议与状态机定义 |

### 变更清单（增加新视觉任务/新区域时）

当增加新视觉任务或新区域时，按以下清单更新：

1. **协议层** — `modules/zw_uart_module/protocol.py`
   - 新增 `CMD_FROM_MCU` 子命令或帧类型（含 build/parse 函数）
   - 新增帧后同步更新本文档 §3.2~§3.3
2. **状态机层** — `utils/state_machine/mission_state_machine.py`
   - 在 `MissionState` 新增状态 ID（须与 STM32 固件同步）
   - 在 `MissionStateMachine.Events` 新增事件 + 注册转换
3. **模型层** — `modules/zw_opencv_module/models/`
   - 新增或扩展 `CargoItem`、`CargoSet`、`Zone` 等数据模型
4. **STM32 固件** — `c/` 或 STM32 项目
   - 同步帧类型枚举、状态 ID 枚举、子命令枚举
5. **本文档** — 更新对应章节并追加版本记录
