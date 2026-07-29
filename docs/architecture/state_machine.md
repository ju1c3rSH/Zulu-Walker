# 统一状态机设计

> 适用项目：Zulu-Walker（Orange Pi 5B + STM32）
> 用途：香橙派（视觉/决策）与 STM32（运动/执行）的双层状态机协同规范

---

## 1. 架构总览

```
┌───────────────────────────────────────────────────────┐
│                   Orange Pi (OP)                        │
│                                                         │
│  ┌────────────────┐     ┌───────────────────────────┐  │
│  │ MissionSM (镜像) │◄───►│      VisualSM (主控)      │  │
│  │  18 states      │     │  IDLE/SEARCH/TRACKING/    │  │
│  │                 │     │  RECOVERY/FAIL            │  │
│  └───────┬─────────┘     └───────────────────────────┘  │
│          │                                                │
│          │ UART 帧 (0x10~0x18)                            │
└──────────┼────────────────────────────────────────────┘
           │
┌──────────┼────────────────────────────────────────────┐
│          ▼                                                │
│                   STM32 (MCU)                             │
│  ┌────────────────┐     ┌───────────────────────────┐  │
│  │ MissionSM (主控) │     │    VisualSM (镜像)         │  │
│  │  18 states      │     │   (心跳中的 visual_state)  │  │
│  └────────────────┘     └───────────────────────────┘  │
│                                                         │
└───────────────────────────────────────────────────────┘
```

### 职责划分

| 角色 | MissionSM | VisualSM |
|------|-----------|----------|
| **STM32** | 主控 — 任务流推进、导航、执行机构 | 镜像 — 通过 `HEARTBEAT` 中 `visual_state` 了解视觉状态 |
| **Orange Pi** | 镜像 — 接收 MCU 事件同步状态 | 主控 — 视觉任务启停、检测结果、伺服数据 |

### 核心原则

1. **STM32 是 MissionSM 的主控** — 它知道走到哪里了、执行机构是否完成
2. **Orange Pi 是 VisualSM 的主控** — 它决定何时找到目标、何时可放置、切换什么视觉任务
3. **双方都维护一份 MissionSM 镜像** — 通过 UART 帧同步
4. **任何一方发现危险都可以切 ERROR**

---

## 2. 关键设计：区域自动控制

### 设计决策

视觉任务的启停 **不由 MCU 通过 CMD 控制**，而是由 OP 监听 MissionSM 状态变化自动管理。MCU 只需要通过 `TYPE_ARRIVED` 告知 OP 当前所处区域。

```
MCU 到达 RAW 区 → 发 TYPE_ARRIVED zone=2
OP 收到 → MissionSM 切换到 ALIGN_RAW
OP 自动检测到状态变化 → 开启 track_cargo
OP 从 batch_order[current_step] 获取颜色 → 传给 processor
```

### 状态→视觉任务映射表

| Mission 状态 | 自动启用的视觉任务 | 颜色来源 | 说明 |
|---|---|---|---|---|
| `READ_QR` | `qr_detect` | 无 | 二维码解码 |
| `ALIGN_RAW` | `track_cargo` | `batch_order[current_step]` | 取料时追踪货物 |
| `RING_DISCOVERY` | `ring_discovery` | MCU 指定（CMD） | 色环发现，建立颜色→坐标映射 |
| `ALIGN_ROUGH` | 无 | - | 靠 mapping + 惯导（camera 被 cargo 遮挡） |
| `ALIGN_TEMP` | 无 | - | 靠 mapping + 惯导（camera 被 cargo 遮挡） |
| 其他状态 | 无（关闭所有视觉任务） | - | - |

> **第二批处理**：仅 `NAV_TO_RAW_SECOND` 是独立状态（用于区分 TEMP→RAW 回程路径）。
> `ALIGN_*` / `PICK_*` / `PLACE_*` / `CHECK_LOAD` / `NAV_TO_ROUGH` / `NAV_TO_TEMP` 在两批次间复用，
> 行为由 `current_batch_order`（颜色顺序）和 `current_batch`（平放/码垛）参数化驱动。

### 对 MCU 的电控对接约定

> **MCU 不需要发送任何 CMD 来开启或关闭视觉任务。**
>
> OP 会在收到 `TYPE_ARRIVED` 后自动推进 MissionSM，检测到状态变化后自动启用对应的视觉任务。目标颜色由 OP 从 `batch_order` 中自动获取——该 `batch_order` 由 OP 在 QR 解析后通过 `STATUS_FROM_VISION` 同步给 MCU。
>
> MCU 只需做三件事：
> 1. 到达区域后发 `TYPE_ARRIVED zone_id`
> 2. 动作完成后发 `ACTION_DONE action_id`
> 3. 定期发/收 `HEARTBEAT` 传递状态快照（纯监控，不参与决策）

### 颜色同步机制

`batch_order` 由双方共同维护：

```
OP 解析 QR "123+231"
  → batch_order = [Color.RED, Color.GREEN, Color.BLUE]
  → 通过 STATUS_FROM_VISION 发回 MCU（QR_RESULT 帧）

每次 pick 成功后
  → current_step += 1
  → 下次 status 帧中携带新的 cargo_count 和 mission_state

MCU 通过 STATUS_FROM_VISION 同步获得 cargo_count 和当前状态
→ 自行推导当前 target_color = batch_order[cargo_count - 1]
```

> **注**：颜色信息不需要额外的 CMD 帧来传递。MCU 通过维护一个与 OP 同步的 `batch_order` 列表 + `cargo_count` 即可推导出当前目标颜色。

---

## 3. MissionStateMachine

### 状态列表（20 状态，0~19 连续，须与 STM32 固件同步）

| ID | 名称 | 含义 |
|----|------|------|
| 0 | `IDLE` | 上电初始化 |
| 1 | `WAIT_START` | 一键启动前就绪 |
| 2 | `READ_QR` | 读取二维码 |
| 3 | `NAV_TO_RAW` | 前往原料区 |
| 4 | `ALIGN_RAW` | 原料区对准 |
| 5 | `PICK_RAW` | 抓取物料 |
| 6 | `CHECK_LOAD` | 确认已装载 |
| 7 | `NAV_TO_ROUGH` | 前往粗加工区 |
| 8 | `ALIGN_ROUGH` | 粗加工区对准色环（放料/取回复用，无视觉伺服） |
| 9 | `PLACE_ROUGH` | 放置物料 |
| 10 | `PICK_ROUGH` | 粗加工区取回物料（放完后重新捡起） |
| 11 | `NAV_TO_TEMP` | 前往暂存区 |
| 12 | `ALIGN_TEMP` | 暂存区对准色环（无视觉伺服） |
| 13 | `PLACE_TEMP` | 放置/码垛物料（两批次复用） |
| 14 | `NAV_TO_RAW_SECOND` | 第二批前往原料区（TEMP→RAW 路径不同于首批） |
| 15 | `RETURN_HOME` | 返回启停区 |
| 16 | `FINISHED` | 任务完成 |
| 17 | `ERROR` | 异常 |
| 18 | `RING_DISCOVERY` | 色环发现：建立颜色→坐标映射 |
| 19 | `CARGO_DISCOVERY` | 码垛货物发现：第二批次 TEMP 区使用 CargoDetector 建立颜色→坐标映射 |

> **第二批策略**：仅 `NAV_TO_RAW_SECOND` 是独立状态。第二批的 ROUGH/TEMP 导航仍使用 `NAV_TO_ROUGH` / `NAV_TO_TEMP`，由 MCU 自行通过 `current_batch` 同步区分回程路径；`ALIGN_*` / `PICK_*` / `PLACE_*` / `CHECK_LOAD` 全部复用首批状态，行为由 `current_batch_order`（颜色顺序）和 `current_batch`（平放/码垛）参数化驱动。

### 事件列表

| 事件 | 触发者 | 用途 |
|---|---|---|
| `START` | MCU (`TYPE_CMD_FROM_MCU CMD_START_QR`) | IDLE → WAIT_START → READ_QR |
| `QR_OK` | OP (QR 解码成功) | READ_QR → NAV_TO_RAW |
| `ARRIVED_RAW` | MCU (`TYPE_ARRIVED zone=RAW`) | NAV_TO_RAW → ALIGN_RAW |
| `READY_TO_PICK` | OP (`VISUAL_SERVO_DATA flags`) | ALIGN_RAW → PICK_RAW |
| `PICK_DONE` | MCU (`ACTION_DONE action_id=ActionId.PICK_RAW`) | PICK_RAW → CHECK_LOAD |
| `LOAD_CONFIRMED` | OP/传感器 | CHECK_LOAD → NAV_TO_ROUGH |
| `ARRIVED_ROUGH` | MCU (`TYPE_ARRIVED zone=ROUGH`) | NAV_TO_ROUGH → RING_DISCOVERY |
| `DISCOVERY_DONE` | OP (on_execute 内部决策) | RING_DISCOVERY → ALIGN_ROUGH / ALIGN_TEMP |
| `PLACE_DONE` | MCU (`ACTION_DONE action_id=ActionId.PLACE_ROUGH / ActionId.PLACE_TEMP`) | PLACE_* → 下一状态 |
| `ARRIVED_TEMP` | MCU (`TYPE_ARRIVED zone=TEMP`, batch=1) | NAV_TO_TEMP → RING_DISCOVERY |
| `ARRIVED_TEMP_BATCH2` | MCU (`TYPE_ARRIVED zone=TEMP`, batch=2) | NAV_TO_TEMP → CARGO_DISCOVERY |
| `ALL_PLACED` | OP/Coordinator | 批次完成 |
| `RETURNED_HOME` | MCU (`TYPE_ARRIVED zone=START`) | RETURN_HOME → FINISHED |
| `RESET` | MCU/OP | ERROR → WAIT_START |
| `ERROR` | MCU/OP | 任何状态 → ERROR |

> **`PLACE_DONE` 机制说明**：`PLACE_DONE` 事件不通过事件总线直接触发，而是通过 `place_action_done` 标志间接驱动：
> 1. `on_action_done()` 收到 `ACTION_DONE action_id=PLACE_ROUGH/PLACE_TEMP` 后，**仅设置** `ctx.cargo_count -= 1` 和 `ctx.place_action_done = True`，**不触发**任何状态转换（`return False`）
> 2. PLACE 状态的 `on_execute()` 在下一轮 `update()` 中检测到 `place_action_done == True`，清除该标志并根据 `cargo_count` 决定下一步跳转
>
> 这种"标志驱动 + on_execute 决策"的混合模型避免了 PLACE 状态在 MCU 动作完成瞬间立刻跳走、导致视觉伺服与机械动作脱节的问题。详见 §3 "批次逻辑与混合事件模型"。

### 转换图

```mermaid
stateDiagram-v2
    direction LR
    [*] --> IDLE
    IDLE --> WAIT_START: START
    WAIT_START --> READ_QR: START
    READ_QR --> NAV_TO_RAW: QR_OK

    %% ===== 第一批：取料 (RAW) =====
    NAV_TO_RAW --> ALIGN_RAW: ARRIVED_RAW
    ALIGN_RAW --> PICK_RAW: ready_to_pick
    PICK_RAW --> CHECK_LOAD: PICK_DONE
    CHECK_LOAD --> ALIGN_RAW: step<3
    CHECK_LOAD --> NAV_TO_ROUGH: step>=3

    %% ===== 第一批：粗加工 (放 → 取) =====
    NAV_TO_ROUGH --> RING_DISCOVERY: ARRIVED_ROUGH
    RING_DISCOVERY --> ALIGN_ROUGH: DISCOVERY_DONE
    ALIGN_ROUGH --> PLACE_ROUGH: on_execute (picking=False)
    PLACE_ROUGH --> ALIGN_ROUGH: cargo>0
    PLACE_ROUGH --> ALIGN_ROUGH: cargo=0 / picking=True
    ALIGN_ROUGH --> PICK_ROUGH: on_execute (picking=True)
    PICK_ROUGH --> CHECK_LOAD: PICK_DONE
    CHECK_LOAD --> ALIGN_ROUGH: step<3
    CHECK_LOAD --> NAV_TO_TEMP: step>=3

    %% ===== 第一批：暂存 =====
    NAV_TO_TEMP --> RING_DISCOVERY: ARRIVED_TEMP
    RING_DISCOVERY --> ALIGN_TEMP: DISCOVERY_DONE
    ALIGN_TEMP --> PLACE_TEMP: on_execute
    PLACE_TEMP --> ALIGN_TEMP: cargo>0
    PLACE_TEMP --> NAV_TO_RAW_SECOND: cargo=0 / batch=1 → 切 batch=2

    %% ===== 第二批：回程取料 (TEMP→RAW) =====
    NAV_TO_RAW_SECOND --> ALIGN_RAW: ARRIVED_RAW
    ALIGN_RAW --> PICK_RAW: ready_to_pick
    PICK_RAW --> CHECK_LOAD: PICK_DONE
    CHECK_LOAD --> ALIGN_RAW: step<3
    CHECK_LOAD --> NAV_TO_ROUGH: step>=3

    %% ===== 第二批：粗加工 (复用) =====
    NAV_TO_ROUGH --> RING_DISCOVERY: ARRIVED_ROUGH
    RING_DISCOVERY --> ALIGN_ROUGH: DISCOVERY_DONE
    ALIGN_ROUGH --> PLACE_ROUGH: on_execute (picking=False)
    PLACE_ROUGH --> ALIGN_ROUGH: cargo>0
    PLACE_ROUGH --> ALIGN_ROUGH: cargo=0 / picking=True
    ALIGN_ROUGH --> PICK_ROUGH: on_execute (picking=True)
    PICK_ROUGH --> CHECK_LOAD: PICK_DONE
    CHECK_LOAD --> ALIGN_ROUGH: step<3
    CHECK_LOAD --> NAV_TO_TEMP: step>=3

    %% ===== 第二批：暂存 — 码垛发现 (复用 PLACE_TEMP) =====
    NAV_TO_TEMP --> CARGO_DISCOVERY: ARRIVED_TEMP_BATCH2
    CARGO_DISCOVERY --> ALIGN_TEMP: DISCOVERY_DONE
    ALIGN_TEMP --> PLACE_TEMP: on_execute
    PLACE_TEMP --> ALIGN_TEMP: cargo>0
    PLACE_TEMP --> RETURN_HOME: cargo=0 / batch=2

    %% ===== 终态 =====
    RETURN_HOME --> FINISHED: RETURNED_HOME
    ERROR --> WAIT_START: RESET

    %% ===== 异常分支（简化） =====
    WAIT_START --> ERROR
    READ_QR --> ERROR
    NAV_TO_RAW --> ERROR
    ALIGN_RAW --> ERROR
    PICK_RAW --> ERROR
    CHECK_LOAD --> ERROR
    NAV_TO_ROUGH --> ERROR
    ALIGN_ROUGH --> ERROR
    PLACE_ROUGH --> ERROR
    PICK_ROUGH --> ERROR
    NAV_TO_TEMP --> ERROR
    ALIGN_TEMP --> ERROR
    PLACE_TEMP --> ERROR
    NAV_TO_RAW_SECOND --> ERROR
    RING_DISCOVERY --> ERROR
    CARGO_DISCOVERY --> ERROR
    RETURN_HOME --> ERROR
```

> **图例**：
> - 实线箭头：状态转换
> - 标签：触发条件（`ARRIVED_xxx` = MCU 事件；其他 = `on_execute` 决策）
> - 圆角矩形（默认）：MissionState 状态

> **第二批流程**：`_PlaceTempState` 后若 `current_batch == 1` 返回 `NAV_TO_RAW_SECOND`，
> 切 `current_batch = 2`、`current_batch_order = second_batch_order`。
> ALIGN/PICK/PLACE/CHECK 复用首批状态节点。

### 批次逻辑与混合事件模型

状态机采用**混合模型**：MCU 触发的事件（`ARRIVED_*`、`ACTION_DONE`）使用事件驱动即时转换；
内部决策（`ready_to_pick`、`cargo_count`、`picking_from_rough`）通过 `on_execute` 在 `update()` 中处理。

**完整一次 batch 的状态链**：

```
RAW 区:
  ALIGN_RAW → PICK_RAW → CHECK_LOAD [×3]
  cargo_count: 0→1→2→3, current_step: 0→1→2
  → NAV_TO_ROUGH (step=0 重置)

ROUGH 区 — 发现阶段:
  NAV_TO_ROUGH → RING_DISCOVERY → ALIGN_ROUGH

ROUGH 区 — 放料阶段 (picking_from_rough=False):
  ALIGN_ROUGH → PLACE_ROUGH [×3]
  cargo_count: 3→2→1→0
  放完后 → ALIGN_ROUGH (picking_from_rough=True, step=0 重置)

ROUGH 区 — 取料阶段 (picking_from_rough=True):
  ALIGN_ROUGH → PICK_ROUGH → CHECK_LOAD [×3]
  cargo_count: 0→1→2→3, current_step: 0→1→2
  取完后 → NAV_TO_TEMP (picking_from_rough=False, step=0 重置)

TEMP 区 — 发现阶段:
  NAV_TO_TEMP → RING_DISCOVERY → ALIGN_TEMP

TEMP 区 — 批次二发现阶段:
  NAV_TO_TEMP → CARGO_DISCOVERY → ALIGN_TEMP

TEMP 区 — 放置阶段（批次一/二复用）:
  ALIGN_TEMP → PLACE_TEMP [×3]
  cargo_count: 3→2→1→0
  batch=1 → NAV_TO_RAW_SECOND（切 batch=2）
  batch=2 → RETURN_HOME
```

**`on_execute` 决策逻辑**：

| State | 条件 | 下一步 |
|---|---|---|
| `_AlignRawState` | `ready_to_pick && !color_mismatch` | `PICK_RAW` |
| `_AlignRoughState` | `picking_from_rough` | `PICK_ROUGH` |
| `_AlignRoughState` | `!picking_from_rough` | `PLACE_ROUGH` |
| `_AlignTempState` | (无条件) | `PLACE_TEMP` |
| `_RingDiscoveryState` | `discovery_done && zone=ROUGH` | `ALIGN_ROUGH` |
| `_RingDiscoveryState` | `discovery_done && zone=TEMP` | `ALIGN_TEMP` |
| `_RingDiscoveryState` | `timeout 30s` | `ERROR` |
| `_CargoDiscoveryState` | `discovery_done` | `ALIGN_TEMP` |
| `_CargoDiscoveryState` | `timeout 360s` | `ERROR` |
| `_CheckLoadState` | `zone=RAW, step<3` | `ALIGN_RAW` |
| `_CheckLoadState` | `zone=RAW, step>=3` | `NAV_TO_ROUGH` |
| `_CheckLoadState` | `zone=ROUGH, step<3` | `ALIGN_ROUGH` |
| `_CheckLoadState` | `zone=ROUGH, step>=3` | `NAV_TO_TEMP` |
| `_PlaceRoughState` | `place_action_done && cargo>0` | `ALIGN_ROUGH` |
| `_PlaceRoughState` | `place_action_done && cargo==0` | `ALIGN_ROUGH (picking=True)` |
| `_PlaceTempState` | `place_action_done && cargo>0` | `ALIGN_TEMP` |
| `_PlaceTempState` | `place_action_done && cargo<0` | `RETURN_HOME` (打印异常) |
| `_PlaceTempState` | `place_action_done && cargo==0 && batch==1` | `NAV_TO_RAW_SECOND` (切 batch=2) |
| `_PlaceTempState` | `place_action_done && cargo==0 && batch==2` | `RETURN_HOME` |

#### 批次二处理策略

第一批结束后，`_PlaceTempState.on_execute` 检测 `cargo_count == 0`：
- `current_batch == 1 && second_batch_order` → 切换 `current_batch = 2`，`current_batch_order = second_batch_order`，`current_step = 0`，返回 `NAV_TO_RAW_SECOND`
- `current_batch == 2` → `RETURN_HOME`
- `cargo_count < 0`（异常）→ 打印错误日志，返回 `RETURN_HOME`

**第二批完整状态链**：

```
TEMP 放完第一批最后一个（_PlaceTempState，batch=1→2）
  → NAV_TO_RAW_SECOND
  → ALIGN_RAW（复用，current_batch_order 已切换为第二批颜色顺序）
  → PICK_RAW → CHECK_LOAD [×3]
   → NAV_TO_ROUGH（复用，MCU 自行根据 current_batch 区分路径）
   → RING_DISCOVERY → ALIGN_ROUGH 放料 [×3] → ALIGN_ROUGH 取回 [×3]
   → NAV_TO_TEMP（复用，MCU 自行根据 current_batch 区分路径）
   → CARGO_DISCOVERY → ALIGN_TEMP → PLACE_TEMP [×3]（复用，batch=2 时由 _PlaceTempState 决定终态）
  → RETURN_HOME → FINISHED
```

**状态复用 vs 独立决策**：

| 首批状态 | 是否有 _SECOND？ | 原因 |
|---|---|---|
| `NAV_TO_RAW` | **是** (`NAV_TO_RAW_SECOND`) | 回程路径 TEMP→RAW 不同于 START→RAW |
| `NAV_TO_ROUGH` | 否（复用） | 路径差异由 MCU 通过 `current_batch` 自行处理 |
| `NAV_TO_TEMP` | 否（复用） | 路径差异由 MCU 通过 `current_batch` 自行处理 |
| `ALIGN_RAW` | 否（复用） | `current_batch_order` 参数化颜色顺序 |
| `PICK_RAW` | 否（复用） | 抓取动作无差异 |
| `ALIGN_ROUGH` | 否（复用） | `picking_from_rough` 区分放料/取回阶段 |
| `PLACE_ROUGH` | 否（复用） | 无差异 |
| `ALIGN_TEMP` | 否（复用） | 无差异 |
| `PLACE_TEMP` | 否（复用） | `current_batch=2` 时由 `_PlaceTempState.on_execute` 决定下一状态 |
| `CHECK_LOAD` | 否（复用） | `current_zone + current_step` 路由 |
| `PICK_ROUGH` | 否（复用） | 第一批 ROUGH 取回已有此状态 |

**`update()` 调用时机**（在 `MissionCoordinator` 中）：

**MissionSM update**: 每次外部事件处理后立即调用 `mission_sm.run_to_completion()`，确保 `on_execute` 及时检查状态并完成级联转换。

```python
# 每个事件处理器末尾
self.mission_sm.on_arrived(zone_id)
self.mission_sm.run_to_completion()

self.mission_sm.on_action_done(action_id, result)
self.mission_sm.run_to_completion()
```

**VisualSM update**: 由 `coordinator.loop()` 内的 `drain_results` → `_process_vision_results` → `_handle_track_result` 触发。
与 MissionSM **运行在同一主线程**，而非相机处理线程。

```python
# coordinator.loop() 每 tick 执行
def loop(self):
    for all_results in self._vision_manager.drain_results():
        self._process_vision_results(all_results)  # ← 其中调 visual_sm.update()
    with self._sm_lock:
        while self._sm_queue:
            self._sm_queue.popleft()()
    self.mission_sm.run_to_completion()
```

---

## 4. VisualStateMachine

### 线程归属

VisualSM **运行在主线程**（与 MissionSM 同线程），而非相机处理线程。

```
视觉数据流:
  Camera 采集线程 → queue.Queue → VisionManager._process_loop (相机线程)
    → _pending_results.append(all_results) [deque]
    → coordinator.loop() → drain_results() [主线程]
      → _process_vision_results → _handle_track_result
        → visual_sm.update()        ← 主线程
        → _send(visual_servo_frame) ← 主线程
```

- `update()` 由主线程每 tick 调用一次（约 300Hz），每次处理一帧视觉结果
- `start()` / `stop()` 可能由 UART 接收线程（通过 EventBus → `_on_mcu_cmd`）触发，持有 `visual_sm._lock` (RLock)
- 不存在跨线程 `update()` 争用，因为 `update()` 只在主线程被执行

### 状态

| 状态 | 含义 |
|---|---|
| `IDLE` | 待机，无视觉任务运行 |
| `SEARCH` | 全局搜索目标 |
| `TRACKING` | 找到目标，输出伺服偏差 |
| `RECOVERY` | 跟踪丢失，扩大搜索（当前未启用） |
| `FAIL` | 视觉异常终止 |

### 事件与转换

```
IDLE → SEARCH: START
SEARCH → TRACKING: 由 on_execute 自动转换（consecutive_detected_frames >= 10）
TRACKING → SEARCH: 由 on_execute 自动转换（consecutive_lost_frames >= 5）
任意 → IDLE: STOP
FAIL → IDLE: RESET
```

---

## 5. Cargo 数据流

```
MissionCoordinator.__init__
  → ctx.cargo_set = CargoSet.create_standard()  # 6 个 CargoItem

QR 解析完成
  → ctx.batch_order = [Color.RED, Color.GREEN, Color.BLUE]
  → ctx.current_batch = 1
  → ctx.current_step = 0

进入 ALIGN_RAW
  → ctx.target_color = ctx.batch_order[ctx.current_step]
  → item = ctx.cargo_set.get_by_color(target_color, batch)[0]

PICK 成功
  → item.pick()           # available=False, zone=ON_ROBOT
  → ctx.cargo_count += 1

PLACE 成功
  → item.place(zone)      # zone=ROUGH/TEMP
  → ctx.cargo_count -= 1
  → ctx.current_step += 1

批次完成
  → ctx.current_step = 0
  → ctx.current_batch += 1
  → ctx.batch_order = 来自 QR 的第二批顺序

全部完成 / RESET
  → ctx.cargo_set.reset_all()
```

---

## 6. 错误与恢复

| 错误场景 | 处理方式 |
|---|---|
| 视觉连续丢失 5 帧 | TARGET_LOST → SEARCH，重置伺服数据 |
| 视觉超时（3s 无检测结果） | VISUAL_FAIL → ERROR |
| 颜色不匹配 | COLOR_MISMATCH 标志位上报，不切 ERROR |
| 心跳丢失 3 次（300ms） | 仅标记 is_linked=false，不切 ERROR |
| CMD_STOP_VISUAL | 关闭所有视觉任务，VisualSM → IDLE |

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v1.3** | 2026-07-18 | 新增 `CARGO_DISCOVERY(19)` 状态 + `ARRIVED_TEMP_BATCH2` 事件；第二批次 TEMP 区改为 CARGO_DISCOVERY（CargoDetector 扫描 3 色），不复用 RING_DISCOVERY |
| **v1.0** | 2026-07-04 | 初始化：确定区域自动控制设计，定义状态↔任务映射表 |
| **v1.1** | 2026-07-06 | 18 状态连续编号 0~17；删除 `NAV_TO_TEMP_SECOND` / `PLACE_TEMP_STACK`，仅保留 `NAV_TO_RAW_SECOND` 作为第二批独立状态；`_PlaceTempState.on_execute` 实现批次切换 + `cargo_count<0` 保护；mermaid 图重绘显示双批次流程；新增 `place_action_done` 机制说明 |
| **v1.2** | 2026-07-08 | 新增 `RING_DISCOVERY(18)` 状态；NAV_TO_ROUGH/TEMP → RING_DISCOVERY；ALIGN_ROUGH/TEMP 改为无视觉伺服（camera 被 cargo 遮挡，靠 mapping+惯导）；新增 `run_to_completion()` 级联；on_execute 简化；事件列表移除 `READY_TO_PLACE`、新增 `DISCOVERY_DONE` |
