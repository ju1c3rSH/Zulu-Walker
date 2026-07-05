# 工创赛项目进度（2026-07-04）

## 已完成

### 代码清理
- [x] 删除 `utils/state.py`（328 行死代码，已被 `utils/state_machine/` 替代）
- [x] 删除 `_archived_tests/`（8 个旧测试脚本）
- [x] 删除 `.VSCodeCounter/`（自动生成）
- [x] 删除 `report/`（旧比赛文档）
- [x] 清理 `zw_uart_module/__init__.py` 未用便捷函数
- [x] 清理 `zw_uart_module/protocol.py` 未用 `parse_error_payload`

### 模型层
- [x] 新建 `models/color.py` — `Color(IntEnum)`: RED=1, GREEN=2, BLUE=3
- [x] 新建 `models/cargo.py` — `CargoItem`, `CargoSet`, `CargoZone`, Slot 映射常量

### 协议层（v1.1）
- [x] `FLAG_*` 常量 → `class VisualFlags`（统一到 `protocol.py`）
- [x] `COLOR_RESULT` payload: 移除 `slot_idx`
- [x] `CMD_START_COLOR_DETECT` 参数: 移除 `slot_idx`
- [x] 删除 `CMD_SET_EXPOSURE`（相机曝光 OP 自主管理）
- [x] 删除 legacy `orange_send` 协议栈（`build_orange_send_frame`、`ORANGE_STATE_*`）
- [x] 删除 `send_orange_frame` 路径（`camera_manager.py` → `uart_driver.py` → `protocol.py` 全链路）
- [x] 时序图 B 修复：新增 `CMD_START_COLOR_DETECT` 步骤，`STATUS_FROM_MCU` → `HEARTBEAT`
- [x] `HEARTBEAT` 说明：MCU→OP 时 `visual_state` 恒为 0
- [x] 新增机器人 3 槽位固定颜色约束

### 架构复位
- [x] 新建 `context/event_bus.py` — 类型化 pub/sub（同步 + RLock + ring buffer）
- [x] `mission_state_machine.py` 从 `utils/state_machine/` 搬到 `context/`
- [x] `context/` 内部 import 修复（`from .base` → `from utils.state_machine.base`）
- [x] `Color` 定义统一来源 → `models/color.py`
- [x] `VisualFlags` 定义统一来源 → `protocol.py`
- [x] `utils/state_machine/__init__.py` 移除 Mission* 导出
- [x] `AGENTS.md` 更新架构文档
- [x] `vision_mcu_sync.md` 更新文件组织

### Phase 1：事件类型
- [x] 新建 `context/events.py`（15 个事件 dataclass）

### Phase 2：UART 收帧扩展
- [x] `uart_driver.py:_handle_frame()` 调度 0x10-0x18（含 parse 调用 + EventBus publish）
- [x] `STM32UartInterface` 新增 `set_event_bus()`、`send_raw()`
- [x] `zw_uart_module/__init__.py:init()` 接受 `event_bus` 参数

### Phase 3：CameraManager 纯粹化
- [x] 删除 `CameraManager` 中的 `VisualStateMachine` 实例
- [x] 删除 `_handle_detection()`、`_send_error_frame()`、`_setup_state_callbacks()`、`set_target_color()`
- [x] 删除 `from modules.zw_uart_module import send_orange_frame` 和 `ORANGE_STATE_*`
- [x] 新增 `_event_bus` + `set_event_bus()` + publish `FrameResult`

### Phase 4：MissionCoordinator 接线
- [x] `context/mission_context.py` 全面重写：
  - 8 种事件订阅（McuCmdReceived, ArrivedEvent, ActionDoneEvent, ...）
  - 6 种 MCU 命令处理 → 后续精简为 2 种（CMD_START_QR, CMD_STOP_VISUAL）
  - `VisualStateMachine` 接管、视觉结果→状态机转换、心跳、QR 门控
- [x] `main.py` 启动流程重构
- [x] `AUTO_START_MODULES` 中移除 `zw_uart_module`
- [x] `zw_opencv_module/__init__.py:init()` 支持注入 `event_bus`
- [x] `zw_uart_module/__init__.py` 恢复 `get_interface()`

### 协议层（v1.2）
- [x] 删除 `CMD_START_COLOR_DETECT(0x02)` / `CMD_TRACK_TARGET(0x03)` / `CMD_TRACK_TOP(0x05)` — 视觉任务改为区域自动控制
- [x] 删除 `CMD_TRACK_RING(0x04)` — 同上
- [x] 最终子命令表仅保留 `CMD_START_QR(0x01)` + `CMD_STOP_VISUAL(0x06)`
- [x] `TYPE_COLOR_RESULT` 注释修正：`slot_idx + color_id + confidence` → `color_id(1B) + confidence(1B)`

### 架构文档
- [x] 新增 `docs/architecture/state_machine.md` — 统一状态机设计（含区域自动控制约定）
- [x] 新增 `docs/competition/protocol.md` — 供电控组对接的协议规范
- [x] `docs/competition/vision_mcu_sync.md` 顶部加拆分指引，标注以新文档为准

### Processor 基础
- [x] 新增 `ColorTrackable Protocol` — 使用 `@runtime_checkable` + `isinstance` 检测
- [x] `CircleTargetProcessor.set_target_color` 签名 `Optional[str]` → `Optional[Color]`
- [x] 删除遗留 `_apply_detect_params` 硬编码方法（`camera_manager.py`）

### Coordinator / Context 重构
- [x] `MissionContext` 字段重命名：`first_batch/second_batch/current_index/target_color:int` → `batch_order/current_step/target_color:Color`
- [x] `cargo_set` 全局唯一，挂在 `MissionContext` 下
- [x] 摄像头路由：`source == "qr_cam"` → `cam_id.endswith("_qr"/"_cargo")`
- [x] `hasattr` → `isinstance(processor, ColorTrackable)`

---

## 待实现（按优先级排列）

### P0 — 核心链路（必须先完成）
- [ ] `camera_manager.py:_create_processor()` 注册三种 Processor 类型：
  - `QRProcessor` / `TrackCargoProcessor` / `RingTrackProcessor`
  - 需要先创建这 3 个 stub Processor 文件（`qr_processor.py`, `track_cargo_processor.py`, `ring_track_processor.py`）
- [ ] `MissionCoordinator` 监听 `MissionSM` 状态变化，自动激活对应视觉任务：
  - `READ_QR` → 自动 `_activate_task("qr_detect")`
  - `ALIGN_RAW` → 自动 `_activate_task("track_cargo", batch_order[step])`
  - `ALIGN_ROUGH` / `ALIGN_TEMP` → 自动 `_activate_task("ring_track", batch_order[step])`
  - 离开上述状态 → 自动 `_deactivate_all_visual()`

### P1 — 功能补全
- [x] `MissionSM` 参数化改造（3 循环 + 2 批次）
  - `_CheckLoadState.on_execute` 根据 zone+step 路由（RAW→RAW 循环 / ROUGH→ROUGH 循环 / 去下一站）
  - `_PlaceRoughState.on_execute` 循环放料 → 切换 `picking_from_rough` 进入取料阶段
  - `_PlaceTempState.on_execute` 循环放料 → `cargo_count==0` 回家 ✅ 修死锁
- [x] 新增 `PICK_ROUGH` 状态（MissionState=10）+ `_PickRoughState` 类 + 事件转换
- [x] 混合事件模型：视觉决策改用 `on_execute`（6 个视觉事件转换删除），MCU 事件保留事件驱动
- [x] `place_action_done` 标志解决 PLACE 状态不等 ACTION_DONE 就跳走的问题
- [x] `mission_context.py` 5 处事件处理器后加 `mission_sm.update()`
- [x] `_handle_track_frame` 中 ALIGN_ROUGH 区分 `picking_from_rough` → 正确设置 READY_TO_PICK/PLACE
- [x] 文档同步：`protocol.md` action_id=4 + 3 循环说明；`state_machine.md` PICK_ROUGH + 混合模型

### P2 — 架构纯净化
- [ ] Coordinator 桥接代码 EventBus 化：`_activate_task` / `_deactivate_all_visual` 改为 `EnableTask` / `DisableTask` 事件
- [ ] 移除 Phase 4 临时桥接（Coordinator 直接持有 `CameraManager` 引用）
- [ ] `_on_vision_results` 派发逻辑差异化：QR 结果、track 结果、ring 结果各自走不同 handler

### P3 — 未来扩展
- [ ] 第二批状态（13-22）清理或实现（如果参数化完成后这些独立状态不再需要则删除）
- [ ] `CargoSet` / `CargoItem` 状态联动（pick/place 时自动更新 zone/available）
- [ ] `ColorResult` 事件 — 确认是否仍需保留（颜色现从 `batch_order` 获取）

### AGENTS.md 更新
- [x] cargo_detector 架构说明
- [ ] 本轮架构变更同步（任务系统、Protocol、文档结构）
