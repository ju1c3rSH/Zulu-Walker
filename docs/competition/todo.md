# 工创赛项目进度

## 已完成

### 代码清理
- [x] 删除 `utils/state.py`（328 行死代码，已被 `utils/state_machine/` 替代）
- [x] 删除 `_archived_tests/`（8 个旧测试脚本）
- [x] 删除 `.VSCodeCounter/`（自动生成）
- [x] 删除 `report/`（旧比赛文档）
- [x] 清理 `zw_uart_module/__init__.py` 未用便捷函数
- [x] 清理 `zw_uart_module/protocol.py` 未用 `parse_error_payload`
- [x] 移除 `ColorResult` 事件类、协议常量 `TYPE_COLOR_RESULT`、`build_color_result_frame` / `parse_color_result_payload`

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

### 协议层（v1.2）
- [x] 删除 `CMD_START_COLOR_DETECT(0x02)` / `CMD_TRACK_TARGET(0x03)` / `CMD_TRACK_TOP(0x05)` — 视觉任务改为区域自动控制
- [x] 删除 `CMD_TRACK_RING(0x04)` — 同上
- [x] 最终子命令表仅保留 `CMD_START_QR(0x01)` + `CMD_STOP_VISUAL(0x06)`
- [x] `TYPE_COLOR_RESULT` 注释修正：`slot_idx + color_id + confidence` → `color_id(1B) + confidence(1B)`

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
- [x] 移除 `ColorResult` dataclass（无人发布/订阅，以 batch_order 替代）

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

### 状态机参数化改造（3 循环 + 2 批次）
- [x] `_CheckLoadState.on_execute` 根据 zone+step 路由（RAW→RAW 循环 / ROUGH→ROUGH 循环 / 去下一站）
- [x] `_PlaceRoughState.on_execute` 循环放料 → 切换 `picking_from_rough` 进入取料阶段
- [x] `_PlaceTempState.on_execute` 循环放料 → batch 1 完成后切换 batch 2 → `NAV_TO_RAW_SECOND`
- [x] 新增 `PICK_ROUGH` 状态（MissionState=10）+ `_PickRoughState` 类 + 事件转换
- [x] 混合事件模型：视觉决策改用 `on_execute`（6 个视觉事件转换删除），MCU 事件保留事件驱动
- [x] `place_action_done` 标志解决 PLACE 状态不等 ACTION_DONE 就跳走的问题
- [x] `mission_context.py` 5 处事件处理器后加 `mission_sm.update()`
- [x] `_handle_track_frame` 中 ALIGN_ROUGH 区分 `picking_from_rough` → 正确设置 READY_TO_PICK/PLACE
- [x] 文档同步：`protocol.md` action_id=4 + 3 循环说明；`state_machine.md` PICK_ROUGH + 混合模型
- [x] `MissionState` 重编号（PICK_ROUGH=10，精简为 18 个，删除 8 个死 `_SECOND` 常量）
- [x] 第二批参数化：`NAV_TO_RAW_SECOND` 注册并复用 `_NavToRawState`，状态重用 batch 1 逻辑
- [x] `advance_target()` 返回 bool（True = batch done，zone 转场）

### 视觉任务自动切换
- [x] `StateActionBridge` 桥接层实现（`utils/state_machine/bridge.py`）
- [x] `_wire_state_actions()` 注册状态 enter 回调：
  - `READ_QR` → `qr_detect`
  - `ALIGN_RAW` → `track_cargo`
  - 导航/等待/结束状态 → `_deactivate_all_visual`
  - `ALIGN_ROUGH` / `ALIGN_TEMP` → `_deactivate_all_visual`（v1.2：camera 被 cargo 遮挡，无视觉伺服）

### 配色不匹配防御
- [x] `_AlignRawState`：`color_mismatch` → ERROR
- [x] `_AlignRoughState`：`color_mismatch` → ERROR

### CargoSet 运行时接入
- [x] `_CheckLoadState.on_execute`：pick 确认后按 `color + batch + !is_on_robot` 匹配 CargoItem，调 `pick()`
- [x] `on_action_done` PLACE_ROUGH/TEMP：从 `cargo_pick_stack` popleft 出最先 pick 的 item，调 `place()`
- [x] `cargo_pick_stack`（FIFO）保证非标准色序下 place 匹配正确
- [x] 无匹配时打 WARNING 日志，不阻塞状态机

### 代码清理 & 注释
- [x] `parse_qr` docstring 更新（移除 `(Stub — full impl TBD)`）
- [x] `is_batch_complete()` 重定义为 `cargo_count == 0 and current_step == 0`，在 `_PlaceTempState` 转场前做断言检查
- [x] `_PlaceRoughState` 改用 `current_target_color()` 替代硬编码 `batch_order[0]`
- [x] `on_action_done` 返回值语义注释
- [x] `_AlignRoughState.on_enter` 新增 `ready_to_pick = False` 重置

---

## 待实现（按优先级排列）

### P0 — 核心链路（必须先完成）
- [x] `camera_manager.py:_create_processor()` 注册三种 Processor 类型（stub 文件已有定义但未测试）：
  - `QRCodeProcessor` / `TrackCargoProcessor` / `RingDiscoveryProcessor`
- [x] `TrackCargoProcessor.process()` 完整实现
- [x] `RingDiscoveryProcessor.process()` 完整实现（委托 RingDetector + FastRingMethod）
- [x] `RingTrackProcessor.process()` 使用 RingDetector 重写（已废弃——v1.2 架构 ALIGN_ROUGH/TEMP 无视觉伺服，processor 已删除）
- [x] `CircleTargetProcessor` 对齐新 `Color` 枚举和 `ColorTrackable` Protocol
- [x] 色环发现协议层：`CMD_START_RING_DISCOVERY(0x07)`、`CMD_DISCOVERY_DONE(0x08)`、`TYPE_COLOR_RESULT(0x13)` 恢复、`RING_CENTERED=0x80`
- [x] 状态机：`RING_DISCOVERY(18)` 状态 + `_RingDiscoveryState` + `run_to_completion()` 级联
- [x] 检测器架构：`RingDetector` + `RingDetectMethod` + `FastRingMethod` + `EdgeDrawingRingMethod`（参考 CargoDetector）
- [x] 共享卡尔曼：`detectors/_shared/kalman_utils.py`

### P1 — 功能补全
- [ ] 所有状态加超时保护（ALIGN_*/PLACE_*/NAV_* 等 — 当前仅 `_CheckLoadState` 有 3s 超时）
- [ ] `_PlaceRoughState` 加 `cargo_count < 0` 保护（`_PlaceTempState` 已有）

### P2 — 架构纯净化
- [ ] Coordinator 桥接代码 EventBus 化：`_activate_task` / `_deactivate_all_visual` 改为 `EnableTask` / `DisableTask` 事件
- [ ] 移除 Phase 4 临时桥接（Coordinator 直接持有 `CameraManager` 引用）
- [ ] `_on_vision_results` 派发逻辑差异化：QR 结果、track 结果、ring 结果各自走不同 handler
- [ ] `ctx.target_color` 字段清理：只写不读，决定保留还是删除

### P3 — 未来扩展
- [ ] 视觉处理器实现：QR 解码、颜色圆目标检测、环形目标检测
- [ ] `CargoSet.get_by_color()` 返回 `List` → 可加 `get_one_by_color()` 简化调用方
- [ ] `_CheckLoadState` 超时后调 `cargo_set` 回滚/回查（目前超时直接 ERROR）

### AGENTS.md 更新
- [ ] 本轮架构变更同步（CargoSet 运行时、ColorResult 移除、状态机参数化）
