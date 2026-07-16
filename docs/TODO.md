# TODO

## Processor 层

### ~~pick_watch — 转盘守株待兔（PICK_RAW）~~ 已解决，实时追踪
- **当前状态**: 未开始；团队讨论中
- **场景**: 原料区转盘 6-10s/圈，物料 120° 间隔，平移式机械臂无法横向避让
- **需求**: 固定 ROI 监测夹爪下方，识别到达物料颜色，匹配 current_target_color() 则触发抓取，不匹配则等待转盘送来下一个
- **待定**: 是否打破 QR 顺序抓取 + 事后重排？是否改协议加打断/收回信号？

---

## 已有隐患（审查发现）

### ~~CHECK_LOAD `cargo_confirmed` 死锁~~ 已缓解，直接跳过检测
- **严重度**: 高
- **现象**: `VisualFlags.CARGO_CONFIRMED (0x20)` 从未被 vision pipeline 或 MCU 帧设置。`CHECK_LOAD.on_enter` 重置 `cargo_confirmed = False`，当前唯二设置路径 (`PICK_RAW.on_exit` / `PICK_ROUGH.on_exit`) 在进入 CHECK_LOAD 前已被覆盖。结果：CHECK_LOAD 3 秒后必定超时进 ERROR。
- **可能原因**: 本应由 MCU 通过 UART 帧设置此 flag，对应帧类型尚未实现；或是 vision 处理器应返回此 flag 但未实现。

### ~~`build_visual_servo_data_frame` 期望 int 但收到 float~~ 已修复
- **状态**: ✅ 已修复（2026-07-09）
- **修复方式**: 三个处理器（cargo / ring_discovery / circle_target）统一输出 `int[-5000, 5000]`，`_handle_track_result` 移除 `int(x*5000)` 转换，直接传递。

### ~~`cv2.ocl.setUseOpenCL(True)` 分散在多处构造函数~~ 已修复
- **严重度**: 低
- **现象**: 全局副作用调用分散在 `TrackCargoProcessor.__init__`（已删除）和 `CircleTargetProcessor.__init__` 中。
- **修复方向**: 集中到 `CameraManager.__init__` 或专用初始化函数，只调用一次。

### ~~多个 VisualFlags 位定义了但从未设置~~ 已修复
- **严重度**: 中
- **涉及标志位**: `QR_OK(0x10)`、`VISUAL_FAIL(0x08)`、`COLOR_MISMATCH(0x40)`
- **现象**: 三个位在 `protocol.py:VisualFlags` 中定义，`update_visual_flags()` 中解析，但没有任何代码在 flags 字节中设置它们。
- **已知不影响功能**:
  - `QR_OK` — MCU 实际使用 `TYPE_QR_RESULT` 专用帧获取二维码，不检查此位
  - `VISUAL_FAIL` / `COLOR_MISMATCH` — C 侧没有对应的 `VISION_FLAG_IS_*` 宏调用
- **建议**: 确认无使用计划后删除这些位的定义和解析逻辑，减少困惑

---

## 暂时解决不了

### 内核 HZ 配置检查（主循环 1ms tick 依赖）
- **当前平台**: ~300Hz（实测），`MAIN_LOOP_DELAY = 0.00333`，每 tick 最快 3.33ms
- **目标**: 升级内核配置到 `HZ=1000` + `CONFIG_HIGH_RES_TIMERS=y` 后可将 `MAIN_LOOP_DELAY` 降到 `0.001`
- **现状**: 更改系统内核tick需要换内核，暂时不做
