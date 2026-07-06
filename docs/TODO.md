# TODO

## Processor 层

### RingTrackProcessor — 色环对准（ALIGN_ROUGH / ALIGN_TEMP）
- **当前状态**: stub，`process()` 返回 `success=False`
- **依赖**: 比赛场地粗加工区/暂存区顶面有色环标记（`GongChuang2026.md:55`），用于测量物料摆放准确程度
- **待定**: 色环检测方案（复用 circle_target 检测器？新建 ring 专用方法？）

### pick_watch — 转盘守株待兔（PICK_RAW）
- **当前状态**: 未开始；团队讨论中
- **场景**: 原料区转盘 6-10s/圈，物料 120° 间隔，平移式机械臂无法横向避让
- **需求**: 固定 ROI 监测夹爪下方，识别到达物料颜色，匹配 current_target_color() 则触发抓取，不匹配则等待转盘送来下一个
- **待定**: 是否打破 QR 顺序抓取 + 事后重排？是否改协议加打断/收回信号？

---

## 已有隐患（审查发现）

### CHECK_LOAD `cargo_confirmed` 死锁
- **严重度**: 高
- **现象**: `VisualFlags.CARGO_CONFIRMED (0x20)` 从未被 vision pipeline 或 MCU 帧设置。`CHECK_LOAD.on_enter` 重置 `cargo_confirmed = False`，当前唯二设置路径 (`PICK_RAW.on_exit` / `PICK_ROUGH.on_exit`) 在进入 CHECK_LOAD 前已被覆盖。结果：CHECK_LOAD 3 秒后必定超时进 ERROR。
- **可能原因**: 本应由 MCU 通过 UART 帧设置此 flag，对应帧类型尚未实现；或是 vision 处理器应返回此 flag 但未实现。

### `build_visual_servo_data_frame` 期望 int 但收到 float
- **严重度**: 中
- **现象**: `TrackCargoProcessor.process()` 返回 `float` 类型 `percent_error_x/y`（[-1.0, 1.0]），但 `build_visual_servo_data_frame(error_x: int, error_y: int)` 调用 `.to_bytes()`。float 类型无此方法，会触发 `AttributeError`。
- **修复方向**: 在 `_handle_track_result` 中转换为定点整数后传入，或修改 `build_visual_servo_data_frame` 接受 float 并用 `struct.pack` 处理。

### `cv2.ocl.setUseOpenCL(True)` 分散在多处构造函数
- **严重度**: 低
- **现象**: 全局副作用调用分散在 `TrackCargoProcessor.__init__`（已删除）和 `CircleTargetProcessor.__init__` 中。
- **修复方向**: 集中到 `CameraManager.__init__` 或专用初始化函数，只调用一次。
