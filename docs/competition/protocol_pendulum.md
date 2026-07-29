# 摆杆钢球位置 — 数据流协议

> 适用项目：Zulu-Walker（MaixCAM2 + MSPM0）
> 用途：MaixCAM2 检测钢球在摆杆凹槽中的位置，将归一化误差和物理坐标通过 UART 推送给 MSPM0
>
> 基于 Master-Slave 协议 v3.0，帧结构定义见 [`protocol.md`](protocol.md)
> 数据流类型定义见 [`protocol_content.md`](protocol_content.md) §3

---

## 1. 概述

MaixCAM2 运行 `steelball_640.mud`（YOLO11n 检测模型），摄像头垂直固定在摆杆上方。每帧推理后提取钢球 bbox 中心坐标，通过标定参数换算为摆杆上的厘米位置和归一化误差，通过 UART 推送给 MSPM0。MSPM0 使用这些数据做摆杆角度 PID 控制。

测量为一维（钢球沿摆杆凹槽左右滚动），Y 方向恒为 0。

```
 ┌──────────────────────────┐
 │          MaixCAM2         │  ← 垂直固定在摆杆正上方
 │          ● 镜头            │     距离摆杆 z cm
 │          │                 │
 ├──────────────────────────┤
 │   ← -12.5cm    0    +12.5cm → │
 │   ┌──┬──┬──┬──┬──┬──┬──┬──┐  │  摆杆刻度 (0.1cm/格)
 │   │  │  │  │○ │  │  │  │  │  │  ○ = 钢球
 │   └──┴──┴──┴──┴──┴──┴──┴──┘  │
 └──────────────────────────┘
          ball_cm = 0.00
```

### 1.1 坐标换算（MaixCAM2 侧）

标定参数 `pixels_per_cm` 在安装后实测获得：在摆杆两端（±12.5cm）放置标记，测量像素距离。

```python
frame_w = frame.shape[1]
half = max(frame_w, frame.shape[0]) / 2.0
cx = ball.x + ball.w / 2

pe_x = int(((cx - frame_w / 2.0) / half) * 5000.0)   # [-5000, 5000]
ball_cm = (cx - frame_w / 2.0) / pixels_per_cm         # 物理厘米
ball_cm_scaled = int(ball_cm * 100)                    # 0.01cm 分辨率
```

---

## 2. 订阅方式

### 2.1 开始订阅

MSPM0 发送 `CMD_REQUEST` 指定 `data_type = 0x07`：

```
MSPM0 → MaixCAM2:
  TYPE_CMD_REQUEST (0x20): data_type=0x07, min_interval_ms=0, reserved=0x00
```

MaixCAM2 回复 ACK 后开始推送：

```
MaixCAM2 → MSPM0:
  TYPE_CMD_ACK (0x21): data_type=0x07, max_freq_hz=60, payload_size=0
```

### 2.2 停止订阅

```
MSPM0 → MaixCAM2:
  TYPE_CMD_STOP (0x23): (empty payload)
```

### 2.3 数据流类型

| 常量 | 值 | 说明 |
|---|---|---|
| `DATA_PENDULUM_POSITION` | **0x07** | 摆杆钢球位置数据流 |

---

## 3. Payload 格式

`TYPE_DATA_STREAM (0x24)` 的外层 header（seq + data_type）见 Master-Slave 协议 v3.0，此处定义 `sub_payload` 部分：

```
Byte 0-1:   percent_error_x (s16 LE)  — 归一化位置误差 [-5000, +5000]
                                         0=中心, 负=左, 正=右
Byte 2-3:   ball_cm (s16 LE)           — 物理位置, 0.01cm 单位
                                         -1250 ~ +1250 对应 ±12.50cm
Byte 4:     flags (u8)                 — 标志位 (见 §3.1)
Byte 5:     reserved (u8)              — 预留, 恒为 0
```

总 sub_payload = **6 字节**（固定长度）。

`DATA_PAYLOAD_SIZES[0x07] = 8`（含 seq + data_type 的 2 字节 header）。

### 3.1 flags 位定义

```
bit 0: TARGET_FOUND    — 当前帧检测到钢球
bit 1-7: 预留          — 恒为 0
```

### 3.2 使用建议（MSPM0 侧）

| 场景 | 方法 |
|------|------|
| PID 输入 | 直接使用 `percent_error_x`（归一化，与画面分辨率解耦） |
| 判定居中 | `\|percent_error_x\| ≤ 200`（约 2% 画面偏移）→ 已居中 |
| 物理误差判定 | `ball_cm / 100` → 厘米值，赛题要求误差 ≤ ±1cm |
| 未检测到目标 | `flags.bit0 == 0` → 保持上次有效值或回退安全位置 |

---

## 4. 时序示例

```
MSPM0                        MaixCAM2
  |                              |
  |  摆杆控制阶段开始              |
  |                              |
  |── CMD_REQUEST(data_type=0x07) ─→|
  |                              | 验证 data_type=0x07
  |←── CMD_ACK(0x07, 60, 0) ────┤ 订阅成功，开始推送
  |                              |
  |←── DATA_STREAM(0x00, 0x07, [pe_x=0,    ball_cm=0,    flags=0x01, rsv=0]) ──┤ 钢球在中心
  |←── DATA_STREAM(0x01, 0x07, [pe_x=1500, ball_cm=375,  flags=0x01, rsv=0]) ──┤ 钢球在 +3.75cm
  |←── DATA_STREAM(0x02, 0x07, [pe_x=-2000,ball_cm=-500, flags=0x01, rsv=0]) ──┤ 钢球在 -5.00cm
  |←── DATA_STREAM(0x03, 0x07, [pe_x=0,    ball_cm=0,    flags=0x00, rsv=0]) ──┤ 未检测到钢球
  |                              |
  |  控制完成，停止               |
  |── CMD_STOP() ──────────────→|
  |                              | 停止推送
```

---

## 5. 映射参考（640px 画面, pixels_per_cm=25.6）

| 钢球实际位置 | ball_cm (0.01cm) | percent_error_x |
|---|---|---|
| -12.50 cm（左端） | -1250 | -5000 |
| -6.25 cm | -625 | -2500 |
| 0.00 cm（中心） | 0 | 0 |
| +6.25 cm | +625 | +2500 |
| +12.50 cm（右端） | +1250 | +5000 |

---

## 6. 代码映射

| 协议元素 | Python 常量/函数 | 位置 |
|---|---|---|
| 数据流类型 | `DATA_PENDULUM_POSITION = 0x07` | `protocol.py` |
| Payload 大小 | `DATA_PAYLOAD_SIZES[0x07] = 8` | `protocol.py` |
| 支持数据类型 | `SUPPORTED_DATA_TYPES` 自动包含 | `protocol.py` |
| Payload 构建 | `Ti2026Coordinator._build_pendulum_position_payload()` | `coordinator.py` |
| 订阅 dispatch | `_build_stream_payload()` 分支 `0x07` | `coordinator.py` |
| 配置参数 | `pendulum.pixels_per_cm` | `project_config.yaml` |

---

## 7. 视觉配置参数

MaixCAM2 侧需要以下配置才能正确计算钢球位置。

### 7.1 模型

| 参数 | 值 |
|------|-----|
| 模型文件 | `/root/models/steelball_640.mud` |
| 模型类型 | YOLO11n 检测 (`auto` → `yolo`) |
| 输入分辨率 | 640×640 |
| 目标类别 | class_id=0（steel ball） |

### 7.2 标定参数

`pixels_per_cm` 将像素偏移换算为摆杆上的物理厘米，为实现稳定的前提。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pixels_per_cm` | 25.6 | 每厘米对应像素数，640px / 25cm 摆杆，安装后实测标定 |
| `length_cm` | 25.0 | 摆杆总长度 |

### 7.3 标定方法

安装好摄像头后，在摆杆两端（±12.5cm 处）各放一个明显标记：

1. 读取两端标记在画面中的像素坐标 `x_left`, `x_right`
2. `pixels_per_cm = (x_right - x_left) / length_cm`

也可只在中心 0 点放已知偏移的物体（如钢球放在 +5cm 刻度），单点标定：

3. `pixels_per_cm = (cx - frame_w/2) / 5.0`

### 7.4 坐标系约定

```
图像坐标:          (0,0) ─────────── X+ ──→ (639,0)
                       │
                       │    ● (cx, cy)   钢球 bbox 中心
                       │
                       ↓ (0,639)
                      Y+

物理坐标:     左端(-12.5cm) ← 0(中心) → 右端(+12.5cm)
```

### 7.5 关键实现逻辑

`_build_pendulum_position_payload()` 的完整逻辑：

```
输入: AIInferenceProcessor 的 detections[]，每帧一次
  1. 遍历 detections，筛选 class_id == 0 (steel ball)
  2. 取 score 最高的 ball
  3. 若无 ball → percent_error_x=0, ball_cm=0, flags=0x00
  
  4. cx = ball.x + ball.w / 2
  5. frame_w = 640（相机固定分辨率）
  6. half = max(frame_w, frame_h) / 2.0
  7. pe_x = int(((cx - frame_w / 2.0) / half) * 5000.0)
  8. ball_cm = (cx - frame_w / 2.0) / pixels_per_cm
  9. ball_cm_scaled = int(ball_cm * 100)
  
输出: percent_error_x(2B) + ball_cm(2B) + flags(1B) + 0x00(1B)
```

### 7.6 配置文件参考

```yaml
# project_config.yaml 相关段落
ai:
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_640.mud"
      model_type: "auto"
  active: "yolo11n"

pendulum:
  pixels_per_cm: 25.6       # 安装后实测标定
  length_cm: 25.0           # 摆杆总长度
```

---

## 8. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v1.0** | 2026-07-29 | 初版定义 |
