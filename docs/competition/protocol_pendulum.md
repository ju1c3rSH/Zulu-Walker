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
  TYPE_CMD_ACK (0x21): data_type=0x07, max_freq_hz=60, payload_size=10
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
                                         0=中心, 负=左, 正=右（经轨道标定投影）
Byte 2-3:   ball_cm (s16 LE)           — 物理位置, 0.01cm 单位
                                         -1250 ~ +1250 对应 ±12.50cm
Byte 4:     flags (u8)                 — 当前实现恒为 0x01（见 §3.1）
Byte 5:     reserved (u8)              — 预留, 恒为 0
Byte 6-7:   ball_vel (s16 LE)          — 钢球横向速度（α-β 滤波输出）,
                                         0.01cm/s 单位, 负=向左, 正=向右
```

总 sub_payload = **8 字节**（固定长度）。

`DATA_PAYLOAD_SIZES[0x07] = 10`（含 seq + data_type 的 2 字节 header；
CMD_ACK.payload_size 返回同值, MSPM0 直接据此分配缓冲）。

**发送时机（对齐 coordinator.py 实现）**：连续 `_BALL_ARM_FRAMES=2` 帧有效检测后开始
发帧（arm）；发帧期间短暂丢失不中断——α-β 滤波以预测值续发（coast），flags 仍为
0x01；连续 `_BALL_DROP_FRAMES=3` 次无效判定后 disarm 停止发帧，直至重新满足 arm 条件。
即：**未锁定时不发帧，而不是发 flags=0x00 的帧**。

### 3.1 flags 位定义

```
bit 0: STREAM_VALID — 当前实现恒为 0x01：仅在数据流有效（已 arm）时才发帧,
        未锁定时整帧不发。接收侧不应期待 flags=0x00 的帧。
bit 1-7: 预留          — 恒为 0
```

### 3.2 使用建议（MSPM0 侧）

| 场景 | 方法 |
|------|------|
| PID 输入 | 直接使用 `percent_error_x`（归一化，与画面分辨率解耦） |
| 判定居中 | `\|percent_error_x\| ≤ 200`（约 2% 画面偏移）→ 已居中 |
| 物理误差判定 | `ball_cm / 100` → 厘米值，赛题要求误差 ≤ ±1cm |
| 速度前馈 | `ball_vel / 100` → cm/s，可用于微分项或预测补偿 |
| 目标丢失 | **按帧超时判定**：>100ms（约 6 帧 @60fps）未收到新帧即视为丢失，
  保持上次有效值或回退安全位置。不要等待 flags=0x00 的帧——当前实现不会发出。 |

---

## 4. 时序示例

```
MSPM0                        MaixCAM2
  |                              |
  |  摆杆控制阶段开始              |
  |                              |
  |── CMD_REQUEST(data_type=0x07) ─→|
  |                              | 验证 data_type=0x07
  |←── CMD_ACK(0x07, 60, 10) ───┤ 订阅成功，开始推送
  |                              |
  |←── DATA_STREAM(seq=00, [pe_x=0,    ball_cm=0,   flg=01, rsv=0, vel=0])   ──┤ 钢球在中心
  |←── DATA_STREAM(seq=01, [pe_x=1500, ball_cm=375, flg=01, rsv=0, vel=-120]) ──┤ +3.75cm, 向左移动
  |←── DATA_STREAM(seq=02, [pe_x=-2000,ball_cm=-500,flg=01, rsv=0, vel=80])  ──┤ -5.00cm
  |                              |  （短暂丢失: α-β 预测值续发, flags 仍 0x01;
  |                              |    连续无效达阈值后停止发帧直至重新锁定）
  |                              |
  |  控制完成，停止               |
  |── CMD_STOP() ──────────────→|
  |                              | 停止推送
```

---

## 5. 映射参考（1280px 画面, pixels_per_cm=51.0）

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
| Payload 大小 | `DATA_PAYLOAD_SIZES[0x07] = 10` | `protocol.py` |
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
| 模型文件 | `/root/models/steelball_1280x352.mud` |
| 模型类型 | YOLO11n 检测 (`auto` → `yolo`) |
| 输入分辨率 | 1280×352 @60fps（横向长条 ROI:省算力 + 保证横向测距精度） |
| 目标类别 | class_id=0（steel ball） |

### 7.2 标定参数

`pixels_per_cm` 将像素偏移换算为摆杆上的物理厘米，为实现稳定的前提。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pixels_per_cm` | 51.0 | 每厘米对应像素数，1280px 宽画面 / 25cm 摆杆，安装后实测标定（50→51 修正） |
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
  3. 门控（对齐源码）:
     - 有效检测 → hit_count++；连续 ≥2 帧(_BALL_ARM_FRAMES)后 arm
     - 无有效检测 → miss_count++；armed 时以 α-β 滤波预测值续发(coast)；
       连续 ≥3 次(_BALL_DROP_FRAMES)后 disarm → 整帧不发(返回 None)
     - 未 arm 过 → 整帧不发

  4. cx = ball.x + ball.w / 2
  5. frame_w = 1280（相机固定分辨率）
  6. half = max(frame_w, frame_h) / 2.0
  7. 距离投影: 已轨道标定时 dist_px = calib.project(cx, cy);
     未标定回退 dist_px = cx_f - frame_w/2
  8. pe_x   = int((dist_px / half) * 5000.0)，clamp ±32767
  9. ball_cm_scaled = int(dist_px / pixels_per_cm * 100)
 10. ball_val = int(vx / pixels_per_cm * 100)   ← α-β 速度估计, clamp ±32767

输出: pe_x(2B s16LE) + ball_cm×100(2B s16LE) + flags=0x01 + rsv(1B)
      + ball_vel×100(2B s16LE)    共 8 字节 sub_payload
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
| **v1.1** | 2026-08-22 | **对齐源码**:sub_payload 6B→8B(新增 Byte6-7 `ball_vel`,α-β 速度输出,0.01cm/s);`DATA_PAYLOAD_SIZES[0x07]` 8→10,CMD_ACK 示例 payload_size 0→10;flags 语义改为恒 0x01+按帧超时判丢失(旧"flags=0x00 帧"语义作废);§7 模型/分辨率/标定参数更新至当前配置(1280×352, ppc=51.0) |
