# ball_val — 钢球速度扩充字段

> 目标读者：MCU 侧开发人员（STM32F4）
> 适用协议：视觉-MCU 通信 v2.1（Master-Slave v3.0），数据流类型 `DATA_PENDULUM_POSITION(0x07)`
> 生效日期：2026-07-30

## 1. 概述

在原摆杆数据集 `DATA_PENDULUM_POSITION(0x07)` 中新增 **`ball_val (s16 LE)`** 字段，
提供钢球的**实时运动速度**。MCU 可用此值实现速度前馈、阻尼控制或轨迹预测。

视觉端通过帧间位置差和精确时间戳计算速度，以 0.01 cm/s 为单位编码为 s16。

## 2. Payload 布局变更

### 旧格式（8 字节，仍被接受）

| 偏移 | 字段 | 类型 | 说明 |
|-----:|------|------|------|
| 0 | `seq` | u8 | 帧序号 |
| 1 | `data_type` | u8 | 固定 0x07 |
| 2-3 | `percent_error_x` | s16 LE | 归一化位置误差 [-5000, +5000] |
| 4-5 | `ball_cm_x100` | s16 LE | 物理坐标，0.01cm 单位 |
| 6 | `flags` | u8 | bit0=TARGET_FOUND |
| 7 | `reserved` | u8 | 恒为 0 |

### 新格式（10 字节，当前视觉端输出）

| 偏移 | 字段 | 类型 | 说明 |
|-----:|------|------|------|
| 0-7 | 同旧格式 | — | 位置数据不变 |
| 8-9 | **`ball_val`** | **s16 LE** | **钢球速度，0.01 cm/s** |

- 总 Payload = **10 字节**（含 seq + data_type 的 2 字节 header）
- 订阅 ACK 的 `payload_size` = **10**
- MCU 同时接受旧 8 字节和新 10 字节格式

## 3. ball_val 语义

| 项目 | 定义 |
|------|------|
| **物理意义** | 钢球沿摆杆凹槽方向的瞬时速度 |
| **单位** | 0.01 cm/s（与 `ball_cm_x100` 同分辨率） |
| **编码** | s16 LE, `int(round(vel_cm_s * 100))` |
| **范围** | [-32768, +32767]，即 ±327.67 cm/s（远大于实际钢球速度） |
| **符号** | 正=向右（+ball_cm 方向），负=向左（-ball_cm 方向） |
| **无目标时** | `ball_val = 0`，且 `ball_val_valid = 0` |
| **首帧检测** | `ball_val = 0`（无历史数据，无法计算速度） |
| **数据陈旧** | 相邻帧间 dt > 1.0s 时重置为 0，重新开始追踪 |

### 换算示例

| ball_val 值 | 实际速度 | 含义 |
|------------:|----------|------|
| 120 | +1.20 cm/s | 钢球向右滚动 1.2 cm/s |
| -45 | -0.45 cm/s | 钢球向左滚动 0.45 cm/s |
| 1500 | +15.00 cm/s | 钢球快速右滚 |
| 0 | 0 cm/s | 静止/首帧/无目标 |
| 0 + valid=0 | — | 旧版视觉端或兼容帧 |

## 4. MCU 接收端适配

当前 F4 固件（commit 487178）已支持 10 字节解析。业务代码中使用：

```c
Protocol_PendulumData_t pendulum;

if ((Protocol_GetPendulumData(&pendulum) != 0U) &&
    (Protocol_IsPendulumDataFresh(100U) != 0U) &&
    (pendulum.target_found != 0U) &&
    (pendulum.ball_val_valid != 0U))
{
    int16_t ball_val_raw = pendulum.ball_val;
    float ball_vel_cm_s = ball_val_raw * 0.01f;  // 转换为 cm/s

    // 使用 ball_vel_cm_s 做速度前馈等
}
```

## 5. 视觉端计算方式

视觉端在 `coordinator.py` 中计算速度：

1. 每帧 AI 推理后得到钢球的 `ball_cm_x100`（0.01cm 位置）
2. 记录帧的时间戳 `time.monotonic()`
3. 仅在新旧位置值不同时计算：`velocity = (ball_cm_x100_new - ball_cm_x100_old) / dt`
4. 位置未变化时复用缓存的速度值（避免 coordinator 超采样 ~500Hz 导致的噪声）
5. 结果四舍五入取整，裁剪到 `[-32768, +32767]`

### 精度分析

- 位置分辨率：0.01cm（1 个 ball_cm_x100 单位）
- 帧间时间：~16.7ms（@60fps）
- 最小可检测速度：1 / 0.0167 = ~60 单位 = 0.60 cm/s
- YOLO bbox 抖动（~1-3 像素）在高帧率下可能产生噪声，需 MCU 侧低通滤波

## 6. 兼容性

| 条件 | ball_val | ball_val_valid |
|------|----------|:---:|
| 10 字节帧（当前 MaixCAM2 视觉端） | 有效速度值 | 1 |
| 8 字节帧（旧版视觉端 / 降级模式） | 0 | 0 |
| 无钢球检测 | 0 | 1 |
| 首次检测到钢球 | 0 | 1 |
| 帧间间隔 > 1s | 0 | 1 |
| 钢球完全静止 | 0 | 1 |

> MCU 应同时检查 `ball_val_valid == 1` 和连续多帧有效后再启用速度控制，避免首次检测时的无效速度。

## 7. 测试向量

```
AA 55 0D 24 2A 07 DC 05 77 01 01 00 88 FF B0 59
```

| 字节 | 值 | 含义 |
|---:|---:|---|
| 0-1 | `AA 55` | SOF |
| 2 | `0D` | Length = 13 |
| 3 | `24` | TYPE_DATA_STREAM |
| 4 | `2A` | seq = 42 |
| 5 | `07` | DATA_PENDULUM_POSITION |
| 6-7 | `DC 05` | percent_error_x = 1500 |
| 8-9 | `77 01` | ball_cm_x100 = 375 (+3.75cm) |
| 10 | `01` | TARGET_FOUND |
| 11 | `00` | reserved |
| 12-13 | `88 FF` | **ball_val = -120 (-1.20 cm/s, 向左滚动)** |
| 14-15 | `B0 59` | CRC16-CCITT |

## 8. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-30 | 初始版本，同步 MaixCAM2 视觉端 ball_val 实现 |
