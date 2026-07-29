# 钢板分割 — 数据流协议

> **⚠ 已废弃 — 仅供参考，不再使用**  
> 钢板检测已从项目需求中移除，本文档保留作为协议格式参考。
>
> 适用项目：Zulu-Walker（MaixCAM2 + MSPM0）
> 用途：通过 YOLO11n-seg 模型检测钢板（Metal-sheet），将掩码中心坐标和面积实时推送给 MSPM0
>
> 基于 Master-Slave 协议 v3.0，帧结构定义见 [`protocol.md`](protocol.md)
> 内容定义见 [`protocol_content.md`](protocol_content.md)

---

## 1. 概述

MaixCAM2 运行 `steelsheet_npu_only.mud`（YOLO11n-seg 模型），每帧推理后提取钢板掩码的质心坐标和面积，通过 UART 推送给 MSPM0。MSPM0 使用这些数据做导航定位（靠近钢板中心）。

数据推送采用 Master-Slave 订阅模式：MSPM0 发送 `CMD_REQUEST` 订阅，MaixCAM2 以 `DATA_STREAM` 帧持续推送。

## 2. 订阅方式

### 2.1 开始订阅

MSPM0 发送 `CMD_REQUEST` 指定 `data_type = 0x06`：

```
MSPM0 → MaixCAM2:
  TYPE_CMD_REQUEST (0x20): data_type=0x06, min_interval_ms=0, reserved=0x00
```

MaixCAM2 回复 ACK 后开始推送：

```
MaixCAM2 → MSPM0:
  TYPE_CMD_ACK (0x21): data_type=0x06, max_freq_hz=60, payload_size=0
```

### 2.2 停止订阅

```
MSPM0 → MaixCAM2:
  TYPE_CMD_STOP (0x23): (empty payload)
```

### 2.3 数据流类型

| 常量 | 值 | 说明 |
|---|---|---|
| `DATA_SEGMENTATION_MASK` | **0x06** | 钢板分割掩码数据流 |

## 3. Payload 格式（DATA_STREAM 帧）

`TYPE_DATA_STREAM (0x24)` 的外层 header（seq + data_type）见 Master-Slave 协议，此处定义 `sub_payload` 部分：

```
Byte 0:          count (u8, 0~4)
Bytes 1-7:       mask[0]:
  Byte 1:          class_id (u8)      — 钢板固定为 0
  Byte 2-3:        center_x (u16 LE)  — 掩码质心 X 像素坐标 (0~639)
  Byte 4-5:        center_y (u16 LE)  — 掩码质心 Y 像素坐标 (0~639)
  Byte 6-7:        area_px (u16 LE)   — 掩码面积 (像素数, 0~65535)

Bytes 8-14:      mask[1]: 同上
Bytes 15-21:     mask[2]: 同上
Bytes 22-28:     mask[3]: 同上
```

**约束：**
- 最多 4 个掩码（取面积最大的 4 个）
- `count` 为 0 时无后续数据（sub_payload 仅 1 字节）
- 最大 sub_payload = 1 + 4×7 = 29 字节，远在 252 字节限制内
- `area_px` 如超过 65535 截断为 65535

## 4. 坐标含义

```
 ┌──────────────────────┐
 │                      │  ↑ 图像坐标系 (640×640)
 │         ●            │  │ Y+
 │      (cx, cy)        │  │
 │         掩码区域       │  │
 │     ██████████        │  │
 │     ██████████        │  │
 │     ██████████        │  │
 │                      │  │
 └──────────────────────┘  └──→ X+

  center_x/y = 掩码像素在检测框内的局部质心 + 检测框左上角 (obj.x, obj.y)
              即：实际图像坐标中的掩码质心位置

  area_px = 掩码 True 像素计数（np.nonzero 结果长度）
            可用于判断钢板远近：面积大 → 近，面积小 → 远
```

## 5. 时序示例

```
MSPM0                        MaixCAM2
  |                              |
  |  钢板检测阶段开始              |
  |                              |
  |── CMD_REQUEST(data_type=0x06) ─→|
  |                              | 验证 data_type
  |←── CMD_ACK(0x06, 60, 0) ────┤ 订阅成功，开始推送
  |                              |
  |←── DATA_STREAM(seq=0x00, 0x06, [count=1, class=0, cx=320,cy=240,area=8500]) ──┤
  |←── DATA_STREAM(seq=0x01, 0x06, [count=1, class=0, cx=318,cy=242,area=8600]) ──┤
  |←── DATA_STREAM(seq=0x02, 0x06, [count=0]) ──┤  无检测到钢板
  |←── DATA_STREAM(seq=0x03, 0x06, [count=2, ...]) ──┤  检测到 2 块钢板
  |                              |
  |  导航完成，停止               |
  |── CMD_STOP() ──────────────→|
  |                              | 停止推送
```

## 6. MSPM0 侧使用建议

1. **获取钢板位置**：解析 DATA_STREAM(0x06)，读 `center_x`, `center_y`
2. **计算误差**：`error_x = center_x - 320`（画面中心），用于 PID 控制
3. **判断距离**：`area_px` 越大 → 钢板越近 → 接近目标速度降低
4. **目标锁定**：当 `area_px` 超过阈值（如 5000px）且 `error_x` 在容差范围内 → 已对准
5. **多钢板处理**：取面积最大的钢板（第一个 mask）作为当前目标

## 7. 代码映射

| 协议元素 | Python 常量/函数 | 位置 |
|---|---|---|
| 数据流类型 | `DATA_SEGMENTATION_MASK = 0x06` | `protocol.py` |
| Payload 大小 | `DATA_PAYLOAD_SIZES[0x06] = None` | `protocol.py` |
| Payload 构建 | `Ti2026Coordinator._build_seg_mask_payload()` | `coordinator.py` |
| 订阅 dispatch | `_build_stream_payload()` 分支 `0x06` | `coordinator.py` |
| 模型 | `steelsheet_npu_only.mud` (YOLO11n-seg, labels: Metal-sheet) | `project_config.yaml` |
