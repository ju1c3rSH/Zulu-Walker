# Cargo 圆形检测策略与调参指南

## 1. 适用场景

- 相机俯视正拍，目标为纯色圆形物料（红 / 绿 / 蓝）。
- 物料表面存在 3D 打印纹理，边缘在光照下会产生反光。
- 画面中可能存在其他圆形干扰物，需要按输入 `Color` 进行区分。

## 2. 检测方法：`EDGE_DRAWING_CIRCLE`

### 2.1 整体流程

```text
输入 frame + target_color
  ↓
[ROI 优先] 若存在上一帧中心，先在 ROI 内检测；失败再全局搜索
  ↓
降采样到 640×480
  ↓
灰度化 + GaussianBlur（抑制 3D 打印纹理）
  ↓
EdgeDrawing 提取边缘
  ↓
形态学闭运算（连接反光导致的边缘断裂）
  ↓
查找闭合轮廓
  ↓
对每个轮廓：
  - 面积过滤
  - 圆度过滤（4π·area/peri²）
  - cv2.fitEllipse 拟合椭圆/圆，得到浮点候选中心
  - 颜色验证：统计候选圆内部目标颜色占比
  - 综合评分
  ↓
选取得分最高候选
  ↓
亚像素精修（颜色 mask 矩）
  ↓
Kalman 滤波平滑 → 返回 CargoItem
```

### 2.2 反光处理策略

物料边缘反光会在图像中形成亮边，容易导致：

1. 真实边缘处梯度断裂；
2. 检测到一个“仅由反光晕圈”构成的假圆。

对应处理：

- **预处理**：用 `GaussianBlur` 平滑纹理与反光过渡，降低 EdgeDrawing 对细小边缘的敏感度。
- **边缘图闭运算**：对 EdgeDrawing 输出的二值边缘图执行 `MORPH_CLOSE`，连接反光造成的缺口。
- **颜色验证过滤**：在候选圆内部用 HSV 颜色范围统计目标颜色占比，低于阈值则丢弃。反光晕圈内部不是目标颜色，因此会被自然过滤。
- **同心圆过滤**：若出现近似同心圆，优先保留内部颜色匹配度高的轮廓；长轴/短轴比过大时也丢弃。

### 2.3 亚像素中心策略（方案 A）

采用 **边缘拟合椭圆 + 颜色 mask 矩** 的混合策略：

1. 用 `cv2.fitEllipse` 对边缘轮廓拟合，得到浮点候选圆（中心、长短轴、角度）。
2. 在候选圆内部生成目标颜色 mask。
3. 对颜色 mask 计算 `cv2.moments()`，用 `m10/m00`、`m01/m00` 得到最终亚像素中心。

**优点**：

- 边缘提供形状约束，保证圆的整体轮廓正确；
- 颜色矩对边缘噪声、反光缺口不敏感；
- 比纯几何最小二乘圆拟合更抗遮挡和局部边缘缺失。

**替代方案**：

- 若后续实测颜色 mask 破洞严重，可改用最小二乘圆拟合；
- 若对实时性要求极高，可直接用颜色 mask 矩，但抗干扰性会下降。

## 3. 卡尔曼滤波调参

### 3.1 参数含义

`CargoDetector` 使用匀速模型（Constant Velocity），状态向量为 `[x, y, vx, vy]`。

| 参数 | 含义 | 影响 |
|---|---|---|
| `measurementNoiseCov` (R) | 测量噪声 | 越大越信任模型预测，输出越平滑但滞后越大 |
| `_kf_q_base` | 位置过程噪声 | 越大允许位置突变越多 |
| `_kf_q_vel_base` | 速度过程噪声 | 越大允许速度变化越快 |

### 3.2 调参顺序

1. 先把 `measurementNoiseCov` 固定在一个中间值（如 2.0~3.0），不动过程噪声；
2. **观察静止画面**：
   - 中心点还抖 → 增大 R，减小 `_kf_q_base`；
   - 中心点很稳 → 进入下一步；
3. **观察小车/相机移动画面**：
   - 滞后明显 → 减小 R，增大 `_kf_q_vel_base`；
   - 跟随过快/抖动 → 增大 R，减小 `_kf_q_vel_base`；
4. **观察丢帧恢复**：
   - 丢帧后预测飘太远 → 减小 `max_lost_frames` 或减小 `_kf_q_vel_base`；
   - 丢帧后恢复太慢 → 增大 `max_lost_frames`。

### 3.3 推荐初值

针对 `EDGE_DRAWING_CIRCLE` + 亚像素中心的精度，建议从以下值开始：

```python
measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
_kf_q_base = 0.2
_kf_q_vel_base = 0.15
max_lost_frames = 10
```

旧比赛参数通常针对其他运动特性，不建议直接照搬。

## 4. Debug 参数调参指南

在 `python run.py debug` 的 Cargo Debug 窗口中可通过滑块实时调整。

### 4.1 通用参数

| 参数 | 现象 | 调整方向 |
|---|---|---|
| `roi_size` | ROI 太小容易跟丢移动目标 | 适当增大 |
| `max_roi_miss` | ROI 连续失败次数 | 根据帧率调整 |
| `min_area` | 小噪声被误检 | 增大；小目标漏检则减小 |
| `min_circularity` | 轻微椭圆/缺边被丢弃 | 过大时减小 |
| `smooth_window` | 历史平均窗口 | 越大越稳但越滞后 |

### 4.2 EdgeDrawing 专用参数

| 参数 | 现象 | 调整方向 |
|---|---|---|
| `blur_sigma` 过小 | 边缘图受纹理干扰，碎边多 | 增大 |
| `blur_sigma` 过大 | 边缘变粗，圆边界模糊 | 减小 |
| `ed_gradient_threshold` 过高 | 边缘缺失，圆不完整 | 降低 |
| `ed_gradient_threshold` 过低 | 背景噪声边缘多 | 提高 |
| `ed_min_path_length` 过大 | 短边缘被忽略，圆缺段 | 减小 |
| `ed_min_path_length` 过小 | 保留过多碎边 | 增大 |
| `edge_morph_kernel` 过小 | 反光缺口没连上 | 增大 |
| `edge_morph_kernel` 过大 | 不同物体会粘连 | 减小 |
| `color_match_threshold` 过高 | 正确圆被过滤 | 降低 |
| `color_match_threshold` 过低 | 误检其他颜色圆 | 提高 |

### 4.3 推荐默认参数

```yaml
edge_drawing_circle:
  blur_kernel: 5
  blur_sigma: 1.5
  ed_min_path_length: 50
  ed_gradient_threshold: 36
  edge_morph_kernel: 3
  edge_morph_iterations: 1
  color_match_threshold: 0.6
  min_circularity: 0.75
  min_area: 100
  roi_size: 150
  max_roi_miss: 5
  smooth_window: 5
```

## 5. 与 `FAST_CIRCLE` 的切换

- Debug 窗口提供 `Method` 滑块：
  - `0` = `FAST_CIRCLE`
  - `1` = `EDGE_DRAWING_CIRCLE`
- 切换时自动清空历史缓存（`last_center`、卡尔曼状态等），避免跨方法污染。
- 若运行环境没有 OpenCV contrib（`ximgproc`），`EDGE_DRAWING_CIRCLE` 会自动回退到 `FAST_CIRCLE`。

## 6. HSV 颜色范围

默认复用 `CargoDetector` 中定义的现有范围：

- **Red**: `[0,30,0]~[10,255,255]` + `[170,30,0]~[180,255,255]`
- **Green**: `[40,50,50]~[80,255,255]`
- **Blue**: `[100,50,50]~[130,255,255]`

如果现场光照导致颜色偏移，优先在 debug 模式下微调这些范围，而不是改检测逻辑。

## 7. 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|---|---|---|
| 检测不到圆 | 边缘图太干净或太脏 | 调 `ed_gradient_threshold` 和 `blur_sigma` |
| 圆边界不完整 | 反光缺口未闭合 | 增大 `edge_morph_kernel` 或迭代次数 |
| 中心点抖动 | 颜色 mask 有洞 / 卡尔曼参数不合适 | 加 blur、调 KF 参数 |
| 误检其他颜色圆 | `color_match_threshold` 太低 | 提高阈值或收紧 HSV 范围 |
| 快速移动时丢失 | ROI 太小或 KF 滞后 | 增大 `roi_size`、减小 R、增大 `_kf_q_vel_base` |

## 8. 验证步骤

1. 删除旧的 `modules/zw_opencv_module/config/cargo_debug_params.yaml`，让默认值重建；
2. 运行 cargo 调试入口；
3. 切到 `EDGE_DRAWING_CIRCLE`，观察 Mask 预览中的边缘图是否完整包围圆；
4. 依次调 `blur_sigma`、`ed_gradient_threshold`、`edge_morph_kernel`；
5. 调 `color_match_threshold` 确保 R/G/B 不误检；
6. 切到 `FAST_CIRCLE` 做 A/B 对比，观察中心点稳定性。
