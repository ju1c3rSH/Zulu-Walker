# 摆杆水平线整定 — 技术文档

> 适用项目：Zulu-Walker (MaixCAM2 + MSPM0)  
> 创建时间：2026-07-30  
> 相关文件：`modules/zw_opencv_module/detectors/pendulum_calibrator/`

---

## 1. 问题背景

摆杆（白色 PPR 水管，25cm × 2cm 外径，剖半凹槽）在摄像头画面中不可能保持绝对水平。传统做法仅用钢球 bounding box 中心 X 坐标计算位置误差，忽略 Y 分量。当摆杆存在倾角 θ 时，钢球沿摆杆移动会同时改变 (x, y)，仅用 x 会导致系统误差。

**目标**：自动检测摆杆轴线（方向 + 中心），将钢球 (cx, cy) 投影到轴线上计算 1D 位置误差，替代原有的纯 X 轴方案。

**约束**：
- 不能对摆杆放置人为高对比度标记（赛题规则）
- 摄像头垂直固定在摆杆上方，背景为黑色车体
- 检测**仅用 OpenCV**，不占用 NPU

---

## 2. 架构概览

```
PendulumCalibrator (纯 OpenCV)
    │
    ├── _detect_rail_column_centroid()  ← 主方法: 列中心线 + fitLine (自适应倾角)
    ├── _detect_rail_contour()          ← 回退: 阈值分割 + 轮廓 + minAreaRect
    └── _detect_rail_edges()            ← 降级: Canny + HoughLinesP 聚类
         │
         ▼
    RailCalibration (frozen dataclass)
         │
         ├── project(px, py) → float     ← 点→轴线投影
         ├── replace_origin(x, y)        ← Phase2 球心更新原点
         ├── to_dict() / from_dict()     ← YAML 持久化
         └── dir_cos / dir_sin            ← 预计算方向余弦
```

---

## 3. 核心算法

### 3.1 主方法：列中心线 + fitLine（自适应倾角）

```
帧 BGR
  → cv2.cvtColor(COLOR_BGR2GRAY)
  → 阈值降级: column_threshold(180) → 150 → 120, 首个成功即停
  → 逐列扫描白色像素: top/bottom 中点 = 轨道中心线采样点 (x, (top+bottom)/2)
  → 高度自适应: 取各列白跨度中位数 med, 接受 [0.6·med, 1.6·med] 内的列
       (剔钢球遮挡列/镜面噪声列; 管为圆弧面时亮带只有弧顶也能自适应)
  → 护栏: med ≥ 5% 帧高 (防整帧背景色块); 拟合中心须在中部带 [0.2h, 0.8h]
  → 至少 50% 列有效，否则 fail_reason='column_insufficient'
  → cv2.fitLine(DIST_L2) 拟合所有中心点 → 直线方向 + 中心点
  → 方向 vx<0 取反 → angle_rad = atan2(vy, vx)
```

**为什么主方法选列中心线**：轨道横向铺满/超出画面宽度时，`minAreaRect` 对贴边或整帧色块返回帧对齐矩形，倾角被帧边界锁死恒为 0。逐列中心点拟合不依赖轮廓包围盒，倾角完全自适应，且钢球黑色块（遮挡约 50px）会改变该列白跨度而被过滤。

**为什么高度门限用中位数自适应而非固定窗口**：PPR 管是半剖圆弧面，俯视时只有弧顶最亮，固定 `pipe_h_min/max` 会误杀。改为逐帧取白跨度中位数 `med`，接受 `[0.6·med, 1.6·med]`，轨距变化、亮带收窄、球/噪声列均可自动处理，无需人工调参。

### 3.2 回退方法：阈值分割 + 轮廓 + minAreaRect

```
帧 BGR
  → cv2.cvtColor(COLOR_BGR2GRAY)
  → np.std(gray) < 30 ?
       Yes → cv2.threshold(fixed, binary_threshold)
       No  → cv2.threshold(OTSU)
  → cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
       RETR_EXTERNAL 忽略钢球孔洞（球为黑色，管体为白色）
  → max(contours, key=cv2.contourArea)
  → 校验：area > frame_w*frame_h*min_contour_area_ratio
  → 校验：area < frame_w*frame_h*max_contour_area_ratio (0.55)  ← 拒绝整帧色块(原 angle 恒 0 根因)
  → cv2.minAreaRect(largest_contour)
  → 校验：w/h > min_aspect_ratio
  → 校验：center 在 [10%, 90%] 区间内
  → angle ∈ [-90°，0)，若 angle < -45 → angle += 90
  → angle_rad = math.radians(angle)
```

**为什么不用 Otsu 做唯一阈值**：低对比度场景（std<30）下 Otsu 结果不可靠，退化到固定阈值。

### 3.3 降级方法：Canny + HoughLinesP

```
gray
  → cv2.GaussianBlur(5,5)
  → cv2.Canny(canny_low, canny_high)
  → 早退：np.sum(canny>0)==0 → None
  → cv2.HoughLinesP(rho=1, theta=1°, threshold, minLineLength, maxLineGap=50)
  → 筛 |line_angle| < edge_angle_max_deg° 的线段
  → 至少 3 条合格线 → 加权平均中点 + 平均角度
```

### 3.4 降级链

`calibrate()` 依次尝试：**列中心线 → contour → edge**，三者全失败才返回 `calibrated=False`。`_diag['method']` 记录实际命中的方法（`column_centroid` / `contour` / `edge`）。

### 3.5 投影公式

```python
def project(px, py):
    dx = px - origin_x
    dy = py - origin_y
    return dx * dir_cos + dy * dir_sin   # 正值 = 原点右侧
```

`pe_x` 和 `ball_cm` 均在投影距离上计算，MSPM0 协议字节序完全不变。

---

## 4. 可调参数

全部通过 `project_config.yaml` 的 `pendulum.calib_params` 段注入：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `binary_threshold` | 127 | Otsu 退化固定阈值 (contour 回退) |
| `min_contour_area_ratio` | 0.04 | 最小轮廓面积 / 画幅面积 (contour 回退) |
| `max_contour_area_ratio` | 0.55 | 最大轮廓面积 / 画幅面积 (contour 回退), 拒绝整帧色块 |
| `min_aspect_ratio` | 1.0 | 最小 w/h 比 (contour 回退) |
| `column_threshold` | 180 | 列中心线法: 首个阈值, 失败自动降级 150→120 |
| `canny_low` | 50 | Canny 低阈值 (edge 降级) |
| `canny_high` | 150 | Canny 高阈值 (edge 降级) |
| `hough_threshold` | 50 | HoughLinesP 累加器阈值 (edge 降级) |
| `hough_min_line_len` | 150 | 最小线段长度 (px) (edge 降级) |
| `edge_angle_max_deg` | 15 | 边缘线段最大倾角 (°) (edge 降级) |

```yaml
# project_config.yaml
pendulum:
  pixels_per_cm: 25.6
  length_cm: 25.0
  calib_params:               # 可选，按需解除注释
    binary_threshold: 127
    min_contour_area_ratio: 0.04
    max_contour_area_ratio: 0.55
    min_aspect_ratio: 1.0
    column_threshold: 180
```

---

## 5. 双阶段标定流程

### Phase 1：启动时自动列中心线标定

```
启动 → 检查 project_config.yaml 有无持久化标定
  ├─ 有 → 加载，跳过所有标定
  └─ 无 → 从 CameraHub 读一帧
           → PendulumCalibrator.calibrate(frame_bgr)  // 列中心线 → contour → edge 降级链
           → 成功 → 存储轨道轴线 + 中心点
                   → LCD 右下角显示【定位】按钮
           → 失败 → 退化到旧行为 (cx - frame_width/2)
```

### Phase 2：手动球放 0cm 触控确认

```
Phase1 完成后，LCD 右下角显示【定位】按钮（持续可见直到确认）

用户把钢球放到 0cm 刻度处 → 触摸【定位】按钮
  → coordinator.calibrate_origin_from_ball()
     读取 _latest_ai 中 YOLO class_id=0 检测结果
     → 找到球 → origin = (ball.x+w/2, ball.y+h/2)
       → call RailCalibration.replace_origin(cx, cy)
       → LCD 绿色闪烁框 ×1s 确认
       → 按钮消失
       → 持久化到 project_config.yaml
     → 未找到球 → log_print 失败原因，按钮保持可见
```

### 降级路径

| 标定状态 | pe_x 计算 | 按钮 |
|----------|-----------|------|
| 持久化加载成功 | `project(cx, cy)` | 无 |
| Phase1 完成、Phase2 未做 | `project(cx, cy)`（原点=中心点） | 可见 |
| Phase2 球心确认 | `project(cx, cy)`（原点=精确） | 消失 |
| Phase1 失败 | `cx - frame_width/2`（原有行为） | 无 |

---

## 6. 诊断系统

### 6.1 `_diag` 字典

每个 `PendulumCalibrator.calibrate()` 调用重置 `_diag`，沿途记录：

| 键 | 说明 |
|----|------|
| `method` | 命中方法：`column_centroid` / `contour` / `edge` |
| `column_threshold` | 列中心线法实际命中阈值 |
| `column_points` | 列中心线法有效列数 |
| `column_median_h` | 列中心线法白跨度中位数 (px) |
| `column_fail_reason` | 列中心线法失败原因：`column_insufficient`, `column_median_height`, `column_off_band` |
| `fail_reason` | 最终失败原因码：`max_area`, `no_contours`, `min_area`, `aspect_ratio`, `center_out_of_bounds`, `hough_insufficient_lines`, `no_edges`, `hough_no_lines`… |
| `gray_mean / gray_std` | 灰度图均值和标准差 |
| `threshold_method / threshold_ret / white_px_pct` | 阈值方式(Otsu/fixed)、结果值、白色像素比 (contour) |
| `contours_found / contours_max_area / contour_area` | 轮廓数、最大面积、选中轮廓面积 (contour) |
| `contour_aspect / min_aspect_limit` | 长宽比 vs 阈值 (contour) |
| `rect_center / rect_angle` | minAreaRect 中心和角度 (contour) |
| `edge_lines_raw / edge_lines_filtered` | Hough 原始线数 vs 滤后线数 (edge) |
| `used_fallback` | 是否走到 edge 降级 |

### 6.2 `debug_device.py` 独立调试工具

```bash
cd /maixapp/apps/Zulu-Walker
python3 modules/zw_opencv_module/detectors/pendulum_calibrator/debug/debug_device.py
```

交互式命令：
- `Enter` — 拍一帧跑标定
- `calib.binary_threshold=100` — 在线调参
- `?` — 显示当前配置
- `save` — 持久化到 `calib_debug.yaml`
- `q` — 退出

输出：LCD 实时预览（标注帧 + 二值化帧上下拼接）+ 串口诊断文本 + `/root/calib_debug_frame.png` + `/root/calib_debug_binary.png`

配置文件自动热加载（检测 mtime 变化）。

---

## 7. LCD 按钮 UI

### 绘制

`VisionManager._draw_calib_button()` — 右下角，黑底 + `assets/calibrate.png` 图标覆盖。

在 `_update_display_frame()` 的调用顺序：
```
_draw_overlays → _draw_calib_button → _draw_calib_flash → _draw_exit_icon
```

### 触控

`main.py` 的 `main_callback` 中，在 exit 按钮检测之后：
```python
calib_rect = vm.get_calib_button_rect()
if _in_btn(tx, ty, cbx, cby, size=cbw):
    coordinator.calibrate_origin_from_ball()
```

按钮 rect 通过 `maix.image.resize_map_pos_reverse()` 将 LCD 触控坐标映射回帧坐标。

### 确认闪烁

`VisionManager.trigger_calib_flash(bbox)` → 在钢球 bbox 上绘制绿色实线框 1 秒（`_calib_flash_until = time.monotonic() + 1.0`）。

---

## 8. Coordinator 集成

### 新增方法

| 方法 | 职责 |
|------|------|
| `set_rail_calibration(calib)` | 设置 RaiCalibration |
| `get_rail_calibration()` | 获取当前标定 |
| `is_origin_exact()` | Phase2 是否完成 |
| `calibrate_origin_from_ball()` | 读 `_latest_ai` 球心 → replace_origin |
| `get_last_ball_bbox()` | 返回 (x,y,w,h) 供闪烁绘制用 |

### `_build_pendulum_position_payload()` 改动

```python
cx = ball.x + ball.w / 2
cy = ball.y + ball.h / 2     # ← 新增
calib = self._rail_calib
if calib and calib.calibrated:
    dist_px = calib.project(cx, cy)            # ← 沿轴投影
else:
    dist_px = cx - self._frame_width / 2.0     # ← 旧行为
pe_x = int((dist_px / half) * 5000.0)
ball_cm = dist_px / self._pixels_per_cm
```

协议字节序、范围、flag 位完全不变。

---

## 9. YAML 持久化

### project_config.yaml 段

```yaml
pendulum:
  pixels_per_cm: 25.6
  length_cm: 25.0
  rail_calibration:          # Phase2 完成后自动写入
    origin_x: 322.5
    origin_y: 318.0
    angle_rad: -0.034
    calibrated: true
  calib_params:              # 可选，调试阈值
    binary_threshold: 127
    min_contour_area_ratio: 0.04
    max_contour_area_ratio: 0.55
    min_aspect_ratio: 1.0
    column_threshold: 180
```

### 读写函数

| 函数 | 位置 |
|------|------|
| `_load_persisted_calibration(cfg)` | `app/main.py:326` |
| `_persist_calibration(calib)` | `app/main.py:360` |
| `RailCalibration.to_dict()` | `pendulum_calibrator/__init__.py:34` |
| `RailCalibration.from_dict(d)` | `pendulum_calibrator/__init__.py:45` |

---

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| `modules/zw_opencv_module/detectors/pendulum_calibrator/__init__.py` | RailCalibration + PendulumCalibrator |
| `modules/zw_opencv_module/detectors/pendulum_calibrator/debug/runner.py` | PC debug runner (cv2 trackbar) |
| `modules/zw_opencv_module/detectors/pendulum_calibrator/debug/debug_device.py` | MaixCAM2 独立调试工具 |
| `modules/zw_opencv_module/detectors/pendulum_calibrator/debug/calib_debug.yaml` | 调试参数文件 (gitignored) |
| `app/coordinator.py` | 标定存储 + 投影 + 球心标定方法 |
| `app/main.py` | Phase1/2 启动流程 + 触控回调 + 持久化 |
| `modules/zw_opencv_module/vision_manager.py` | 按钮绘制 + 确认闪烁 |
| `assets/calibrate.png` | 定位按钮图标 |
| `project_config.yaml` | `pendulum.rail_calibration` + `pendulum.calib_params` |
| `app.yaml` | 部署文件清单 |

---

## 11. 性能与安全

| 关切 | 结论 |
|------|------|
| Phase 1 耗时 | 列中心线法 ~4-8ms (A53 NEON, 逐列扫描 + fitLine); 仅启动时执行 |
| WDT 安全 | Phase 1 在 `coordinator.start()` 前执行，WDT 10s 超时内 |
| 内存 | 临时分配 < 500KB |
| 线程安全 | `_rail_calib` 仅主线程读写，`_latest_ai` 沿用 GIL 保护 |
| 运行时投影开销 | 每帧 2-3 次浮点运算 |
| 日志输出 | `start_log_writer()` 在 main() 开头调用，log_print → stdout + `logs/debug.log` |

---

## 12. 调试工作流

```
1. 实车上电，python3 debug_device.py
2. 观察串口诊断输出，确认 method=column_centroid、angle 随摆杆倾角变化（不再恒为 0）
3. 如需调整阈值 → calib.column_threshold=xxx 在线调参（Enter 每拍一帧）
4. 找到合适的参数 → save 到 calib_debug.yaml
5. 将调好的参数抄入 project_config.yaml pendulum.calib_params 段
6. 重启 python3 main.py → Phase 1 自动标定成功（日志 method=column_centroid pts=N angle=...）→ 右下角定位按钮出现
7. 球放 0cm → 按定位按钮 → 绿框闪烁确认 → 按钮消失 → 持久化完成
```
