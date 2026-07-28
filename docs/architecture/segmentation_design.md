# YOLO11n-seg 钢板分割支持 — 设计说明

## 目录

1. [背景与需求](#1-背景与需求)
2. [MaixPy 分割 API 分析](#2-maixpy-分割-api-分析)
3. [类型设计](#3-类型设计)
4. [双方案 Mask 分析](#4-双方案-mask-分析)
5. [文件变更清单](#5-文件变更清单)
6. [方案切换方式](#6-方案切换方式)
7. [设计考量与取舍](#7-设计考量与取舍)

---

## 1. 背景与需求

Zulu-Walker 项目需要基于 MaixCAM2 的 YOLO11n-seg 分割模型检测钢板（钢板），需要：
- **中心坐标**：画面中钢板形状的质心（用于抓取瞄准）
- **面积**：钢板在画面中占据的像素数（用于距离/尺寸估算）

MaixCAM2 双核 A53 @ 1.2GHz，NPU 用于 YOLO 推理，CPU 资源紧张。方案必须在精度和性能之间做权衡。

## 2. MaixPy 分割 API 分析

参考 [MaixPy 图像语义分割文档](https://wiki.sipeed.com/maixpy/doc/zh/vision/segmentation.html)：

```python
detector = nn.YOLO11(model="/root/models/yolo11n_seg.mud", dual_buff=True)
objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
for obj in objs:
    detector.draw_seg_mask(img, obj.x, obj.y, obj.seg_mask, threshold=127)
    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
```

关键发现：
- `nn.YOLO11` 同时支持检测和分割模型（同一类，不同 .mud 文件）
- `detect()` 返回的对象在分割模型下多出 `obj.seg_mask` 属性（`maix.image.Image`）
- `seg_mask` 通过 `draw_seg_mask(img, x, y, mask, threshold=127)` 绘制，阈值 127 划分前景/背景
- `maix.image.image2cv(img, ensure_bgr=False)` 可将 `maix.image.Image` 转为 numpy 数组

## 3. 类型设计

### 3.1 `MaskStats` — 掩码统计信息

新增 `framework/hal/interface/ai.py:28-32`：

```python
@dataclass
class MaskStats:
    center_x: float = 0.0   # 画面坐标 (image coordinates)
    center_y: float = 0.0   # 画面坐标
    area_px: int = 0        # 掩码前景像素数
```

**设计理由**：
- 使用 `dataclass`，与现有 `Detection`、`Keypoint` 风格一致
- `center_x/y` 使用 `float`：质心可能是亚像素级别（`find_blobs` 返回浮点）
- `area_px` 使用 `int`：像素数是离散的
- 所有字段有默认值：下游代码可以安全构造零值对象

### 3.2 `Detection` 扩展

```python
@dataclass
class Detection:
    # ... 现有字段 ...
    seg_mask: Optional[np.ndarray] = None       # 新增 — 掩码数据 (h×w uint8)
    mask_stats: Optional[MaskStats] = None       # 新增 — 掩码统计
```

**设计理由**：
- 直接在 `Detection` 上添加字段，而非新建子类：
  - 下游所有代码已接受 `list[Detection]`，无需修改接口
  - 非分割模型的检测 `seg_mask=None`，零开销（一个指针大小的内存）
  - 避免了整个并行类型体系
- `seg_mask` 存 numpy 数组而非 `maix.image.Image`：
  - HAL 层的契约是平台无关的数据类型
  - numpy 可被 OpenCV/maix.image 双路径消费
  - `image2cv` 转换在 HAL 内完成，对上层透明

### 3.3 为什么不在 `Detection` 上直接放 `center_x/center_y`？

- `MaskStats` 是一个语义分组，清晰表明这些值来源于掩码分析
- 未来可扩展（如添加 `perimeter_px`、`bbox_ratio` 等）
- 不会污染 `Detection` 的命名空间（`x` 和 `center_x` 已足够混淆）

## 4. 双方案 Mask 分析

### 4.1 方案 A：`find_blobs`（C++ 原生，零拷贝）

`framework/hal/platforms/maixcam2/ai.py:400-425`

```python
def _analyze_mask_blobs(mask_img, det_x, det_y, det_w, det_h) -> MaskStats:
    def _fallback():
        return MaskStats(
            center_x=float(det_x + det_w / 2),
            center_y=float(det_y + det_h / 2),
            area_px=0,
        )
    blobs = mask_img.find_blobs(
        [[50, 100, -128, 127, -128, 127]],  # L≥50 matches threshold=127 convention
        area_threshold=50,
    )
    if not blobs:
        return _fallback()
    best = max(blobs, key=lambda b: b.area())
    return MaskStats(
        center_x=float(det_x + best.cx()),  # mask-relative → image coordinates
        center_y=float(det_y + best.cy()),
        area_px=best.area(),
    )
```

**原理**：
- `find_blobs` 是 MaixPy C++ 实现的连通域分析（connected component labeling）
- 在 `seg_mask` (`maix.image.Image`) 上直接执行，**零数据拷贝**
- LAB 阈值 `[50, 100, -128, 127, -128, 127]`：L≥50 对应 RGB≥128，与 `draw_seg_mask(threshold=127)` 对齐
- `area_threshold=50` 过滤噪声斑点

**性能**：约 100–300 μs（含连通域标记开销）
**精度**：质心 + 面积均为掩码的实际值
**内存**：零额外分配

### 4.2 方案 B：`numpy`（面积精确，中心用 bbox）

`framework/hal/platforms/maixcam2/ai.py:427-434`

```python
def _analyze_mask_numpy(mask_np, det_x, det_y, det_w, det_h) -> MaskStats:
    area = int(np.count_nonzero(mask_np > 127))
    return MaskStats(
        center_x=float(det_x + det_w / 2),  # bbox 中心
        center_y=float(det_y + det_h / 2),
        area_px=area,
    )
```

**原理**：
- `np.count_nonzero` 是 C 级 SIMD 操作（单次扫描，计数器累加）
- 中心坐标使用 YOLO bbox 中心（`x + w/2`），**不计算质心**
- YOLO seg 的 bbox 是最小外接矩形，对钢板等凸物体 bbox 中心 ≈ 质心

**性能**：约 30 μs（image2cv 拷贝 ~4KB + SIMD popcount）
**精度**：面积精确，中心近似（bbox 中心）
**内存**：一次 numpy 数组拷贝（约 4KB）

### 4.3 性能对比

| 指标 | 方案 A (find_blobs) | 方案 B (numpy) |
|------|---------------------|----------------|
| 数据拷贝 | 无（零拷贝） | image2cv 一次 (~4KB) |
| 计算方式 | C++ 连通域分析 | numpy SIMD + bbox 算术 |
| 耗时（80×60 掩码） | ~100–300 μs | ~30 μs |
| 中心精度 | 掩码质心 | bbox 中心 |
| 面积精度 | 精确 | 精确 |
| Python 开销 | 无（C++ 内完成） | numpy 调用开销 |

### 4.4 仲裁逻辑

`_compute_mask_stats` 根据 `self._mask_method` 分发：

```
mask_method="find_blobs" → _analyze_mask_blobs(mask_img, ...)
mask_method="numpy"      → _analyze_mask_numpy(mask_np, ...)
mask_method="none"       → 跳过，mask_stats=None（向后兼容）
```

## 5. 文件变更清单

| 文件 | 变更类型 | 内容 |
|------|----------|------|
| `framework/hal/interface/ai.py` | 修改 | +`MaskStats` dataclass；`Detection` 新增 `seg_mask`/`mask_stats` 字段 |
| `framework/hal/interface/__init__.py` | 修改 | 导出 `MaskStats` |
| `framework/hal/platforms/maixcam2/ai.py` | 修改 | +`yolo_seg` 注册；+mask_method 属性/设置器；detect() 提取 seg_mask 和统计；+4 个私有 mask 分析方法 |
| `framework/hal/machine.py` | 修改 | 从 `project_config.yaml` 读取 `mask_method` 并注入 AI 实例 |
| `project_config.yaml` | 修改 | +`mask_method: "find_blobs"` 配置项 |
| `modules/zw_opencv_module/processors/handlers/segmentation.py` | **新建** | `SegmentationHandler`（注册为 `"yolo_seg"`），绘制 bbox + 掩码叠加 + 中心点 + 面积 |
| `modules/zw_opencv_module/processors/handlers/__init__.py` | 修改 | 导入 `SegmentationHandler`（触发注册） |
| `modules/zw_opencv_module/vision_manager.py` | 修改 | `_draw_detections_on_image` 支持 seg_mask 绘制（maix.image 路径） |

### 未修改的文件

- `framework/hal/platforms/linux/ai.py` — stub，无 NPU，无需变更
- `framework/hal/platforms/mock/ai.py` — stub，无需变更
- `modules/zw_opencv_module/processors/ai_inference_processor.py` — 透传 `list[Detection]`，不检查具体字段
- `app/coordinator.py` — 仅读取检测数量，不涉及掩码

## 6. 方案切换方式

### 6.1 YAML 配置（推荐）

修改 `project_config.yaml` 中一行即可：

```yaml
ai:
  mask_method: "find_blobs"   # 方案 A — C++ 原生，质心+面积
  # mask_method: "numpy"      # 方案 B — numpy，面积+bbox中心
  # mask_method: "none"       # 关闭（向后兼容，性能最优）
  models:
    - nick_name: "steel_plate_seg"
      model: "/root/models/steel_plate_seg.mud"
      model_type: "yolo_seg"       # 分割模型必须显式指定 yolo_seg
    - nick_name: "yolo11n"
      model: "/root/models/steelball_640.mud"
      model_type: "auto"           # 普通检测模型
  active: "steel_plate_seg"
```

### 6.2 运行时切换

```python
ai.set_mask_method("find_blobs")  # 或 "numpy" / "none"
```

当前方案可通过 `ai.mask_method` 属性查询。

### 6.3 模型类型说明

- `model_type: "auto"` → 解析为 `maix.nn.YOLO11`，handler 使用 `"yolo"`（普通检测绘制）
- `model_type: "yolo_seg"` → 解析为 `maix.nn.YOLO11`（同一类），handler 使用 `"yolo_seg"`（分割绘制）
- 两者使用相同的 `maix.nn.YOLO11` 类，区别仅在 handler 选择
- 分割模型 `detect()` 结果中才会出现 `seg_mask` 属性（由 .mud 文件决定）

## 7. 设计考量与取舍

### 7.1 为什么不在 YOLO → numpy 之后用 `np.where` + `np.mean` 算质心？

`np.where(mask > 127)` 返回两个布尔索引数组（大小 = 前景像素数），`np.mean` 再做一次归约。
对于一个 80×60 掩码中 2000 个前景像素：
- `where` 分配 `(2000,) * 2` int64 数组 = 32KB
- `mean` 做两次浮点归约

在 MaixCAM2 的双核 A53 上，每次帧都做这套操作不可取。方案 B 主动放弃质心精度换取无分配的计算路径。

### 7.2 为什么 find_blobs 的阈值用 `[50, 100, -128, 127, -128, 127]` 而不是精确范围？

`find_blobs` 工作于 LAB 颜色空间。二值掩码的"白色"像素在 RGB 中为 (255,255,255)，LAB 中 L=100。`draw_seg_mask(threshold=127)` 以 RGB≥128 为前景，对应 LAB L≥50。A/B 通道设为全范围确保不会因为格式转换中的微量色差而漏检像素。

### 7.3 为什么 `seg_mask` 总是转 numpy 而不是保持 `maix.image.Image`？

HAL 层的契约：向上层（应用层）暴露平台无关类型。`numpy` 是 Python CV 生态的通用载体：
- OpenCV（`cv2`）路径直接使用 `np.ndarray`，由 `SegmentationHandler` 绘制 alpha 混合叠加层
- MaixCAM2 显示帧路径：`detect(_raw=True)` 内部调用 `self._model.draw_seg_mask(img, ...)` 直接在 `maix.image.Image` 上绘制掩码，随后 `_draw_detections_on_image` 在之上叠加 bbox 和中心点标记（`draw_circle`）

如果保持 `maix.image.Image`，则 Linux/Mock 平台无法消费，且 `Detection` 的序列化/日志变得困难。

### 7.4 `mask_method` 为什么是 AI 实例级而非模型级？

mask 分析策略是**运行时性能权衡**，而非模型属性：
- 同一硬件上对不同模型可能用不同策略
- 可以在不重新加载模型的情况下切换（调试/对比用）
- 如果放在模型级，每次 `switch()` 都需传递，增加注册表复杂度

### 7.5 向后兼容性

- 无 `mask_method` 配置 → 默认 `"none"` → 不计算 `mask_stats`，`seg_mask` 为 None → 行为与修改前完全一致
- 使用 `model_type: "auto"` 的检测模型 → handler 仍为 `"yolo"` → 绘制行为不变
- Linux/Mock 平台的 AI stub 不受影响（无 NPU，detect() 返回空列表）
- 现有的 `get_mask()` 方法保留（返回 None），不破坏 `AIInference` 协议

### 7.6 为什么不用 `np.count_nonzero` 而用 `np.sum`？

对于布尔数组，`np.sum` 调用 `np.count_nonzero`。直接用 `count_nonzero` 避免类型提升和额外调用层。在 C 级别两者相同，但 Python 开销不同。
