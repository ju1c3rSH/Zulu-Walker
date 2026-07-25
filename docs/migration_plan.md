# Zulu-Walker: MaixCAM2 电赛迁移方案

> 从 Orange Pi 5B → MaixCAM2，物流搬运赛项 → 2026 电赛
> 日期: 2026-07-25

---

## 1. 背景

原项目 Zulu-Walker 面向 2026 广东省大学生工程实践与创新能力大赛智能物流搬运赛项，采用 Orange Pi 5B (RK3588) + STM32 双计算机架构。现需迁移至 MaixCAM2 (Axera AX630C) 平台，面向 2026 全国大学生电子设计竞赛。

### 1.1 硬件平台对比

| 项目 | Orange Pi 5B | MaixCAM2 |
|------|-------------|----------|
| SoC | Rockchip RK3588 | Axera AX630C |
| CPU | 4×A76 @2.4GHz + 4×A55 @1.8GHz | 2×A53 @1.2GHz + RISC-V E907 |
| NPU | 6 TOPS (RKNN) | **3.2 TOPS INT8 / 12.8 TOPS INT4** |
| RAM | 8GB LPDDR4 | **4GB LPDDR4** |
| Camera | USB / CSI (可扩展) | **板载 8MP (4K) MIPI CSI** |
| Display | HDMI | **板载 2.4" 640×480 触摸屏** |
| OS | Ubuntu 22.04 | **Ubuntu (MaixPy 固件)** |
| AI SDK | RKNN Toolkit | **MaixPy (maix.nn) / MaixCDK** |

### 1.2 关键变化

- **CPU 核心数大幅减少** (8核→2核)，需要精简线程模型
- **AI 推理从 RKNN 转为 MaixPy maix.nn**，接口不同但抽象层级更高
- **新增板载屏幕 + 摄像头**，硬件集成度更高
- **新增 AI ISP**，可提升图像质量但会占用一半 NPU 资源
- **运行 Ubuntu，支持 apt 安装包**

---

## 2. 总体架构

```
                    ┌───────────────────────────────────────┐
                    │            app/ (完全重写)              │
                    │    Coordinator + StateMachine(赛题)    │
                    └──────────┬────────────────────────────┘
                               │ EventBus
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
 ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
 │ framework/(保留) │  │ zw_opencv_module │  │ zw_uart_module   │
 │                  │  │ (保留+增强)       │  │ (驱动保留,协议重写)│
 │ HAL maixcam2    │  │                  │  │                  │
 │ EventBus        │  │ processors/      │  │ FrameParser保留   │
 │ ModuleManager   │  │ detectors/       │  │ protocol重写      │
 │ StateMachine    │  │ AIInferenceProc  │  │ events重写        │
 └─────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 3. 各层变更明细

### 3.1 framework/ — ✅ 保留，极小修改

| 文件 | 操作 | 说明 |
|------|------|------|
| `event_bus.py` | 保留 | 通用 pub/sub 总线 |
| `module_manager.py` | 保留 | 模块生命周期管理 |
| `state_machine/base.py` | 保留 | 通用状态机引擎 |
| `state_machine/bridge.py` | 保留 | 状态动作桥接 |
| `visual_state_machine.py` | 保留 | 5 态视觉跟踪状态机 |
| `hal/interface/` | 保留 | Camera, Display, Uart, AIInference 协议 |
| `hal/machine.py` | 保留 | DI 容器 |
| `hal/camera_hub.py` | 保留 | 相机注册表 |
| `hal/platforms/maixcam2/` | 保留+增强 | 见下方 |
| `log.py` | 保留 | 日志工具 |

#### hal/platforms/maixcam2/ — 增强点

**ai.py**:
- `_SLOT_CLASSES` 增加 `"yolo26": maix.nn.YOLO11`
- 启用 `dual_buff=True` 双缓冲模式提升帧率
- 增加 `switch_model()` 运行时热切换支持

```python
_SLOT_CLASSES = {
    "yolo": maix.nn.YOLO11,
    "yolo26": maix.nn.YOLO11,      # <-- 新增
    "classifier": maix.nn.Classifier,
    "hand_landmarks": maix.nn.HandLandmarks,
    "nn": maix.nn.NN,
}
```

### 3.2 modules/zw_opencv_module/ — ✅ 保留 + 增强

| 文件 | 操作 | 说明 |
|------|------|------|
| `__init__.py` | 🔧 精简 | 移除多相机相关逻辑 |
| `vision_manager.py` | 🔧 精简 | 单管道模式，去除多相机编排 |
| `pipeline_camera.py` | 保留 | 单管道运行 |
| `task_manager.py` | 保留 | 串行任务执行 |
| `frame_composer.py` | 🔧 简化 | 单相机时可直接跳过合成 |
| `camera_stream.py` | 保留 | 可作为备选抓取方式 |
| `performance.py` | 保留 | 性能分析工具 |
| `processors/base.py` | 保留 | Processor 基类 |
| `processors/registry.py` | 保留 | 处理器注册表 |
| `processors/qr_processor.py` | 保留 | 二维码识别 |
| `processors/cargo_processor.py` | 保留 | 圆检测处理器 |
| `processors/circle_target_processor.py` | 保留 | 椭圆/靶标检测 |
| `processors/ring_discovery_processor.py` | 保留 | 圆环检测 |
| `processors/ai_inference_processor.py` | 🔧 增强 | YOLO26 适配 |
| `processors/handlers/` | 保留 | AI 画图处理器 |
| `detectors/` (全部) | 保留 | 3 个检测器全保留 |
| `models/` | 保留 | 数据模型 |
| `config/vision_config.yaml` | 🔧 更新 | 适配新赛题 |

### 3.3 modules/zw_uart_module/ — 驱动保留，协议重写

| 文件 | 操作 | 说明 |
|------|------|------|
| `__init__.py` | 🔧 修正 | 模块生命周期 |
| `uart_driver.py` | 保留 | FrameParser 状态机、STM32UartInterface 驱动层 |
| `protocol.py` | ★★ 重写 | 按电赛需求定义帧格式 |
| `events.py` | ★★ 重写 | 新的事件类型 |
| `exceptions.py` | 保留 | 异常类 |

### 3.4 app/ — ★★ 完全重写

| 文件 | 操作 | 说明 |
|------|------|------|
| `__init__.py` | 重写 | |
| `main.py` | 重写 | 简化版启动入口 |
| `coordinator.py` | 重写 | 通用协调器，赛题无关框架 |
| `state_machine.py` | 重写 | 电赛状态机（赛题公布后填充） |

### 3.5 顶层文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `run.py` | 重写 | 简化入口 |
| `project_config.yaml` | 更新 | 适配 MaixCAM2 单相机 + AI 模型 |
| `requirements.txt` | 更新 | 移除 linux-only 依赖 |
| `main.py` (根目录) | 新增 | MaixPy 应用入口，用于打包 |

---

## 4. YOLO26 集成方案

### 4.1 模型转换流程

```
ultralytics YOLO26 训练 → .pt
    ↓ docker (ultralytics==8.3.240)
ONNX (固定输入尺寸, 如 640×480)
    ↓ onnxsim 简化
简化 ONNX
    ↓ onnx.utils.extract_model (裁剪到6个输出节点)
export.onnx
    ↓ Pulsar2 build --target_hardware AX620E (INT8 量化)
.axmodel (NPU2 / NPU1 两个版本)
    ↓ + .mud 描述文件
MaixCAM2 可用模型
```

### 4.2 YOLO26 输出节点

YOLO26 有 6 个 ONNX 输出节点（YOLO11 只有 3 个）：

| 序号 | 节点名称 |
|------|----------|
| 1 | `/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0` |
| 2 | `/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0` |
| 3 | `/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0` |
| 4 | `/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0` |
| 5 | `/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0` |
| 6 | `/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0` |

### 4.3 MUD 文件 (MaixCAM2)

```ini
[basic]
type = axmodel
model_npu  = yolo26_custom_npu.axmodel
model_vnpu = yolo26_custom_vnpu.axmodel

[extra]
model_type = yolo26
type = detector
input_type = rgb
labels = target_a, target_b, target_c

input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true

mean = 0,0,0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
```

### 4.4 Pulsar2 配置 (YOLO26, MaixCAM2)

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU2",
  "quant": {
    "input_configs": [{
      "tensor_name": "images",
      "calibration_dataset": "datasets/train.tar",
      "calibration_size": 64,
      "calibration_mean": [0, 0, 0],
      "calibration_std": [255, 255, 255]
    }],
    "calibration_method": "MinMax",
    "precision_analysis": true
  },
  "input_processors": [{
    "tensor_name": "images",
    "tensor_format": "RGB",
    "tensor_layout": "NCHW",
    "src_format": "RGB",
    "src_dtype": "U8",
    "src_layout": "NHWC",
    "csc_mode": "NoCSC"
  }],
  "output_processors": [
    {"tensor_name": "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0", "dst_perm": [0,2,3,1]},
    {"tensor_name": "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0", "dst_perm": [0,2,3,1]},
    {"tensor_name": "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0", "dst_perm": [0,2,3,1]},
    {"tensor_name": "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0", "dst_perm": [0,2,3,1]},
    {"tensor_name": "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0", "dst_perm": [0,2,3,1]},
    {"tensor_name": "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0", "dst_perm": [0,2,3,1]}
  ],
  "compiler": {
    "check": 3,
    "check_mode": "CheckOutput",
    "check_cosine_simularity": 0.9
  }
}
```

### 4.5 代码调用方式

```python
from maix import nn

# YOLO26 加载 (与 YOLO11 同一 API 类，MUD 的 model_type 决定后处理)
detector = nn.YOLO11(model="/root/models/yolo26_custom.mud", dual_buff=True)

# 推理
objects = detector.detect(img, conf_th=0.5, iou_th=0.45)
for obj in objects:
    print(f"class={obj.class_id}, score={obj.score}, box=({obj.x},{obj.y},{obj.w},{obj.h})")
```

---

## 5. 线程模型 (MaixCAM2 优化)

MaixCAM2 仅 2×A53 核心，线程模型需精简：

```
Main Thread (绑定 Core 0):
  ├── ModuleManager.run_main_loop()
  ├── camera.read()                     # 同步读取 (maix.camera 有内部缓冲)
  ├── pipeline process_frame()
  │   ├── task_manager.run_tasks_serial()
  │   │   ├── OpenCV 处理器 (QR/Cargo/Circle/Ring)
  │   │   └── AIInferenceProcessor      # NPU 异步推理
  ├── coordinator.loop()
  └── display.show()

UART Receiver Thread (绑定 Core 0):
  └── FrameParser.process_byte()        # 串口监听

模型加载/切换 (按需，不常驻):
  └── maix.nn.YOLO11(path, dual_buff)   # NPU 模型加载
```

**设计原则**:
- 取消独立的 CameraStream 抓取线程（MaixCAM2 的 `maix.camera` 自带内部缓冲）
- AI 推理在 NPU 上硬件异步执行，不阻塞 CPU
- 主要计算负载在 NPU (YOLO26) 和 OpenCV 处理器 (CPU)

---

## 6. UART 协议框架

### 保留

| 组件 | 说明 |
|------|------|
| `FrameParser` | 状态机：WAITING_SOF → GOT_SOF → GOT_LEN → READING_DATA |
| `STM32UartInterface` | connect/disconnect/send/receive/start_receiver |
| XOR 校验机制 | 协议层校验 |
| 线程安全访问 | `RLock` 保护 |

### 重写

| 组件 | 说明 |
|------|------|
| `protocol.py` | 按电赛需求定义帧类型、数据结构、build/parse 函数 |
| `events.py` | 新的事件 dataclass |

---

## 7. 实施步骤

### Phase 1: MaixCAM2 基础验证 (1-2天)

- [ ] 烧录 MaixPy 固件到 MaixCAM2
- [ ] 确认 `maix.camera`、`maix.display`、`maix.nn` 可用
- [ ] 运行 YOLO26 官方 demo 验证 NPU 推理
- [ ] 确认 UART 外设可用 (`/dev/ttyS1`)

### Phase 2: 框架移植 (2-3天)

- [ ] `framework/hal/platforms/maixcam2/` — 增强 ai.py
  - [ ] 添加 `yolo26` 到 `_SLOT_CLASSES`
  - [ ] 启用 `dual_buff=True`
  - [ ] 添加多模型注册/切换支持
- [ ] `project_config.yaml` — 适配单相机 + AI 模型配置
- [ ] `run.py` — 简化入口，去掉 debug 子命令
- [ ] `app/main.py` — 重写启动逻辑
- [ ] `modules/zw_opencv_module/` — 精简 vision_manager.py 多相机逻辑

### Phase 3: AI 模型集成 (2-3天)

- [ ] 训练/获取 YOLO26 模型 (按赛题数据集)
- [ ] ONNX 导出 → 裁剪 → 简化
- [ ] Pulsar2 量化 → .axmodel
- [ ] 编写 .mud 描述文件
- [ ] 验收 `AIInferenceProcessor` + `YoloHandler` 管线

### Phase 4: app/ 层开发 (赛题公布后)

- [ ] 编写 CompetitionStateMachine
- [ ] 编写 Coordinator 协调逻辑
- [ ] `modules/zw_uart_module/protocol.py` — 新协议定义
- [ ] 集成调试

### Phase 5: 优化

- [ ] 性能调优 (AI ISP 开关权衡)
- [ ] 内存优化 (4GB 管理)
- [ ] 稳定性测试
- [ ] 打包为 MaixPy 应用

---

## 8. 保留的检测能力

| 检测器 | 方法 | 保留原因 |
|--------|------|----------|
| CargoDetector | 圆检测 (Fast/EdgeDrawing/Heuristic) | 通用圆形物体检测 |
| CircleTargetDetector | 椭圆/四边形 + UV 斑点 | 通用靶标/标记检测 |
| RingDetector | 圆环检测 | 通用环形标记检测 |
| QRCodeProcessor | pyzbar + OpenCV QR | 赛题信息读取 |
| AIInferenceProcessor | NPU YOLO26 | 通用物体检测 (赛题主力) |

---

## 9. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `main.py` (根目录) | MaixPy 应用入口 |
| `app.yaml` | MaixPy 应用打包描述 |
| `models/yolo26_custom.mud` | YOLO26 模型描述 |
| `models/yolo26_custom_npu.axmodel` | YOLO26 NPU 模型 |
| `models/yolo26_custom_vnpu.axmodel` | YOLO26 vNPU 模型 |
| `docs/migration_plan.md` | 本文档 |

### 重写文件

| 文件 | 说明 |
|------|------|
| `app/main.py` | 新启动逻辑 |
| `app/coordinator.py` | 新协调器 |
| `app/state_machine.py` | 新状态机 |
| `run.py` | 简化入口 |
| `project_config.yaml` | 新配置 |
| `modules/zw_uart_module/protocol.py` | 新协议 |
| `modules/zw_uart_module/events.py` | 新事件 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `framework/hal/platforms/maixcam2/ai.py` | 增加 yolo26 支持、dual_buff |
| `modules/zw_opencv_module/vision_manager.py` | 精简多相机 |
| `modules/zw_opencv_module/__init__.py` | 精简生命周期 |
| `modules/zw_opencv_module/config/vision_config.yaml` | 新流水线配置 |

### 删除文件

| 文件 | 说明 |
|------|------|
| `app/mission_state_machine.py` | 物流赛状态机 |
| `modules/zw_uart_module/protocol.py` (旧) | 物流赛协议 |
| `modules/zw_uart_module/events.py` (旧) | 物流赛事件 |
| `requirements-linux.txt` | 不再需要 |
| `modules/zw_opencv_module/ffmpeg_pusher.py` | 不再需要 RTMP |
| `modules/zw_opencv_module/ai/qrcode/` | 无必要 (有 QRProcessor) |
| `stories/` | 无关内容 |

---

## 10. 注意事项

### MaixCAM2 限制

1. **仅 2 个 A53 核心** — 不要启动多余线程，避免线程切换开销
2. **AI ISP 占用 NPU** — 启用 AI ISP 时 NPU 算力减半，根据赛题需求权衡
3. **内存 4GB** — 相对于 Orange Pi 的 8GB 减半，注意内存泄漏
4. **存储 32GB eMMC** — 模型文件注意体积
5. **双缓冲模式** — `dual_buff=True` 可提升帧率但增加内存占用

### 技术选型建议

1. **优先使用 `maix.nn.YOLO11` API** — 内置后处理，无需手写
2. **模型输入分辨率** — 推荐 640×480 (与屏幕比例一致) 或 320×240 (更高帧率)
3. **校准图片** — 使用真实赛题场景图片做 INT8 量化校准，而非通用图片
4. **启用 `input_cache` / `output_cache`** — 提升推理性能
