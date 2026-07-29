# Zulu-Walker — Agent Guide

## 项目概述

基于 MaixCAM2 (Axera AX630C) 的 YOLO11n 钢珠检测小车，预留视觉循迹能力，面向 2026 全国大学生电子设计竞赛（赛题未发布，目前按通用机器人平台预研）。

## 核心架构

```
app/            应用层 — Coordinator + StateMachine (赛题逻辑)
framework/      框架层 — HAL, EventBus, ModuleManager, StateMachine 引擎
modules/        功能模块 — zw_opencv_module (视觉), zw_uart_module (UART)
utils/          工具库 — 日志, TUI调试台, CPU亲和性, 焦距计算
F:\MaixPy-main\    上游 Sipeed MaixPy 固件源码 (只读参考)
```

- `platform: maixcam2` — 通过 HAL 抽象层注入，实现见 `framework/hal/platforms/maixcam2/`
- 通信: UART `/dev/ttyS3` @ 115200, 协议 v2.1 (SOF+CRC16)
- AI: `maix.nn` YOLO11 模型 (`/root/models/steelball_640.mud`)
- 迁移自 Orange Pi 5B，详见 `docs/migration_plan.md`

## MaixPy-main — maixcam API 参考

`F:\MaixPy-main\` 是 Sipeed/MaixPy 上游固件的完整源码克隆。**任何关于 maixcam 内部 API、固件行为、编译问题，优先在此目录查找答案：**

| 路径 | 内容 |
|------|------|
| `F:\MaixPy-main\maix/` | Python 层：`maix.__init__` (模块导出)、`maix.v1/` (v1 兼容层)、`maix.sensevoice.py` 等 |
| `F:\MaixPy-main\components/maix/include/` | C++ 头文件：`maixpy.hpp`, `maixpy_bytes.hpp`, `convert_image.hpp`, `convert_tensor.hpp` |
| `F:\MaixPy-main\components/maix/src/` | C++ 源码：`maixpy.cpp` (Python 绑定生成入口) |
| `F:\MaixPy-main\components/maix/gen_api*.py` | API 绑定代码生成器 (`gen_api.py`, `gen_api_cpp.py`) |
| `F:\MaixPy-main\main/` | 固件主入口点 (`main.cpp`, `main.h`, `CMakeLists.txt`, `Kconfig`) |
| `F:\MaixPy-main\projects/` | 官方示例工程：`usb_video_camera`, `demo_xiaozhi_ai`, `app_vlm`, `app_rtsp`, `app_yolo_obb` 等 |
| `F:\MaixPy-main\configs/` | 平台编译配置：`config_platform_maixcam.mk`, `config_platform_maixcam2.mk` |
| `F:\MaixPy-main\examples/` | 使用示例：`maixpy_v1/sensor.py`, `maixpy_v1/lcd.py`, `vision/video/` |
| `F:\MaixPy-main\maix/v1/` | MaixPy v1 兼容 API：`sensor.py`, `lcd.py`, `image.py`, `video.py`, `audio.py`, `machine/` |
| `F:\MaixPy-main\docs/` | 官方文档：中文 API 文档 (`docs/doc/zh/vision/`, `docs/doc/zh/video/` 等) |
| `F:\MaixPy-main\.github/workflows/` | CI 构建脚本 (`build_maixcam.yml`, `release_maixcam.yml`) |

> 本项目使用的 `maix.*` 模块（`maix.camera`, `maix.display`, `maix.nn`, `maix.peripheral.uart`, `maix.image`, `maix.sys` 等）均定义在 MaixPy 固件的 C++ 层，通过 `pybind11` 绑定导出。若在官方文档找不到某个 API 的用法，可以直接搜索 `F:\MaixPy-main\` 源码中的调用示例。

## 关键文件

| 文件 | 作用 |
|------|------|
| `app/main.py` | 应用入口，初始化 Machine/ModuleManager/Coordinator |
| `app/coordinator.py` | LineFollowCoordinator，桥接视觉→UART |
| `app/mission_state_machine.py` | 竞赛任务状态机 (34 个状态) |
| `framework/hal/machine.py` | 依赖注入容器，读取 `project_config.yaml` 创建平台实例 |
| `framework/hal/platforms/maixcam2/` | MaixCAM2 平台实现 (camera, display, uart, ai) |
| `modules/zw_opencv_module/vision_manager.py` | 视觉管道管理器 (Pipeline → Task → Processor) |
| `modules/zw_uart_module/uart_driver.py` | UART FrameParser 状态机 (CRC16) |
| `modules/zw_uart_module/protocol.py` | 协议 v2.1 定义 (帧类型, VisualFlags) |
| `run.py` | 启动器 (main / debug 模式) |
| `project_config.yaml` | 平台/摄像头/AI 模型配置 |

## 已知待改善

- `vision_manager.py` `_update_display_frame` 已改为复用 `pipe.last_results` 缓存的检测结果（commit 5909600），NPU 推理从 2 次/帧降至 1 次/帧。有效推理结果已走 pipeline cv2 帧，而 `detect()` 内部的 `draw_seg_mask`（MaixPy Image 上画 seg mask 叠加）只在 pipeline 的 display 帧输出中绘制，LCD 端 `_update_display_frame` 不会再调 `draw_seg_mask`。LCD 仅显示 bbox + 标签 + mask 中心红点。
- `segmentation.py` SegmentationHandler 注册 key 为 `"yolo_seg"`，但 `project_config.yaml` 中 seg 模型用 `model_type: "auto"` → handler 查找 fallback 到 DefaultHandler，cv2 离线帧不会叠加 seg mask 彩色填充。若需离线流 mask 叠加，可将 `project_config.yaml` 中 `plate_seg` 条目的 `model_type` 改为 `"yolo_seg"`，或修改 handler 注册/查找逻辑。

## 赛题文档（2026 电子设计竞赛 H 题）

`docs/competition/extracted/H题_车载平衡滚球运动控制系统.md` — 2026 年全国大学生电子设计竞赛 H 题（车载平衡滚球运动控制系统）的完整题目文档，提取自官方 DOCX + PDF，含所有文字要求、评分标准和嵌入图片。**Agent 在实现赛题逻辑前应先读取此文档**，理解任务要求（循迹、摆杆控球、图传等）。

提取脚本：`docs/competition/extract_docs.py`，可复用解析同类文档。

## 已过时/需注意的信息

- TODO 中多项已标记 ✅ 修复: `cargo_confirmed` 死锁、`build_visual_servo_data_frame` float 问题、OpenCL 分散调用、未使用 VisualFlags
- `docs/migration_plan.md` 描述的迁移方案已基本完成，部分细节可能滞后于实际代码
- `modules/zw_opencv_module/processors/` 中多个处理器（`cargo_processor`, `circle_target_processor`, `ring_discovery_processor`, `qr_processor`, `ai_inference_processor`）已被 `app.yaml` exclude，但代码仍保留在仓库中

## 开发命令

```bash
python run.py main                    # 运行主程序
python run.py debug cargo             # 物料块检测调试
python run.py debug circle            # 圆靶检测调试
python run.py debug ring              # 环检测调试
python run.py debug line              # 黑线循迹调试
## 完整项目上下文

`PROJECT_CONTEXT.md` 包含完整的项目上下文（目录结构、架构细节、线程模型、数据流、配置、已知陷阱、关键技术细节等），**其他 Agent 应优先读取该文件**以避免重复探索。

## 开发命令

```bash
python run.py main                    # 运行主程序
python run.py debug cargo             # 物料块检测调试
python run.py debug circle            # 圆靶检测调试
python run.py debug ring              # 环检测调试
python run.py debug line              # 黑线循迹调试
# 编译检查 (Windows)
python -c "import py_compile; py_compile.compile('app/main.py', doraise=True)"
python -c "import py_compile; py_compile.compile('framework/hal/machine.py', doraise=True)"
# 导入检查
python -c "from framework.hal import Machine; print('OK')"
```
