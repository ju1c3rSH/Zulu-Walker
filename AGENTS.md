# Zulu-Walker — Agent Guide

## 项目概述

基于 MaixCAM2 (Axera AX630C) 的视觉小车，2026 全国大学生电子设计竞赛 **H 题（车载平衡滚球运动控制系统）**。当前形态：YOLO11n 检测摆杆上钢球位置，经 α-β 滤波后通过 UART 流式发送给 MSPM0G3519 主机。

## 核心架构

```
app/            应用层 — main.py 启动流程 + Ti2026Coordinator (从机数据桥接)
framework/      框架层 — HAL, EventBus, ModuleManager, StateMachine 引擎
modules/        功能模块 — zw_opencv_module (视觉), zw_uart_module (UART), zw_wifi_stream (WiFi图传)
utils/          工具库 — 日志, TUI调试台, CPU亲和性
F:\MaixPy-main\    上游 Sipeed MaixPy 固件源码 (只读参考, 在仓库外)
```

- 平台通过 HAL 注入：`framework/hal/platforms/{maixcam2,linux,mock}/`，由 `Machine.create("project_config.yaml")` 创建
- **主从协议 v3.0**：MaixCAM2 是**从机**，MSPM0 是主机。从机收到 `CMD_REQUEST(0x20)` 后才流式发 `DATA_STREAM(0x24)`（钢球位置、目标位置等订阅类型）。旧 v2.1 心跳帧(0x15)/伺服帧(0x17)已标 DEPRECATED。实现见 `modules/zw_uart_module/protocol.py`，完整文档见 `docs/competition/master_slave_protocol.md`
- 硬件现状（以 `project_config.yaml` 为准）：UART `/dev/ttyS4` @ 921600；摄像头 1280×352 @60fps 带软件 AEC；模型 `/root/models/steelball_1280x352.mud`

## 关键事实（易错）

- **入口是 `app/main.py` 的 `main()`**（根目录 `main.py` 只是 MaixPy 打包转发壳）。启动顺序敏感：WDT(10s) 构造后立即喂 → WiFi AP 启动 → Phase-1 轨道标定（独占摄像头读帧）→ `Machine.create`（模型加载后再喂狗）→ 主循环。改动启动流程时保持 `wdt_feed()` 分布，否则板端会被看门狗复位。
- **摆杆轨道标定两阶段**：Phase-1 开机自动测倾角；Phase-2 触摸屏"标定"按钮用钢球位置设原点。结果持久化回写 `project_config.yaml` 的 `pendulum.rail_calibration`。代码在 `modules/zw_opencv_module/detectors/pendulum_calibrator/`
- **`app/mission_state_machine.py`（34 状态竞赛 SM）是旧赛题遗留，main.py 已不引用**。不要往里加逻辑；当前赛题逻辑在 `app/coordinator.py` (`Ti2026Coordinator`)
- 钢球位置走 α-β 滤波 + 空间门控（速度/帧率约束、连续帧 arm/drop），常量集中在 `coordinator.py` 头部；理论见 `docs/theory/alpha_beta_filter_guide.md`
- 图传录制信令：UDP port 5000（广播/单播，`app/main.py _send_record_cmd`）；设备发现 beacon port 9999；PC 心跳检测 `app/pc_heartbeat.py`

## 开发命令

```bash
python run.py main                    # 运行主程序 (需板端 maix 固件环境)
python run.py debug cargo             # 物料块检测调试
python run.py debug circle            # 圆靶检测调试
python run.py debug ring              # 环检测调试
python run.py debug line              # 黑线循迹调试
python run.py debug pendulum_calib    # 摆杆标定调试
python run.py debug cargo -c 1 -W 1280 -H 720   # 指定摄像头索引/分辨率

# Windows 上无 maix/cv2 依赖，只能做语法/导入级检查（已验证可用）：
python -c "import py_compile; py_compile.compile('app/main.py', doraise=True)"
python -c "from framework.hal import Machine; print('OK')"
```

- 无测试套件、无 lint 配置；验证 = 上述编译/导入检查 + 板端实测
- 依赖：`requirements.txt` (numpy/pyyaml/rich/pyzbar)；Linux 桌面调试另装 `requirements-linux.txt` (pyserial/opencv-contrib-python)；`maix.*` 只存在于 MaixPy 固件内

## MaixPy-main — maixcam API 参考

`F:\MaixPy-main\` 是 Sipeed/MaixPy 上游固件源码克隆。本项目用的 `maix.camera/display/nn/peripheral.uart/image/sys/wdt` 等模块均定义在固件 C++ 层（pybind11 导出），pip 里查不到——**API 用法优先在此目录搜源码和示例**：

| 路径 | 内容 |
|------|------|
| `maix/`, `maix/v1/` | Python 层模块导出与 v1 兼容层 |
| `components/maix/include/`, `src/` | C++ 绑定 (`maixpy.hpp`, `convert_image.hpp`) |
| `projects/`, `examples/` | 官方工程与用法示例 |
| `docs/doc/zh/` | 中文 API 文档 |

## 文档地图

| 文档 | 说明 |
|------|------|
| `PROJECT_CONTEXT.md` | 架构/线程模型/数据流最全，但硬件与协议细节已滞后于 v3.0 改版（仍写 ttyS3@115200、LineFollowCoordinator、640×640），以 `project_config.yaml` 和 `protocol.py` 为准 |
| `docs/competition/extracted/H题_车载平衡滚球运动控制系统.md` | 赛题原文，实现赛题逻辑前必读 |
| `docs/competition/master_slave_protocol.md` | 协议 v3.0 完整定义（含 MSPM0 侧实现指南） |
| `docs/architecture/`, `docs/pitfalls/`, `docs/theory/` | 架构设计、历史陷阱、滤波理论 |

## 已过时/需注意

- `modules/zw_opencv_module/processors/` 中 cargo/circle_target/ring_discovery/qr 处理器已被 `app.yaml` exclude，代码保留但不在打包内（`ai_inference_processor` 仍在用）
- `docs/migration_plan.md`（Orange Pi 5B → MaixCAM2 迁移）已完成，细节滞后
- `docs/pitfalls/vision_data_flow.md` 描述的问题已由 `_pending_results` deque + `drain_results()` 解决
- `docs/TODO.md` 中 seg mask 条目引用的 `plate_seg` 模型条目已从 `project_config.yaml` 移除，该问题不再适用
