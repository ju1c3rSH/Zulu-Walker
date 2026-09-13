# Zulu-Walker — Agent Guide

## 项目概述

基于 MaixCAM2 (Axera AX630C) 的视觉模块。**当前分支 `EOD2026-NearCam`**：2026 工科赛「反恐排爆救援机器人」**近端视觉从站**——爆炸物/靶标/人质识别，对接 **panzer_Brain**（机械臂大脑 MCU，主站）USART1 订阅制链路，协议 **panzer_links_protocol.md v2.1**（`docs/competition/gks_panzer_links_protocol_v2.1.md`，四源核对记录 `docs/competition/gks_protocol_crosscheck.md`）。

> 旧 EOD2026 v3.0 协议（0x01 QR_TASK…0x06 HOME_DEV）分支已不再使用，代码保留在 `EOD2026-Spacial` 分支。

## 核心架构

```
app/            应用层 — GksCoordinator (v2.1 订阅门控/流控/零值心跳) 
framework/      框架层 — HAL, EventBus, ModuleManager, StateMachine 引擎
modules/        功能模块 — zw_opencv_module (视觉), zw_uart_module (UART 协议)
utils/          工具库 — 日志, TUI调试台, CPU亲和性
F:\MaixPy-main\    上游 Sipeed MaixPy 固件源码 (只读 API 参考; 不确定的 maix API 优先查这里)
```

- `platform: maixcam2` — 通过 HAL 抽象层注入，实现见 `framework/hal/platforms/maixcam2/`
- 通信: UART **`/dev/ttyS4`** @ **115200**（官方警告其他波特率不可靠，勿改），UART4 引脚 A21(TX)/A22(RX) → Brain USART1
- 协议 v2.1（帧格式与 serial_protocol C 库逐字节一致）:
  - 帧型 0x20 REQUEST / 0x21 ACK / 0x22 NACK / 0x23 STOP / 0x24 DATA_STREAM；
    CRC16-CCITT 高字节在前；payload 恒 10B = seq + data_id + 8B 子载荷
  - data_id: 0x01 EXPLOSIVE / 0x02 TARGET / 0x03 RESCUE（0x10 QR 属远端，回 NACK）
  - 5B 订阅帧携带任务码（爆色/靶色/形状）；短帧 = 幂等重订阅不清锁存
  - 订阅期**零值心跳不断流**（Brain 1s stale 判定硬要求）
  - 解析器 = C `Protocol_ParserFeed` 四态状态机 1:1 移植（`zw_uart_module/parser.py`），
    含 AA-AA-55 重同步与坏帧整帧丢弃
- 近端视觉核心: `processors/gks_task_processor.py`（EXPLOSIVE/TARGET/RESCUE 单激活,
  识别逻辑照搬 `F:\2026工科赛\maixcam\工科赛上端\main.py` task1/2/3: 三色同帧排序 +
  两帧同序锁存 + `pit_left_is_1` 镜像开关 + 归一化误差定点 ×10000）
- 显示 HUD: `modules/zw_opencv_module/display/hud.py`（状态栏/检测叠加/锁存进度,
  不依赖 AI 模型加载）
- MaixPy 加速层: `detectors/eod_color/maix_accel.py`（设备走 C++ find_blobs, PC 自动回退
  cv2; 开关在 eod_config.yaml `accel:` 段）
- AI: `nn.YOLO11` 人质模型 `eod_hostage`（/root/models/eod_hostage_320.mud, 仓库副本
  `assets/models/hostage_yolo11n/`, 源自 人质yolo11n pulsar2 AX620E 转换链）,
  经 `MaixCam2AI` nick 热切换; 缺失时 RESCUE 回 NACK NOT_READY 不崩溃
- 阈值: `eod_config.yaml` `gks:` 段 + 运行时覆盖文件 `/my-threshold/my_64threshold.txt`
  （mtime 变化热重载不清锁存）; 设备端调参器 `python run.py debug color`（触屏 GUI）
- 测试（PC 端 pytest, 74 用例）: `python -m pytest tests/`
  - `test_protocol_gks.py`: C 自测向量移植 + md §6 业务向量 + Brain interrupt.c 消费偏移对拍
  - `test_gks_task_processor.py` / `test_coordinator_gks.py`: 照搬逻辑行为等价 + 订阅语义

## 关键文件

| 文件 | 作用 |
|------|------|
| `app/main.py` | 应用入口，初始化 Machine/ModuleManager/Coordinator/HUD |
| `app/coordinator.py` | GksCoordinator，v2.1 订阅门控 + 流控 + 零值心跳 |
| `modules/zw_uart_module/protocol.py` | v2.1 协议定义（帧型/data_id/子载荷 builders） |
| `modules/zw_uart_module/parser.py` | C 协议解析器 1:1 移植（四态状态机 + 统计） |
| `modules/zw_uart_module/uart_driver.py` | UART 收发 + 从站帧分发 |
| `modules/zw_opencv_module/processors/gks_task_processor.py` | 三任务处理器（照搬 main.py 逻辑） |
| `modules/zw_opencv_module/debug/threshold_tuner.py` | 设备端 LAB 阈值调参器（触屏） |
| `modules/zw_opencv_module/display/hud.py` | 显示 HUD |
| `framework/hal/machine.py` | 依赖注入容器，读取 `project_config.yaml` 创建平台实例 |
| `modules/zw_opencv_module/vision_manager.py` | 视觉管道管理器 (Pipeline → Task → Processor) |

## 开发命令

```bash
python run.py main                    # 运行主程序
python run.py debug color             # 设备端 LAB 阈值调参（触屏, 写 /my-threshold）
python -m pytest tests/               # PC 端测试
# 编译检查 (Windows)
python -c "import py_compile; py_compile.compile('app/coordinator.py', doraise=True)"
```

## 已知边界 / 注意事项

- 相机 640×480（与旧上端 main.py 标定几何一致）; `gks.target.roi` 等像素常量全部
  进配置，现场标定后覆盖。
- 旧版 `obj.x>240` 人质右 1/4 裁切（审查 P1-4）默认关闭（`gks.rescue.x_crop_px: 0`），
  台架复现旧行为可设像素值。
- **不设主机静默自动 idle**（与旧上端一致）: Brain 看门狗是"收不到 CAM 数据 >1s 才
  重订阅"，健康推流期间主站本就静默; Brain 侧异常由 STOP/超时兜底。视觉结果陈旧
  >1s（相机停帧/处理器异常）自动归零心跳，不透传陈旧 found=1。
- `app/line_follow_sm.py`、`processors/cargo_processor.py` 等遗留文件未删除但不参与
  流水线（vision_config.yaml 只注册 GksTaskProcessor）。
- 台架部署/验证清单: `docs/competition/gks_bench_checklist.md`。
