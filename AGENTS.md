# Zulu-Walker — Agent Guide

## 项目概述

基于 MaixCAM2 (Axera AX630C) 的视觉模块。**当前分支 `EOD2026-NearCam`**：2026 工科赛「反恐排爆救援机器人」**近端视觉从站**——爆炸物/靶标/人质识别，对接 **panzer_Brain**（机械臂大脑 MCU，主站）USART1 订阅制链路，协议 **panzer_links_protocol.md v2.1**（`docs/competition/gks_panzer_links_protocol_v2.1.md`，四源核对记录 `docs/competition/gks_protocol_crosscheck.md`）。

> 旧 EOD2026 v3.0 协议（0x01 QR_TASK…0x06 HOME_DEV）分支已不再使用，代码保留在 `EOD2026-Spacial` 分支。

## 双平台分支（重要）

| 分支 | 设备 | 说明 |
|---|---|---|
| `EOD2026-NearCam` | MaixCAM2 (AX630C) | 主力分支，`.axmodel` + `nn.YOLO11`，`/dev/ttyS4` |
| `feat/maixcam1-port` | **MaixCAM-Pro** (CVI cv181x，有触屏) | 从 `0f4db9b` 切出的回移植分支，`.cvimodel` + `nn.YOLOv5`，`/dev/ttyS0` |

**协议 v2.1 两平台逐字节一致，Brain 侧无需改动。** 协议层/任务处理器/协调器与平台无关，
换平台只需动 HAL + 配置。**协议层用例在两分支上都必须全绿** —— 这是"只改了平台、
没碰语义"的最快信号（PC 端 `python -m pytest tests/`，maixcam1 分支 135 例）。

CAM1 平台的五处硬差异（详见 `docs/competition/gks_bench_checklist_maixcam1.md`）：

- **模型格式互不通用**：CAM2 `.axmodel`(mud 键`model_npu`) ↔ CAM1 `.cvimodel`(mud 键 `model`)，
  不可跨机复用；CAM1 现有 `model_239147.cvimodel`(YOLOv5, 三类) 来自 `F:\2026工科赛\maixcam\工科赛上端`。
- **串口用 UART0**（2026-10-07 改）：`from maix import uart` + `/dev/ttyS0`(A16/A17)。
  A16/A17 **出厂默认即 UART0_TX/RX**（`pinmap.md:42`），代码**跳过** pinmap 复用。
  ⚠⚠ **TX(A16) 是启动模式检测引脚，拉低 = 无法开机**——做 3.3V↔5V 电平转换时务必确保
  TX 默认不拉低；上电起不来先查 A16。⚠ 开机打印日志直到 `serial ready`，Brain 侧丢弃开机段。
  ⚠ UART0 同时是系统 maix protocol 串口，占用后 MaixVision IDE 抢线，台架改用 SSH、
  `app.need_exit()` 可能无法从 IDE 停止（急停断电）。
  （若改回 UART1：A19/A18 出厂是 `GPIOA19/PWM7`，**必须** pinmap 复用否则收不到字节。）
- **非默认引脚必须 pinmap 复用**：`MaixCam1Uart._ensure_pinmap` 只对**非出厂默认**的串口
  设复用（ttyS1 A19/A18、ttyS2 B0/B1）；ttyS0 故意留空跳过。CAM2 的 A21/A22 是板上默认
  引脚所以没这步，**换口后极易漏**（2026-10-07 审查 P0）。
- **模块布局不同**：`wdt` 在 CAM1 是 `maix.wdt`、CAM2 是 `maix.peripheral.wdt`；`uart` 同理。
  只写 CAM2 路径会被 `except` 吞掉 → **看门狗静默失效**（线程死锁无硬件复位兜底）。
- **帧率口径（2026-10-25 实测翻案）**：出帧 **60fps 可达**（`[FPS] pipeline` 实测）；MaixPy 的
  "取图延时 30+ms" 只限**帧龄**不限帧率——旧"上限约 30fps"是旧串行路径开销的误判。
  ⚠ `read(block=False)` 若被固件拒绝会降级为阻塞读（停帧保护失效），
  启动日志出现 `camera.read(block=...) unsupported` 即须升级固件。

⚠ **日志级别坑**：`app/main.py` 把 root logger 钉在 `logging.WARNING`，
`logger.info` 在设备上**不打印**。台架要 grep 的状态（AI 注册/切换、相机构造级别、
pinmap 结果、WDT 状态）一律用 `warning` 级别。

## 核心架构

```
app/            应用层 — GksCoordinator (v2.1 订阅门控/流控/零值心跳) 
framework/      框架层 — HAL, EventBus, ModuleManager, StateMachine 引擎
modules/        功能模块 — zw_opencv_module (视觉), zw_uart_module (UART 协议)
utils/          工具库 — 日志, TUI调试台, CPU亲和性
F:\MaixPy-main\    上游 Sipeed MaixPy 固件源码 (只读 API 参考; 不确定的 maix API 优先查这里)
```

- `platform: maixcam1`（当前工作分支）— 实现见 `framework/hal/platforms/maixcam1/`
  - CAM2 分支上为 `platform: maixcam2` — 实现见 `framework/hal/platforms/maixcam2/`
- 通信（CAM1 分支）: UART **`/dev/ttyS0`** @ **115200**（官方警告其他波特率不可靠，勿改），UART0 引脚 A16(TX)/A17(RX) → Brain USART1
  （⚠ A16 是启动模式检测引脚，**TX 拉低 = 无法开机**；UART0 开机有日志，Brain 侧需丢弃开机段）
- 协议 v2.1（帧格式与 serial_protocol C 库逐字节一致）:
  - 帧型 0x20 REQUEST / 0x21 ACK / 0x22 NACK / 0x23 STOP / 0x24 DATA_STREAM；
    CRC16-CCITT 高字节在前；payload 恒 10B = seq + data_id + 8B 子载荷
  - data_id: 0x01 EXPLOSIVE / 0x02 TARGET / 0x03 RESCUE（0x10 QR 属远端，回 NACK）
  - 5B 订阅帧携带任务码（爆色/靶色/形状）；短帧 = 幂等重订阅不清锁存
  - 订阅期**零值心跳不断流**（Brain 1s stale 判定硬要求）
  - 解析器 = C `Protocol_ParserFeed` 四态状态机 1:1 移植（`zw_uart_module/parser.py`），
    含 AA-AA-55 重同步与坏帧整帧丢弃
- 近端视觉核心: `processors/gks_task_processor.py`（EXPLOSIVE/TARGET/RESCUE 单激活）。
  识别本源 = `2025视觉方案\maixcam\工科赛上端\main.py` task1/2/3
  （`F:\2026工科赛资料汇总\04_2025视觉方案_MaixCam\`）; 打靶为归一化误差定点 ×10000。
  - **坑位定序（task1 爆炸物 / task3 人质 共用, 与 2025 方案一致）**:
    顺序 = **首次检出先后**（比赛流程固定、相机按 pit1→pit3 扫过 ⇒ 最先看见的那个就在
    pit1）; **同一帧里同时检出多个时帧内按画面左右定序**（`pit_left_is_1` 决定方向),
    保证结果不受 find_blobs/YOLO 的输出顺序影响。
    2025 原版写法: `position[2] = 6 - position[0] - position[1]` / `6 - sum(found_shapes)`。
  - **未见者补位**: 三色/三形状互不相同 ⇒ 见到 2 个即可推出第 3 个"是什么", 补在顺序尾部。
    补位只推出"是什么"、推不出"在第几坑"——位置靠定序规则给; 前提不成立会**静默锁出错误
    坑位**（转台打错坑→抓错), 比"锁不上"更危险。台架必须用日志 `latch … inferred=True`
    核对 POS/PIT 与实物一致。
  - 两个开关按任务分开配（`eod_config.yaml`）: `order_by` = `first_seen`(默认) / `x`
    (旧 main.py 的"全程画面 x 排序 + 镜像翻转"); `infer_missing` = `tail`(允许两色/两形状
    即锁, **排爆与人质都用**, 与 2025 方案一致; 2026-09-24 台架决策) / `none`(齐见才锁,
    保守退路 —— 锁不上会超时 FAULT, 可见可诊断)。
  - **台架硬要求**: 排爆既然用补位, 每次出现日志 `latch … inferred=True` 都必须核对 PIT
    与实物一致（2026-09-24 实测: `seen=[2,3]` 即锁 `pit=2`，红色全程未见也照样锁）。
- 排爆扫描降采样: `gks.explosive.scan_scale`（**当前 0.5 = 320×240 扫描**）—— 全帧三色串行
  find_blobs 是排爆帧率瓶颈（台架实测 12~19fps，慢于 Brain 的 50ms 推流节奏）。降采样后
  面积门槛按 scale² 折算、cx/框按 1/scale 还原到全帧坐标，识别语义不变；缩放用
  INTER_NEAREST 不插值（插值会把边缘混成中间色，等价于人为运动模糊）。置 1.0 回全分辨率。
  ⚠ **降采样小图必须交 RGB 字节序**：固件 `convert_to_imlib_image()` 把 `FMT_BGR888`
  直接当 `PIXFORMAT_RGB888`（源码 `// fall through`，不做通道交换），交 BGR 数据会被
  当 RGB 算 LAB → **红蓝互换**（蓝色目标被框成 C1、坑位定序连带全错，2026-09-26 台架实测）。
  `_downscale` 用 `cvtColor(BGR2RGB)+bgr=False`，`maix_accel._as_rgb888()` 兜底转换。
- 显示 HUD: `modules/zw_opencv_module/display/hud.py`（状态栏/检测叠加/锁存进度,
  不依赖 AI 模型加载）
- MaixPy 加速层: `detectors/eod_color/maix_accel.py`（设备走 C++ find_blobs, PC 自动回退
  cv2; 开关在 eod_config.yaml `accel:` 段）
- AI: `nn.YOLO11` 人质模型 `eod_hostage`（/root/models/eod_hostage_320.mud, 仓库副本
  `assets/models/hostage_yolo11n/`, 源自 人质yolo11n pulsar2 AX620E 转换链）,
  经 `MaixCam2AI` nick 热切换; 缺失时 RESCUE 回 NACK NOT_READY 不崩溃
- 阈值: `eod_config.yaml` `gks:` 段 + 运行时覆盖文件 `/my-threshold/my_64threshold.txt`
  （mtime 变化热重载不清锁存）; 设备端调参器 `python run.py debug color`（触屏 GUI）
- 测试（PC 端 pytest, 116 用例）: `python -m pytest tests/`
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
| `app/subscription_sm.py` | 订阅门控四态状态机（IDLE/EXPLOSIVE/TARGET/RESCUE, StateMachine 引擎承载） |
| `app/page_sm.py` + `app/page_router.py` | UI 四页状态机（MAIN/EXPLOSIVE/TARGET/RESCUE）+ KEY_OK 循环切页路由 |
| `modules/zw_opencv_module/debug/lab_session.py` | 六组 LAB 阈值共享会话（调参页按组拆分后的内存事实来源） |
| `modules/zw_opencv_module/debug/rescue_tuner.py` | 人质调试页（conf_th/iou_th/x_crop 滑条 + 模型状态） |

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
- **推流 = 结果驱动 + 心跳兜底**（2026-09-28）: 视觉每产出一帧结果立即推一帧，
  **不受订阅帧 min_interval_ms 限制**（"一帧误差一帧输出"，推流率≈相机帧率）；
  `min_interval_ms` 降级为"无新结果时的心跳周期"（下限 30ms）。闸门在
  `coordinator._push_allowed`，判据 `_result_seq != _sent_result_seq`。
  一个主循环迭代内到达多帧结果时只推**最新**那帧（不排队、不补推陈旧）。
  退路: `project_config.yaml` `gks_stream.push_on_new_result: false` 回到纯 interval
  限速；`min_push_interval_ms` 可压制突发连推。验收看日志 `[UART TX] DATA_STREAM
  <Hz>Hz` 与 `[FPS] pipeline` 是否一致（DebugConsole 的 `TX` 行同义）。
- ⚠ **相机停帧契约**（2026-09-28 修正）: `maixcam2` 的 `read_raw()` 曾在此前"空队列
  回退 `_last_frame`"，使 `pipeline_camera` 的"停帧不复用旧帧"在设备上失效 ——
  视觉线程把同一帧反复当新帧算、结果时间戳持续刷新，1s 陈旧归零永不触发，
  相机挂死时冻结结果会以 found=1 持续喂主站。现已在 `_ingest_raw` 处改为返回
  None（与 `read()`/mock/linux 及 `interface/camera.py` 契约一致）。
- KEY_OK **顺序循环**切四页: MAIN → EXPLOSIVE → TARGET → RESCUE → MAIN（释放沿,
  0.5s 防抖; 进调试页需相机存在, 调试页独占相机 set_paused）。
- 调试页 UI: 爆/靶页各自 3 组 LAB 滑条+点取色, 靶页画 ROI 横线, 爆/靶页有
  MIRROR 开关（pit_left_is_1 运行时切换, 不落盘）与锁存进度; 人质页 conf/iou/
  x_crop 滑条+模型状态; Exit/OK 均回 MAIN。
- HUD 引导叠加: TARGET 模式画 ROI 两横线+水平中线±误差刻度; 告警横幅
  （阈值缺失/模型未加载/相机停摆/NACK 5s TTL）数据源 coordinator.get_info()。
- 运行时参数: GksTaskProcessor.set_runtime_param(section, key, value) 设备端
  立即生效不落盘; LAB 阈值仍走 /my-threshold 热重载。
- **高频性能日志闸门**（2026-10-09）: `project_config.yaml` `perf_log:` 段,
  新写法 `{enabled: false, names: [...]}`（旧布尔 `true/false` 仍兼容）。闸门在
  `utils/log_util.log_print` **拼好 msg 后、入队前**短路（连 queue 都不碰），
  默认组 = `[PERF] vision/mainloop/lcd show` + `[UART TX] DATA_STREAM`/`CMD_ACK`
  + `[WDT] ` + `[MEM] `。**只少打印、不改行为**（推流帧照发、计数器照累加、
  log_print 仍非阻塞），副作用是被吞行不进 logs/debug.log。
  ⚠ 事件类**刻意不在默认组**、保留可见: `[UART ACTION]` 门控/锁存、`[UART RX]`
  收帧、`[UART TX] CMD_NACK`、`[FPS] pipeline` —— 关了就看不出"订没订上/该不该
  NACK"。要全静音就往 names 追加 `[UART ACTION]`/`[UART RX]`；定位"时间花在哪"
  时把 `enabled` 置 true。应用点: `app/main.py::_apply_perf_log_config`（启动早期）。
  回归测试 `tests/test_perf_log_gate.py`。
- 遗留代码已归档 `archive/legacy/`（line_follow_sm、cargo/circle/ring/line 检测链、
  eod_shape、visual_state_machine、legacy handlers 等），整体退出 app.yaml 打包
  （exclude: archive）；`run.py debug` 仅保留 color 调参路由。
- `eod_config.yaml` 只剩 `gks:` 段（旧 `eod:` 段随遗留管线归档删除）。
- 台架部署/验证清单: `docs/competition/gks_bench_checklist.md`。
