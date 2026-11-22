# MaixCAM1 (MaixCAM / MaixCAM-Pro) 平台支持

本文记录 MaixCAM1 (cv181x) 平台支持在 `main` 分支上的落地方式、移植来源、
与 MaixCAM2 的差异，以及部署与验收要点。

## 1. 背景与来源

MaixCAM1 支持最早在分支 `feat/maixcam1-port`（提交 `871792d`）引入，
其后在 `EOD2026-NearCam` 分支持续演进。**该分支是从 `main` 分叉出去的旧
HAL 架构**（缺少后来加入的 `sink.py` / `sysinfo.py` / `watchdog.py`，且
另有一整套 `framework/display/` 展示架构），因此无法直接合并或 cherry-pick，
本 PR 是**逐文件适配移植**的结果。

来源分支（均未推送远端）:

| 分支 | 角色 |
|------|------|
| `feat/maixcam1-port` | MaixCAM1 HAL 平台首发 |
| `EOD2026-NearCam` | CAM1 支持的最新形态（含电池采集链路） |
| `feat/battery-hal` | 电量 HAL 初版（`526d7a0`），已被 NearCam 的 `9c87dd1` 取代 |

> 注意: `feat/battery-hal` 的分支名有误导性 —— 它的 tip 停在 11-03 的初版，
> 反而比 NearCam 里 11-04 的版本少一代（少了 `reason`/`charging` 上报）。

## 2. 移植清单

### 新增

| 文件 | 说明 |
|------|------|
| `framework/hal/platforms/maixcam1/__init__.py` | 平台工厂（相机/显示/UART/AI/电量） |
| `framework/hal/platforms/maixcam1/camera.py` | 四级降级构造、非阻塞读、停帧保护 |
| `framework/hal/platforms/maixcam1/display.py` | 显示对象，按固件能力逐级回退构造 |
| `framework/hal/platforms/maixcam1/ai.py` | CVITEK `.cvimodel` + `nn.YOLOv5` 链路 |
| `framework/hal/platforms/maixcam1/uart.py` | UART0 (`/dev/ttyS0`) |
| `framework/hal/platforms/maixcam1/battery.py` | AXP2101 电量（`maix.ext_dev.pmu`） |
| `framework/hal/platforms/lazy_frame.py` | 延迟转换代理，省掉每帧 921KB 拷贝 |
| `framework/hal/interface/battery.py` | 电量 HAL 接口 |
| `framework/hal/battery_sampler.py` | 电量采样去抖 + 快照发布 |
| `framework/hal/platforms/maixcam2/battery.py` | CAM2 sysfs 电量计 |
| `framework/hal/sink.py` | `MaixLcdSink` 迁到平台中立层 |
| `configs/project_config.maixcam1.yaml` | CAM1 运行配置模板 |
| `docs/architecture/maixcam1_platform.md` | 本文 |

### 修改

| 文件 | 改动 |
|------|------|
| `framework/hal/interface/__init__.py` | 导出 `Battery` / `BatteryReading` / `NullBattery` |
| `framework/hal/interface/camera.py` | Protocol 补齐 `read_raw()` / `last_raw` |
| `framework/hal/machine.py` | 注入 `battery`，按能力探测 `create_battery` |
| `framework/hal/platforms/maixcam2/__init__.py` | 增加 `create_battery` |
| `framework/hal/platforms/linux/__init__.py` | 增加 `create_battery`（`NullBattery`） |
| `framework/hal/platforms/mock/__init__.py` | 增加 `create_battery`（`NullBattery`） |
| `framework/hal/platforms/linux/camera.py` | 增加 `read_raw()` / `last_raw` |
| `framework/hal/platforms/mock/camera.py` | 增加 `read_raw()` / `last_raw` |
| `framework/hal/platforms/maixcam2/sink.py` | 改为对 `framework.hal.sink` 的再导出（兼容壳） |
| `app/main.py` | sink 导入改中立路径；看门狗改为按平台探测；支持 `ZWULU_CONFIG` |

## 3. 适配点（与 NearCam 的差异）

移植不是复制，以下几处必须改，否则在 `main` 上跑不起来:

1. **`framework.display.types` 不存在**。NearCam 的 `maixcam1/display.py` 依赖
   `DisplayCaps` / `PixelFormat`，那是 NearCam 另一套展示架构的东西。
   `main` 的 `Display` 协议只有 `show()` / `width()` / `height()` / `close()`，
   本 PR 按 `main` 的形状裁剪，删掉了 `caps()`。

2. **`MaixLcdSink` 必须离开 maixcam2 平台目录**。它其实只碰 `display.show()`，
   本身与平台无关。但它此前住在 `framework/hal/platforms/maixcam2/sink.py`，
   而 `framework/hal/platforms/maixcam2/__init__.py` 会连带导入 `.ai`，
   后者在模块级引用 `maix.nn.YOLO11` —— CAM1 固件没有这个类。
   于是 `import maixcam2` 在 CAM1 上抛的是 **AttributeError 而不是
   ImportError**，原来 `app/main.py` 里的 `except ImportError` 接不住，
   应用会直接崩。现在 sink 迁到 `framework/hal/sink.py`（平台中立），
   maixcam2 侧保留兼容壳再导出。

3. **相机工厂要吞掉曝光/增益/AEC**。`Machine.create` 会把
   `exposure_us` / `gain` / `aec` 透传给平台工厂，而 CAM1 的相机构造
   不认这三个参数（运行期改曝光要走 `set()`）。不显式 `pop` 掉就是
   `TypeError`，连相机都开不起来。

4. **看门狗按平台探测**。原来的 `from framework.hal.platforms.maixcam2 import
   create_watchdog` 在 CAM1 上每次启动白抛一次异常（CAM1 固件的 WDT 本身是
   桩实现，本来也用不上）。现在从 `cfg["platform"]` 解析平台模块再探测。

5. **`Camera` Protocol 补 `read_raw()` / `last_raw`**。`main` 的
   `pipeline_camera.process_frame()` 无条件调用 `camera.read_raw()`，但
   协议没声明它，linux/mock 平台也没实现。本 PR 一并补齐，契约与实现对齐。

## 4. CAM1 与 CAM2 的关键差异

| 项目 | MaixCAM2 (AX630C) | MaixCAM1 / Pro (cv181x) |
|------|--------------------|---------------------------|
| AI 框架 | `maix.nn.YOLO11` | `maix.nn.YOLOv5` |
| 模型格式 | `.axmodel` | `.cvimodel`（CVITEK） |
| 串口 | `maix.peripheral.uart` / `/dev/ttyS4` | `maix.uart` / `/dev/ttyS0`（A16/A17） |
| 相机构造 | 全关键字 | 四级降级（全关键字 → 去 open → 位置参数 → 最小签名） |
| 取图延时 | 较低 | **最低 30+ms**（固件内部双缓存） |
| 镜头 | 固定焦距 | 默认 M12 手动调焦，标定几何不能沿用 CAM2 |
| 补光灯 | B25 GPIO | 无 |
| 看门狗 | 硬件 WDT | 固件桩实现，不可用 |
| WiFi 推流 | 支持 | `zw_wifi_stream` 声明 `PLATFORMS=("maixcam2",)`，CAM1 上被跳过 |

## 5. 部署

### 模型文件

模型二进制**不入库**（7.2MB `.cvimodel`），需自行部署：

```bash
# 设备上
mkdir -p /root/models
# 把 model_239147.cvimodel 与 model_239147.mud 一起放进 /root/models/
# mud 里的 `model =` 必须指向同目录的 cvimodel，且两者同名
```

CAM1 的 `.cvimodel` 与 CAM2 的 `.axmodel` **不通用**，不能互相替换。

### 接线

- UART: A16(TX) / A17(RX) → Brain USART1。这两根出厂默认就是 UART0 功能，
  平台层会跳过 pinmap。
- **TX(A16) 是启动模式检测引脚**，被拉低会导致无法开机。电平转换电路务必
  确保 TX 默认不拉低。
- UART0 同时是系统默认 maix protocol 串口，占用后 MaixVision IDE 调试通道
  会抢线 —— 台架调试改用 SSH（`ssh root@<设备IP>`）。

### 启动

```bash
set ZWULU_CONFIG=configs/project_config.maixcam1.yaml
python run.py main
```

`ZWULU_CONFIG` 不设时仍读工作区的 `project_config.yaml`（默认 MaixCAM2），
行为不变。

## 6. 已知限制

- **VENC 是单例硬件资源**：CAM1 的 `Display.show()` 内部走 MaixVision 编码，
  与推流抢同一份编码器。开推流前先关 MaixVision 预览。
- **电量读数只上日志，不上 HUD**：采集模块（`modules/zw_battery_module`）在主循环
  慢节拍里采样，每个快照打一行到控制台与 `logs/debug.log`；HUD 上还没有电量位置，
  需要另一层显示接线。
- **相机标定几何需重标**：CAM1 是手动调焦镜头，坑位 ROI、误差中心等
  不能沿用 CAM2 的值。
- **`nn.YOLOv8` / `nn.YOLO11` 在 CAM1 固件上不存在**，因此 CAM1 的 AI 工厂
  对缺失的模型类直接报错，不做静默回退 —— 用 YOLOv5 的后处理去解 YOLOv11
  的输出不会抛异常，只会吐错误坐标。

## 7. 验收

```bash
python -c "from framework.hal import Machine; print('OK')"
python -c "from framework.hal.platforms.maixcam1 import create_camera, create_ai; print('OK')"
python -c "from framework.hal.interface import Battery, BatteryReading, NullBattery; print('OK')"
python -c "from framework.hal.sink import MaixLcdSink; print('OK')"
```

真机验收重点:

1. 相机构造日志出现 `[CAM1] camera open level=1`（而非 level=3/4 降级），
   否则 `fps`/`buff_num` 已被静默丢弃。
2. 启动日志**没有** `display.attempts > 1`。
3. 启动日志**没有** `camera.read(block=False) 不被本固件接受` —— 出现即表示
   已降级为阻塞读，停帧保护失效。
4. UART 能收自发心跳帧。
