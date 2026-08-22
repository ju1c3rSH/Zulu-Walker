# Zulu Framework Guide

> 面向"从这套骨架起新项目"的工程向导。完整比赛级示例(主从 UART 桥、两阶段标定、
> WiFi 图传、录制信令)见 `release/2026H` 分支;本仓库 main 的 `app/` 只保留最小骨架。

## 1. 它是什么

MaixCAM2 与通用 Linux 双平台视觉框架:

```
run.py                 双平台入口: run.py main(跑骨架) / run.py debug <name>(detector 体验)
app/                   你的项目层 —— 仅三个文件:
  main.py              骨架:启动铁律的最小装配(WDT→Machine→ModuleManager→泵)
  coordinator.py       ★ 各项目分支的编排"脏代码区",整个项目的集成成本=替换此文件
  __init__.py
framework/             平台无关框架(禁止 import maix/cv2/serial,fitness 强制)
  slot.py / cmd_queue.py / event_bus.py   调度三原语
  ui_state.py          纯逻辑 UI 几何/去抖
  module_manager.py    装配器 + PLATFORMS 门控
  hal/interface/       协议: Camera/Display/Uart/AI/Sink/InputSource/Watchdog/SysInfo
  hal/platforms/{maixcam2,linux,mock}/   平台实现 + create_* 能力钩子
modules/               可复用功能模块(含 detectors 方案库,默认休眠)
deploy/maixcam/        maixcam 设备打包模板(app.yaml + 固件入口壳)
docs/architecture/     thread_tick_topology.md(选型表)/ 本指南
```

## 2. 双平台运行矩阵

| 能力 | maixcam2 | linux | mock |
|---|---|---|---|
| Camera | 硬件 AEC/重连/stall 检测 | v4l + warmup 线程(`resolve_camera_source` 钩子) | 假相机 |
| Display | LCD(MaixLcdSink 泵) | cv2.imshow(LinuxDisplay) | 有 |
| Watchdog | 硬件 WDT(create_watchdog) | 无 → 安全缺省 None | 无 |
| SysInfo | CMM 内存池 | /proc/meminfo | None |
| exit_check | app.need_exit 注入 | Ctrl-C | Ctrl-C |

平台选择:`project_config.yaml` 的 `platform:` 字段;能力一律经 `machine.*`
按名探测的钩子获取,框架代码永不 import 平台模块。

## 3. 起一个新项目(五步)

1. 从本分支切出你的项目分支。
2. 写 `app/coordinator.py`:替换 AppCoordinator 的 no-op,接 EventBus 订阅、
   CmdQueue 消费、Slot 读取。选型规则见 `thread_tick_topology.md`。
3. 在骨架 `main()` 的 TODO 处补齐平台 bring-up(WiFi/beacon 等),每步之间喂狗。
4. `manager.register_many([...])` 加你需要的模块。
5. Windows 冒烟:`python -m unittest discover -s tests` +
   `python scripts/check_boot_scope.py` + `check_platform_isolation.py`;
   上板前在 linux/mock 平台先跑通 `python run.py main`。

## 4. 启用一个 detector 方案(三步)

方案默认休眠:不 import、零运行时开销、不进设备包。以 cargo 为例:

```python
# ① 项目代码里、流水线启动前:
from modules.zw_opencv_module.processors import enable_optional
enable_optional(["TrackCargoProcessor"])   # 未知名会 KeyError,拼错当场爆
```

② 设备打包:`deploy/maixcam/app.yaml` 的 files 增加
`modules\zw_opencv_module\detectors\cargo_detector` 目录与对应 processor 文件。

③ 本机体验/调参:`python run.py debug cargo`(trackbar 面板,不经注册表)。

可用清单见 `processors/__init__.py::OPTIONAL_PROCESSORS`;
`pendulum_calibrator` 是独立库(非 processor 体系),项目代码直接 import 使用。

## 5. 铁律速查(app/main.py 头部同款)

- WDT 先行,慢步骤之间喂狗
- Machine.create 之前的对象用 late-injection setter 拿平台能力
- 显示无专属线程;lambda 队列与跨线程直调禁止(Slot/CmdQueue/EventBus 三选一)
- 新增常驻线程需在 PR 里给出豁免理由(稳态预算 ≤4)

## 6. 打包部署(maixcam)

复制 `deploy/maixcam/{app.yaml, main.py}` 到仓库根 → 按 §4② 增删 files →
根目录执行 maixpack。固件入口壳会把控制权交给 `app.main.main()`。
