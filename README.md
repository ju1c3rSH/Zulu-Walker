# Zulu-Walker 考核骨架

2026 级新生视觉考核用框架骨架。基于比赛仓库裁剪而来，仅保留框架与调试基建，
检测算法由考核者自行实现。

## 获取

```bash
git clone <仓库地址>
cd Zulu-Walker
python -m pip install -r requirements.txt
python -m unittest discover -s tests      # 应全绿
python run.py debug circle                # 打开教学样板调试窗
```

## 目录说明

```
framework/                       框架层: HAL(平台钩子) + Slot/CmdQueue/EventBus 三原语
modules/zw_opencv_module/
  processors/base.py             Processor 基类与 VisionResult 契约
  processors/registry.py         @register_processor 注册表
  processors/circle_sample_processor.py   教学样板(最小实现, 照此结构写自己的检测器)
  detectors/circle_target_detector/       参考检测器(完整工程化写法)
  detectors/_shared/             调试窗基建(trackbar + YAML 参数持久化)
  vision_manager.py / pipeline_camera.py  视觉管线
utils/                           日志 / TUI调试台 / 几何工具
tests/                           单元测试(验收门禁之一)
scripts/check_platform_isolation.py      fitness: 框架层禁 import 硬件库
```

## 考核规则

1. 基础题: 实现 `rect_processor` / `circle_processor`, 注册进 registry, 通过 golden 图比对;
2. 框架公共文件(`base.py` / `registry.py` / `framework/` 全目录)默认不改;
   确有必要修改须附设计说明, 且全部既有测试保持通过;
3. 参数一律走 YAML(`detectors/_shared/param_def.py`), 禁止硬编码;
4. 每个检测器至少 1 个 PC 可跑的单元测试(可程序合成测试图);
5. `feat/<任务名>` 分支开发, commit 用 `feat:` / `fix:` / `test:` 前缀。
