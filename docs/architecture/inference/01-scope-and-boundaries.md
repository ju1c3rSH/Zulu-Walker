> 本节目标：固定推理层的职责边界、依赖边界、目标目录结构与"明确不做"清单，作为后续所有章节的边界前提。

# 01 范围与边界

来源：架构文档 §1 目标与边界、§2 目录结构、§14 明确不做。整体状态：`[已定]`。

## §1 目标与边界

状态：`[已定]`

- 推理层必须解决"引擎实现被焊死在平台目录下"这一问题。
- 待改造现状：`framework/hal/platforms/maixcam2/ai.py` 直接 import `maix.nn`，换引擎等于换平台。
- 推理层必须把该问题拆成三条互相独立的轴。

下表固定三条互相独立的轴及其归属。

| 轴 | 含义 | 归属 |
|---|---|---|
| 平台 platform | 代码在哪跑、用哪些硬件 API | `framework/hal/platforms/<p>/` |
| 引擎 engine | 谁执行计算图 | `framework/inference/engines/<e>.py` |
| 工件 artifact | 具体加载哪张图 | manifest 内按引擎分组 |

- 平台不等于引擎：`maixcam2` 上可以跑 `ncnn`（性能降级），`linux` 上也可以跑 `maix_nn`（能力声明会拦住）。
- 框架不得替使用者做性能判断。
- 框架只负责把"可选项"和"不兼容项"讲清楚，并在不兼容时显式失败。

### §1.1 依赖边界（硬约束）

状态：`[已定]`

1. `framework/hal/interface/ai.py` 必须对上层暴露稳定契约；业务层不接触引擎细节。
2. `framework/inference/` 下只有 `engines/` 允许 import 第三方推理库或平台库（`maix`、`ncnn`、`onnxruntime`、`rknn_lite`、`torch`、`espdl`、`llama_cpp`）；`cv2` 不属于此类，规则见第 3 条。
3. 连带约束：`ops.py` 只能用跨平台通用计算库（`numpy`、`cv2`），不得 import 平台库（`maix`）或引擎库（`ncnn`、`onnxruntime`、`rknn_lite`）。
   理由：边界按**库的性质**划，不按目录划——`cv2` 不含任何平台知识，把它挡在门外只会换来一个数量级的算子开销（实测见 §10.2（06-performance-and-equivalence.md））。
4. 模型转换工具链（`rknn-toolkit2`、Pulsar2）必须是离线步骤，不进运行时框架，也不进仓库。

### §1.2 隔离脚本要同步扩展

状态：`[待实现]`

- 现役 `scripts/check_platform_isolation.py` 只扫 `framework/`，FORBIDDEN 是 `maix|cv2|serial`，EXEMPT 只有 `framework/hal/platforms`。
- 加上引擎后，隔离规则必须变为：

```python
# 平台库：hal/platforms 与 inference/engines 之外禁止
PLATFORM_LIBS = re.compile(r"^\s*(?:import|from)\s+(maix|serial)\b")
# 引擎库：推理内核禁止，engines/ 目录豁免
ENGINE_LIBS = re.compile(
    r"^\s*(?:import|from)\s+(ncnn|onnxruntime|rknn_lite|rknn|torch|espdl|llama_cpp)\b"
)
SCAN_DIR = ROOT / "framework"
EXEMPT = [
    ROOT / "framework" / "hal" / "platforms",
    ROOT / "framework" / "inference" / "engines",
]
```

- `cv2` 必须从全框架禁用清单里移出，改为按目录判定：`framework/` 其余部分仍禁 `cv2`，`framework/inference/ops.py`、`framework/inference/engines/` 与 `framework/hal/platforms/` 允许。
  理由：`cv2` 不含任何平台知识；引擎适配器有时需要一次像素搬运，把它挡在门外只会逼出更脏的绕法。
  理由：`cv2` 不含平台知识，留在清单里会让 letterbox 与 NMS 慢一个数量级（实测见 §10.2（06-performance-and-equivalence.md））。
- `rknn_lite` 必须单独列出。
  理由：正则里的 `rknn\b` 匹配不到 `import rknn_lite`，因为下划线不构成词边界。
- EXEMPT 必须从单个目录改成目录列表，遍历时按"是否在任一豁免目录之下"判断。
- `inference/engines/` 对两类库都豁免：`maix.nn` 是 MaixPy 绑定加速器的平台库，引擎适配器是唯一该碰它的地方（§1.1 规则 2）。
  理由：把平台库挡在 `engines/` 之外，会逼着适配器用 `importlib` 绕规则，边界反而更难审。

## §2 目录结构

状态：`[待实现]`

推理内核必须落在以下目录结构中。

```text
framework/
├── hal/
│   ├── interface/
│   │   └── ai.py                    # AIInference 契约（对上层稳定）
│   └── platforms/
│       ├── maixcam2/                # 相机 / 显示 / UART / 电量 + 平台能力声明
│       │   └── capabilities.py      #   available_engines = ("maix_nn", "ncnn", ...)
│       ├── maixcam1/
│       └── linux/
└── inference/                       # 平台无关、任务无关的推理内核
    ├── contract.py                  # 引擎协议 + TensorSpec + Capabilities + ArtifactRef
    ├── manifest.py                  # manifest 解析、校验、版本分支
    ├── runtime.py                   # ModelRuntime：engine + artifact + wrapper
    ├── registry.py                  # 引擎可用性探测与选择策略（不认识引擎实现）
    ├── facade.py                    # AIFacade：实现 AIInference 基础协议
    ├── ops.py                       # 通用算子（numpy + cv2）
    ├── wrappers/
    │   ├── __init__.py              # BUILTIN 注册表（名字 -> 元数据）；也接受 "module:attr"
    │   └── vision_det.py            # 内置实现：YOLO 检测解码（raw / end2end）
    └── engines/                     # 唯一允许 import 第三方推理库的目录
        ├── __init__.py              # REGISTRY: engine_id -> 引擎工厂（唯一登记处）
        ├── maix_nn.py
        ├── ncnn.py
        ├── onnxruntime.py
        └── rknn_lite.py
```

- 两处注册表职责不同，不得混用：`framework/inference/engines/__init__.py` 的 `REGISTRY` 是 engine_id 到引擎工厂的唯一登记处；`framework/inference/registry.py` 只做可用性探测与选择策略，不认识引擎实现细节。
- `framework/inference/facade.py` 的 `AIFacade` 是推理层的门面，实现 `AIInference` 基础协议；`framework/hal/platforms/<p>/ai.py` 降级为平台侧薄适配，只负责平台帧形态与错误状态，转调 facade。

### §2.1 构建工具链的位置

状态：`[已定]`

- 模型构建（RKNN 的 `rknn-toolkit2`、Pulsar2 的 Docker）必须是离线步骤，发生在开发机或 CI 上。
- 框架对构建的唯一责任：把配方记在 manifest 里，并在加载期校验工件与配方一致。
- 离线构建脚本必须落在：

```text
scripts/
└── build_models/                   # 可选：离线构建脚本，读取 manifest 的 build 段
    ├── build_rknn.py                #   onnx -> .rknn
    └── build_mud.py                 #   onnx -> .mud
```

## §14 明确不做

状态：`[已定]`

推理层不得出现以下行为。

- 不得做多引擎对同一帧的推理仲裁。
- 不得做在线模型热更新。
- 不得在框架层出现任务语义（边界见 §3.4 与 §3.5，见 §3.4 与 §3.5（02-contracts-and-dataflow.md））。
- 不得把第三方引擎源码或权重放进仓库。
- 不得在设备上构建模型（构建都是离线步骤，设备只加载工件）。
- 不得在推理层重新实现 NMS / letterbox 的第二份；通用算子只有 `ops.py` 一份。
- 不得为三种分辨率策略（letterbox / crop / stretch）各写一份实现；它们必须共用 `ops.py` 的同一套原语。
- 不得让核心层认识任务语义（核心层不出现"框""图像""token"这类词）。
- 不得预写暂时用不到的方言实现（内置只有 `vision_det` 一份）。
