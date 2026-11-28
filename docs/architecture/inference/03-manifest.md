> 本节目标：固定 manifest schema v3 的字段定义、校验规则与存储布局，使 manifest 成为运行期契约与离线转换配方的唯一来源。

# 03 Manifest

来源：架构文档 §5 Manifest（`schema_version: 3`）。整体状态：`[已定]`。

## §5 Manifest（`schema_version: 3`）

状态：`[已定]`

### §5.1 为什么需要 manifest

状态：`[已定]`

各生态都有自描述，但描述的是图，不是任务。下表固定各生态自带信息与查不到的信息。

| 生态 | 工件自带什么 | 查不到什么 |
|---|---|---|
| ONNX | 完整图 + 输入输出名 / shape / dtype / opset | 后处理语义、letterbox 参数、labels |
| ncnn | `.param` 逐层文本图 | IO blob 名约定、letterbox、labels、量化表 |
| RKNN | 二进制；运行期可查输入属性 | 构建期参数基本查不回：target_platform、量化校准集、mean/std、输入尺寸、toolkit 版本 |
| MaixPy MUD | ini `[basic]` / `[extra]` | 单后端；mean / scale 是影子声明 |

- manifest 存在的三个前提必须同时成立，缺一不可：
  1. 同一个逻辑模型要在多个引擎上跑。
  2. 运行期契约不在工件里（RKNN 尤其如此）。
  3. 需要一处同时写下"这个模型是什么"与"它在每个引擎上长什么样"。
- 运行期只有一个引擎（§7.2，见 §7.2（04-runtime-and-engine-selection.md）），但前两条前提依然成立。
- manifest 必须是**运行期契约**的唯一来源（labels、io、preprocess、wrapper、family 全在里面）。
- manifest 同时必须是**离线转换的配方记录**：同一份 onnx 转成 mud / param+bin / rknn 各自的命令与参数都在 build 段，工件本身查不回来。
- manifest 不得因为"只跑一个引擎"就退化成常量；它同时承担契约与配方两件事。

### §5.2 完整 schema

状态：`[已定]`

- 仓库里的实例必须落在 `deploy/maixcam/models/steelball_1280x352/manifest.yaml`（maix_nn / onnxruntime / ncnn 三个工件）。
- 一个逻辑模型对应一个目录，包含 `manifest.yaml` 与各引擎工件。

```yaml
schema_version: 3
name: yolo11n
family: yolo11                # 模型族，逻辑模型级；与工件自述交叉校验

input:
  frame: { color: rgb, dtype: uint8, layout: hwc }   # 模型期望的帧形态
  size: [640, 352]              # 目标分辨率
  resize: { mode: letterbox, value: 114 }
  norm: { scale: 0.00392156862745098, mean: [0, 0, 0], std: [1, 1, 1] }

labels: [class_a, class_b, class_c]

artifacts:
  maix_nn:
    file: yolo11n.mud
    wrapper: "vision.det.end2end"
    wrapper_params: {}
    preprocess: { resize: baked, norm: baked }
    io:
      inputs: { image: "image" }
      outputs: { objects: "objects" }

  onnxruntime:
    file: yolo11n.onnx
    wrapper: "vision.det.raw"
    wrapper_params: {}
    preprocess: { resize: framework, norm: framework, formula: yolo }
    io:
      inputs: { image: "images" }
      outputs: { pred: "output0" }

  ncnn:
    file: yolo11n.param
    bin: yolo11n.bin
    wrapper: "vision.det.raw"
    wrapper_params: {}
    preprocess: { resize: framework, norm: engine, formula: substract_mean_normalize }
    io:
      inputs: { image: "data" }
      outputs: { pred: "output" }

  rknn_lite:
    file: yolo11n.rknn
    wrapper: "vision.det.raw"
    wrapper_params: {}
    preprocess: { resize: framework, norm: baked }
    io:
      inputs: { image: 0 }
      outputs: { pred: 0 }
    runtime: { package: "rknn-toolkit-lite2", version: "2.3.0" }
    build:
      tool: rknn-toolkit2
      version: "2.3.0"
      source: yolo11n.onnx
      target_platform: rk3588
      quant: int8
      calibration: datasets/train.txt
      input_size: [640, 352]
      layout: nhwc
      color: rgb
      mean_values: [0, 0, 0]
      std_values: [255, 255, 255]
```

注意三点：

1. `dialect`（方言）必须挂在工件上，不得挂在逻辑模型上。`.mud` 把后处理烘焙进图（`end2end`），`.onnx` 是裸输出（`raw`）。
2. `family` 必须挂在逻辑模型上；`wrapper` 只是不透明的注册表键——名字里不解析任何东西（§3.4，见 §3.4（02-contracts-and-dataflow.md））。
3. `preprocess.formula` 必须挂在工件上，跟着归一化的执行方走：`framework` 档用 `yolo`，`engine` 档用引擎式，`baked` 档不写（§4.3，见 §4.3（02-contracts-and-dataflow.md））。

注意：`dialect`（方言）挂在工件上，不挂在逻辑模型上。`.mud` 把后处理烘焙进图（`end2end`），`.onnx` 是裸输出（`raw`）。

### §5.3 字段表

状态：`[已定]`

下表固定 schema v3 每个字段的必填性与校验方式。

| 字段 | 必填 | 校验 |
|---|---|---|
| `schema_version` | 是 | 按版本分支；未知版本抛 `E_CONTRACT_MISSING` |
| `name` | 是 | 逻辑模型名 |
| `family` | 视觉任务必填 | 必须在该工件的 wrapper 条目 `families` 内，否则 `E_MODEL_TYPE_UNSUPPORTED` |
| `input.frame` | 视觉任务必填 | `color` / `dtype` / `layout` 枚举；与工件自述一致 |
| `input.size` | 视觉任务必填 | 与工件自述一致 |
| `input.resize.mode` / `anchor` / `value` | 视觉任务必填 | `mode` 枚举 `letterbox` / `crop` / `stretch`；`anchor` 仅 `crop` 有效 |
| `input.norm` | 视觉任务必填 | `scale` / `mean` / `std` 齐备且形状一致 |
| `labels` / `labels_file` | 视觉任务必填 | 行数与模型输出一致；raw 方言靠它算出类别数 |
| `artifacts.<engine>.file` | 是 | 文件存在且可读 |
| `artifacts.<engine>.wrapper` | 是 | 能解析到可调用对象，否则 `E_DIALECT_UNSUPPORTED` |
| `artifacts.<engine>.wrapper_params` | 条件 | 由 wrapper 自己校验，框架不解释其内部语义 |
| `artifacts.<engine>.preprocess` | 视觉任务必填 | `resize` / `norm` 枚举校验 |
| `artifacts.<engine>.preprocess.formula` | `norm` 非 `baked` 时必填 | 枚举 `yolo` / `substract_mean_normalize`；`baked` 时禁止出现 |
| `artifacts.<engine>.io` | 是 | 逻辑名与物理名的映射必须存在且非空；物理名必须被引擎自述（positional 只校验个数）；引擎无物理名时逻辑名与物理名相同 |
| `artifacts.<engine>.runtime` | 条件 | 版本比对，不匹配抛 `E_MODEL_LOAD_FAILED` |
| `artifacts.<engine>.build` | 条件 | 与 `input.size` / `color` 一致 |

- 框架只校验"存在性 + 类型"，不校验任务语义。
- 参数对不对由 wrapper 抛错。
  理由：wrapper 最清楚自己的参数含义。

### §5.4 存储布局

状态：`[已定]`

- 一个逻辑模型一个目录，manifest 与各引擎工件同目录：

```text
/root/models/steelball_1280x352/
  manifest.yaml
  labels.txt                 # 可选
  steelball_1280x352.mud
  steelball_1280x352.onnx
  steelball_1280x352.param  steelball_1280x352.bin
  steelball_1280x352.rknn
```

- **配置里必须写精确的文件名，不得接受目录。**
- `ai.models[].model` 必须指向引擎要加载的那个工件文件本身；`ai.models[].manifest` 必须指向 manifest 文件：

```yaml
model: "/root/models/steelball_1280x352/steelball_1280x352.mud"
manifest: "/root/models/steelball_1280x352/manifest.yaml"
```

- 不得提供"目录 → 自动找 manifest.yaml"的兜底。
  理由：目录形式把"哪个文件是哪个引擎的工件"这件事藏起来，猜错就是加载了一个错的权重而不报错。
- 不得提供"只有工件、靠 `model_type` 猜方言"的单文件模式。
  理由：单文件模式让方言靠推断，推断不出时只能抛错；两者都是把明确的契约换成隐式约定。
- 加载期校验：manifest 里该引擎的 `artifacts.<engine>.file` 必须与 `model` 指向同一个文件（比对规范化后的路径，或至少文件名），不一致抛 `E_CONTRACT_MISSING`。
