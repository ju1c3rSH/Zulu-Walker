# 模型契约目录

这里放的是逻辑模型的声明，不是二进制工件。

| 文件 | 说明 |
|---|---|
| `steelball_1280x352/manifest.yaml` | 一个逻辑模型 + 三个引擎工件的契约 |
| `steelball_1280x352/labels.txt` | 标签表，一行一个 |

## 标签与引擎无关

`labels_file` 挂在逻辑模型上，三个引擎共用同一张表。onnx 与 ncnn 的工件本身不带标签，
没有这张表就无法解码：raw 方言要从输出张量里切出类别分数，类别数就是 `4 + len(labels)`。
加载期还会校验 `labels.txt` 行数与模型输出一致。

mud 自带 labels，maix.nn 能自述；框架仍以 `labels.txt` 为准，装载期与 mud 的 labels 比对，
不一致直接报错（Phase 1 落地）。

## 部署

manifest 与各引擎工件同目录，一并拷到设备：

```
/root/models/steelball_1280x352/
  manifest.yaml
  labels.txt
  steelball_1280x352.mud
  steelball_1280x352.onnx
  steelball_1280x352.param
  steelball_1280x352.bin
```

二进制工件不进仓库（体积大、平台相关），只在 manifest 的 `build` 段记配方。

运行期只有一个引擎，配置里写死：

```yaml
ai:
  engine: "maix_nn"           # 必填，不做候选链，不做静默降级
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_1280x352/steelball_1280x352.mud"   # 精确文件名
      manifest: "/root/models/steelball_1280x352/manifest.yaml"         # 精确文件名
```

`model` 必须是**精确的工件文件**，不接受目录——目录形式把"哪个文件是哪个引擎的工件"
藏起来，猜错就是加载了错的权重而不报错。加载期会核对 manifest 里
`artifacts.<engine>.file` 与 `model` 是否同一个文件，对不上直接报错。

## 离线转换

```
训练权重 .pt
  -> export_onnx.py        导出 raw head 的 onnx（不含 NMS）
  -> ncnnoptimize / pnnx   onnx -> ncnn 的 .param + .bin
  -> Pulsar2 (Docker)      onnx -> AX630C 的 .mud
```

三个产物必须来自同一份 onnx，否则等价性验证没有意义。

## manifest 字段速查

```yaml
schema_version: 3
name: steelball_1280x352     # 逻辑模型名
family: yolo11              # 模型族，逻辑模型级；与工件自述交叉校验

input:
  frame: { color: rgb, dtype: uint8, layout: hwc }   # 模型期望的帧形态
  size: [1280, 352]              # 目标分辨率
  resize: { mode: letterbox, value: 114 }            # 分辨率策略
  norm: { scale: ..., mean: [...], std: [...] }       # yolo 形式的归一化参数

artifacts:
  <engine>:
    wrapper: "vision.det.raw"   # 不透明键，不承载语义
    preprocess: { resize: framework, norm: engine, formula: substract_mean_normalize }
    io: { inputs: { image: "data" }, outputs: { pred: "output" } }
```

**帧形态**（`input.frame`）声明模型要什么帧，与相机实际给的帧不一致时由前处理转换，不是报错。

**io 映射**（`artifacts.<engine>.io`）把逻辑名映射到引擎的物理名，两张表都不能为空。
逻辑输入名固定是 `image`；引擎没有物理名时（maix.nn）逻辑名与物理名相同，
例如 `inputs: { image: "image" }` 与 `outputs: { objects: "objects" }`。
主输出是 `outputs` 的第一个键，wrapper 按这个键取结果，不写死名字。

**分辨率策略**（`input.resize.mode`）三选一：

| mode | 做法 |
|---|---|
| `letterbox` | 等比缩放使整帧可见，不足处用 `value` 补齐（缺省 114） |
| `crop` | 等比缩放使画面覆盖目标，超出部分按 `anchor` 裁掉（默认 `center`） |
| `stretch` | 直接拉伸到目标尺寸，不保宽高比 |

**归一化**（`preprocess.norm`）三档，归一化公式由 `preprocess.formula` 显式声明：

| norm | 谁执行 | formula |
|---|---|---|
| `framework` | 框架规范段 | `yolo` |
| `engine` | 引擎适配器 | `substract_mean_normalize` |
| `baked` | 已烘焙进工件 | 不写 |

## 加载期会校验的东西

manifest 自身的结构一致性由 `framework/inference/manifest.py` 在解析期校验，失败抛
`E_CONTRACT_MISSING`；离线由 `scripts/check_manifest.py` 校验同一份 schema。

下列检查属于运行期加载期，Phase 1 随协议拆分落地，当前尚未实现：

- `family` 在该 wrapper 接受的族内（否则 `E_MODEL_TYPE_UNSUPPORTED`）
- `input.frame` / `input.size` 与工件自述一致
- `io` 里的物理名在引擎里存在（ncnn / ort 能自述，rknn 只能按索引）
- `labels.txt` 行数与工件输出一致
- `build` 段的 `input_size` / `color` 与逻辑模型一致
- `runtime.version` 必须钉住并与板端比对
- ncnn 的 blob 名对不上直接报错，不静默用第一个输出
