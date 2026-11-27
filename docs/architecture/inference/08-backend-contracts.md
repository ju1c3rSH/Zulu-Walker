> 本节目标：固定引擎适配器的统一内部形式、双路径与等价性要求、引擎差异矩阵、GIL 事实与接入验收标准，作为四个引擎适配器的共同上位约束。

# 08 后端接入契约

来源：后端文档 §0 分工、§1 统一内部形式、§2 双路径与等价性、§3 差异矩阵、§5 GIL 与线程、§6 接入验收标准。逐引擎细则见 `09-engine-maix-nn.md` … `12-engine-rknn-lite.md`。

## §0 本文与 inference_architecture.md 的分工

状态：`[已定]`

下表固定两份文档的职责分工。

| 文档 | 回答什么 |
|---|---|
| `inference_architecture.md` | 框架应该长成什么样（分层、协议、manifest、错误码、生命周期） |
| 本文 | 四个引擎到底差在哪，每个引擎要写哪几行适配，差别的部分怎么隔离 |

- 事实标注约定必须沿用：`[源码]` 来自源码 / 官方文档；`[实测]` 已实测；`[推断]` 合理推断但未验证；`[待测]` 需要上机验证。
- 本目录已把两份源文档拆分为 `01`–`13` 各文件，引用一律写成"见 §X.Y（文件名）"形式。

## §1 统一内部形式

状态：`[已定]`

### §1.1 规范形式（canonical）

状态：`[已定]`

下表固定规范形式（canonical）的输入与输出形态。

| | 形式 |
|---|---|
| 输入 | `numpy`，HWC（channels-last），已完成帧形态转换与分辨率处理；**数值域随 `preprocess.norm` owner 变**（见下表） |
| 输出 | `dict[str, Any]`，key 为逻辑名（见 §1.3 的输出形态、§1.4 的 io 映射） |

- 布局选 HWC 的理由见 §1.2：四个引擎里三个原生就是 channels-last，只有 onnxruntime 要 NCHW；选 NCHW 等于让三个引擎每帧各转置一次。
- 规范形式是**规范预处理**的中间形态。`engine.run()` 收到的是**物理适配之后**的张量：规范张量按引擎输入协议转成引擎要的 layout 与 dtype（§1.5）。
- 规范段的数值域由工件级 `preprocess.norm` 三档决定。

下表固定三档归一化下规范段的产出与适配器职责。

| `preprocess.norm` | 规范段产出 | 适配器职责 |
|---|---|---|
| `framework` | float32，已按 `input.norm` 归一化 | 转 layout 与 dtype |
| `engine` | uint8，像素域 HWC | 转 float32 + 引擎内归一化 |
| `baked` | uint8，像素域 HWC | 转 float32，**不得**预先除 255 |

- 公式语义由工件的 `preprocess.formula` 声明（`yolo` = `(x*scale - mean)/std`；`substract_mean_normalize` = `(x - mean)/std`），两档互不解释。
- 分辨率处理必须按 manifest 的 `input.resize.mode` 三选一（`letterbox` / `crop` / `stretch`），产出精确的 `input.size`。
- RKNN 转换期的 `std_values=[255,255,255]` 在转换期烘焙，运行期必须收原始像素。
  理由：若规范输入已除过 255，RKNN 会再除一次，数值差 255 倍且不报错——这正是 `baked` 档规范段保持 uint8 像素域的原因。

### §1.2 为什么是它

状态：`[已定]`

规范形式必须是四个引擎的唯一公共形式。下表固定各引擎的原生输入与到规范形式的代价。

| 引擎 | 原生输入 | 到规范形式的代价 |
|---|---|---|
| maix.nn | `maix.image.Image` / BGR numpy | 快路径直接吃 Image，零转换；规范路径才需转换 |
| ncnn | `ncnn.Mat`（HWC 像素，float32） | `Mat(numpy)` 零拷贝，只差一次 dtype 扩展 [源码] |
| onnxruntime | numpy，NCHW | 一次 HWC→NCHW 转置（唯一需要转置的引擎） |
| rknn-lite | numpy，NHWC，float32 | 单图 NHWC 就是 HWC 加 batch 维，只差 dtype 扩展 |

### §1.3 engine.run 的输出形态

状态：`[已定]`

下表固定两种输出形态的 value 内容、后处理责任方与适用引擎。

| 形态 | value 是什么 | 谁做后处理 | 本文的引擎 |
|---|---|---|---|
| raw | 原始张量 `np.ndarray`，形状按 `output_specs()` 声明 | wrapper 解码 + NMS | onnxruntime / ncnn / rknn-lite |
| end2end | 已解码的对象列表，引擎内置了后处理 | wrapper 只做格式转换 | maix.nn |

- 引擎不得解释输出内容。
- end2end 形态下 wrapper 不得重复做后处理。
  理由：对已带 NMS 的图再跑一次 NMS 会打乱框且不报错。

### §1.4 io 映射：逻辑名与物理名

状态：`[已定]`

- `engine.run()` 的 key 一律用逻辑名，物理名由引擎适配器按 `artifact.io` 翻译。

```python
# wrapper 侧，永远只写逻辑名
out = rt.engine.run({"image": blob})
pred = out["pred"]
```

```yaml
# manifest 侧
io:
  inputs:  { image: "data" }
  outputs: { pred: "output" }
```

- 输入固定用逻辑名 `image`。
- 输出用逻辑名映射；多输出时映射可以有多个条目，第一个键是主输出。
- 物理值可以是名字（`"data"`、`"output0"`）或索引（`0`、`1`）。
- 引擎协议里的 `input_specs()` / `output_specs()` 必须是复数（YOLOv5 与 RKNN 都有多输出），启动期与 manifest 的 io 声明交叉校验。

### §1.5 引擎适配器只做三件事

状态：`[已定]`

- 引擎适配器是引擎侧的一层：把命名字典搬运成引擎原生格式再取回来。
- 任务侧的翻译（准备输入、解释输出、暴露任务接口）由 wrapper 负责（见 §3.4（02-contracts-and-dataflow.md））。

```python
class InferenceEngine(Protocol):
    def load(self, artifact: ArtifactRef, **opts) -> None: ...
    def input_specs(self) -> list[TensorSpec]: ...
    def output_specs(self) -> list[TensorSpec]: ...
    def capabilities(self) -> Capabilities: ...
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]: ...
    def unload(self) -> None: ...
```

1. 物理适配：把规范张量（§1.1）转成引擎原生形式（布局、dtype、通道序）；`norm: engine` 档在这一步做引擎内归一化——这是前处理的第二段（§4.2，见 §4.2（02-contracts-and-dataflow.md））。
2. io 绑定：按 `artifact.io` 把逻辑名翻译成引擎的名字或索引。
3. 输出适配：引擎原生输出回到命名字典。

- 框架独占（只有一份实现，在 `ops.py`）：letterbox、归一化、NMS、IoU、softmax。
- `ops.py` 可用 `numpy` 与 `cv2`，不得 import 平台库或引擎库。
  理由：按库的性质划边界——`cv2` 不含任何平台知识；挡在门外会让 letterbox 慢 13 倍、NMS 慢 35 倍（实测见 §10.2（06-performance-and-equivalence.md））。
  隔离脚本的落地形态：`cv2` 移出全框架禁用清单，改为按目录判定——`framework/` 其余部分仍禁 `cv2`，`framework/inference/ops.py` 与 `framework/inference/engines/` 允许（见 §1.2（01-scope-and-boundaries.md））。
- 任务接口这一侧（解码、输入准备、暴露任务 API）归 wrapper；引擎适配器只做原生格式的搬运。
- 引擎协议必须保持任务无关：它不认识"图像""框""token"，只认识"一个命名字典进、一个命名字典出"。

## §2 双路径与等价性

状态：`[已定]`

- 每个引擎适配器可以提供两条路径。

下表固定双路径的输入形态与角色定位。

| 路径 | 输入 | 角色 |
|---|---|---|
| canonical | numpy，HWC，规范段输出（数值域随 `norm` owner） | 参考实现，正确性基准 |
| native 快路径 | 引擎原生帧 | 可选加速 |

下表固定各引擎快路径的现状与状态。

| 引擎 | 快路径 | 状态 |
|---|---|---|
| maix.nn | 直接吃 `maix.image.Image` | [源码] 已存在 |
| ncnn | `Mat.from_pixels(buffer, w, h, PixelType)` 直吃像素缓冲 | [源码] 已有原语，需适配 |
| onnxruntime | 无（本来就是 numpy） | canonical 即快路径 |
| rknn-lite | 无（本来就是 numpy NHWC） | canonical 即快路径 |

等价性验证必须分两档，不要求所有引擎走同一条路：

1. 张量级引擎（onnxruntime / ncnn / rknn_lite）：canonical 路径输出原始张量，快路径必须与自己的 canonical 路径一致。
2. 端到端引擎（maix.nn）：它不吐原始张量，无法提供张量级 canonical 路径；等价性在 `Detection` 层比对——同一张黄金帧，maix.nn 出的框与 onnxruntime 加 `vision_det` 出的框对比。

- 必须把 maix.nn 排除在张量级不变量之外。
  理由：它内置了 resize、归一化、解码、NMS，无法拆出可比的原始张量；要求它吐张量会迫使 mud 走慢路径，得不偿失。
- 快路径只能是可选加速，不得是实现分支：只要与参考路径偏离，就说明它对预处理做了框架不知道的处理，必须拒绝启用。

## §3 差异矩阵

状态：`[已定]`（标注 `[推断]` / `[待测]` 的单元格需上机确认）

下表固定四个引擎在 io、布局、dtype、归一化、resize、输出、设备、量化、加载与 GIL 等维度的差异。

| 维度 | maix.nn | ncnn | onnxruntime | rknn-lite |
|---|---|---|---|---|
| io 寻址 | 隐式（单模型） | blob 名，`input_names()` 可自述 [源码] | tensor 名，`get_inputs()` 可自述 [源码] | 隐式（顺序） |
| 原生布局 | RGB888 / BGR 图像 | HWC 像素 | NCHW（默认） | NHWC |
| dtype | uint8 | float32 | float32 | float32 |
| 归一化 | 烘焙在 mud 里 | 引擎适配器调 `substract_mean_normalize` [源码] | 框架做 | 转换期烘焙（`std_values`）[源码] |
| resize | 烘焙在 mud 里 | 框架做 | 框架做 | 框架做（官方示例自己 `cv2.resize`）[源码] |
| 输出 | 已解码的对象列表 | 原始 blob | 原始 tensor | 原始 tensor（按顺序） |
| 后处理 | 引擎自带 | 框架做 | 框架做 | 框架做 |
| 设备 | NPU | CPU / Vulkan | CPU / CUDA / TensorRT EP | NPU（`core_mask` 选核） |
| 量化 | int8（厂商工具链） | fp16 / int8（量化表） | fp16 / int8（QDQ） | int8（厂商工具链） |
| 加载耗时 | 200-500ms（flash 重载） | 快 [推断] | 中；持 GIL [源码] | 快 [推断] |
| 推理时 GIL | 持有（pybind11 默认）[待测] | 持有（pybind11 默认）[待测] | `run()` 释放；`io_binding` 历史有坑 [源码] | 持有（C API 包装）[待测] |
| 单模型多输出 | 内建 | 多个 blob，按名取 | 多个 tensor，按名取 | 多个输出，按顺序 |
| Python 绑定 | maix.nn | pybind11 | pybind11 | C API 的 Python 包装 |

## §5 GIL 与线程

状态：`[已定]`（多数单元格 `[待测]`）

### §5.1 事实

状态：`[待测]`

下表固定各引擎在推理期间与加载期间的 GIL 事实。

| 引擎 | 推理期间 | 加载期间 |
|---|---|---|
| onnxruntime | 释放 [源码] | 持有 [源码, #27063] |
| ncnn | 持有（pybind11 默认）[待测] | 持有 [待测] |
| maix.nn | 持有（pybind11 默认）[待测] | 持有 [待测] |
| rknn-lite | 持有（C API 包装）[待测] | 持有 [待测] |

### §5.2 后果

状态：`[已定]`

推理在 921600 baud 的 UART 链路旁边跑时：

- 释放 GIL 的引擎：UART 接收线程能穿插执行，安全。
- 持有 GIL 的引擎：推理期间 Python 线程全部停摆，UART 字节可能溢出硬件 FIFO。

### §5.3 对策（按优先级）

状态：`[已定]`

1. 把推理放进专用线程，并让 UART 接收不依赖及时拿到 GIL：接收用 C 层中断 / 环形缓冲，Python 只在处理时取。
2. 降低推理频率，给 UART 留出时间窗。
3. 在链路紧张的场景选释放 GIL 的引擎（ORT 是现成的）。

- 模型加载与切换必须发生在主循环的安全点，不得在 UART 订阅进行中切模型。
- 预加载后 `switch()` 退化为指针交换，O(1) 且不碰加载路径，彻底绕开这个问题。

## §6 接入验收标准

状态：`[已定]`

每个引擎适配器必须全部满足以下各项，缺一不可。

- 加载失败抛 `E_MODEL_LOAD_FAILED`，不静默。
- 未知模型族抛 `E_MODEL_TYPE_UNSUPPORTED`，不兜底到通用类。
- 输出 shape 与 manifest 声明不符，抛 `E_OUTPUT_LAYOUT_MISMATCH`。
- canonical 路径在黄金帧集上与参考引擎结果一致（IoU / 类别 / 置信度容差）。
- 若有快路径，与自己的 canonical 路径一致；maix.nn 走 `Detection` 级对齐（§2）。
- 在 `Capabilities` 里声明 device / quant / layouts / dynamic_shape / input_accepts / releases_gil / load_holds_gil / io_binding。
- 缺依赖时抛 `E_ENGINE_UNAVAILABLE`，不静默变 stub。
- 加载耗时与是否持 GIL 已标注（供上层决定加载时机）。
- io 映射在引擎适配器内完成，wrapper 代码里不出现引擎物理名。
