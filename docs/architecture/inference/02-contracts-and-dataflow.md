> 本节目标：固定核心数据结构、引擎协议、wrapper 机制，以及从原始帧到 `Detection` 的三层数据流契约。

# 02 契约与数据流

来源：架构文档 §3 核心数据结构与协议、§4 数据流契约。整体状态：`[已定]`。

## §3 核心数据结构与协议

状态：`[已定]`

### §3.1 contract.py

状态：`[已定]`

以下类型必须定义在 `framework/inference/contract.py`，且为唯一定义处。

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]           # -1 表示动态维度
    dtype: str                       # "float32" / "uint8" / "int8"


@dataclass(frozen=True)
class Capabilities:
    device: str                      # "cpu" / "cuda" / "vulkan" / "npu" / "axera"
    quant: str                       # "fp32" / "fp16" / "int8"
    layouts: tuple[str, ...]         # ("nchw",) / ("nhwc",)
    dynamic_shape: bool
    input_accepts: tuple[str, ...]   # 见 §3.3
    io_binding: str                  # "named" / "positional"
    releases_gil: bool               # run() 期间是否释放 GIL
    load_holds_gil: bool             # load() 期间是否持有 GIL
    memory_hint: str | None = None   # 常驻开销提示
    load_cost_hint: str | None = None  # 加载耗时提示


@dataclass(frozen=True)
class ArtifactRef:
    engine_id: str
    path: str                        # 引擎加载的主工件文件
    wrapper: str                    # 任务适配层入口（内置名或 "module:attr"）
    wrapper_params: dict[str, Any]
    preprocess: PreprocessSpec
    io: IOMapping
    runtime: dict[str, Any]          # 包名 / 版本（可选）
    build: dict[str, Any]            # 构建配方（可选）
    aux_paths: dict[str, str]        # 伴随文件，按角色取值（如 ncnn 的 "bin"）
```

- `TensorSpec`、`Capabilities`、`ArtifactRef`、`DetectionAPI` 必须在旧文档被反复引用而无定义的地方给出唯一定义处。
- `aux_paths` 是 ncnn 这类"主工件 + 伴随文件"形态的落点：`.param` 是主工件，`.bin` 按角色 `"bin"` 挂在 `aux_paths` 里。
- 其余结构（`PreprocessSpec`、`IOMapping`）必须定义在 `manifest.py`。

### §3.2 引擎协议

状态：`[已定]`

```python
class InferenceEngine(Protocol):
    engine_id: str

    def load(self, artifact: ArtifactRef, **opts) -> None: ...
    def input_specs(self) -> list[TensorSpec]: ...
    def output_specs(self) -> list[TensorSpec]: ...
    def capabilities(self) -> Capabilities: ...
    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]: ...
    def unload(self) -> None: ...
```

- `input_specs` / `output_specs` 必须允许有多个成员。
  理由：YOLOv5 与 RKNN 都有多输出（`outputs: [0, 1, 2]`），单数 `TensorSpec` 的协议不支持这类模型。
- `input_specs` / `output_specs` 唯一用途是启动期与 manifest 的 io 声明交叉校验。
- `run()` 除了输入张量还接受**运行选项**：方言要把数值参数交给引擎时用。端到端引擎自带阈值筛选，方言不把阈值交下去就只能吃引擎的默认值——真机实测两者结果不同（现役 `conf_th=0.3`/`iou_th=0.4` 得 4 框，引擎默认 `0.5`/`0.45` 得 3 框）。
  引擎只把参数转交给自己的运行时，不解释其含义，不认识的键忽略；这不违反"引擎不认识任务类型"，因为引擎并不需要知道这些参数是什么。

Guards（硬规矩）：

1. 引擎不得做后处理，不得认识 `Detection`，不得认识任务类型。
2. 引擎只做"跑一步"：命名字典进，命名字典出。
   理由：多步循环（如 LLM 生成）属于 wrapper。
3. 图内是否已带 NMS 必须由方言（wrapper 名）声明，不得放在引擎能力里。
   理由：对已带 NMS 的图再跑一次 NMS 会打乱框且不报错，必须显式建模。
4. 快路径只能是可选加速，不得是实现分支；必须通过等价性验证才允许启用。

### §3.3 Capabilities 字段表（唯一定义）

状态：`[已定]`

下表是 `Capabilities` 各字段的唯一含义与校验对象定义。

| 字段 | 含义 | 校验对象 |
|---|---|---|
| `device` | `cpu` / `cuda` / `vulkan` / `npu` / `axera` | 平台 |
| `quant` | `fp32` / `fp16` / `int8` | 模型 |
| `layouts` | 支持的输入布局 `nchw` / `nhwc` | 模型 |
| `dynamic_shape` | 是否支持动态输入尺寸 | 模型 |
| `input_accepts` | 能直接吃的帧形态 | 平台 |
| `releases_gil` | 推理期间是否释放 GIL | 引擎 |
| `load_holds_gil` | 加载期间是否持有 GIL | 引擎 |
| `io_binding` | `named` / `positional` | 引擎 |
| `memory_hint` | 常驻开销提示 | 提示 |
| `load_cost_hint` | 加载耗时提示 | 提示 |

下表固定各引擎的 `input_accepts` 取值与零拷贝能力。

| 引擎 | input_accepts | 零拷贝 |
|---|---|---|
| `maix_nn` | `("native_image", "numpy_bgr")` | 直接吃 `maix.image.Image` |
| `ncnn` | `("numpy_pixels", "numpy_bgr")` | `Mat.from_pixels` 直吃像素缓冲 |
| `onnxruntime` | `("numpy_bgr",)` | 本来就是 numpy |
| `rknn_lite` | `("numpy_bgr",)` | 本来就是 numpy |

- `input_accepts` 必须是元组，且按优先级排序。
- 三种形态的取舍：`native_image` / `numpy_pixels` / `numpy_bgr`；其中 `numpy_pixels` 指 HWC uint8 原始像素缓冲。
- `builtin_nms` 字段不得存在：它不是引擎能力，而是方言的属性。

### §3.4 wrapper 机制

状态：`[已定]`

- wrapper 必须是任务适配层：把底层通用的 Engine 适配成某个具体任务的接口。
- Engine 是通用的——只跑一步，输入输出都是不解释内容的命名字典。
- wrapper 必须完成这层翻译：业务层要的不是"跑一步"，而是"用这个模型做检测"或"用这个模型做一次对话"。

下表固定四个层次的分工与对外形状。

| 层次 | 关心什么 | 对外形状 |
|---|---|---|
| Engine | 怎么算一步 | `dict[str, Any]` 进，`dict[str, Any]` 出，不解释内容 |
| 引擎适配器 | 怎么搬运成引擎原生格式 | 布局、dtype、通道序、io 寻址 |
| wrapper | 怎么变成这个任务的接口 | `detect(frame) -> list[Detection]`、`chat(prompt) -> str` |
| 业务层 | 用这个任务 | 只认 wrapper 暴露的接口 |

- wrapper 只做两件事：按方言解释输出（解码 / NMS），把结果装进任务的数据类型。
- 输入准备不得放在 wrapper 里：缩放与归一化由"模型 + 引擎"这一对的契约决定，属于 runtime（§4.2），与任务无关。
- 此处的 wrapper 不是 Python 语境里的函数修饰器。

```python
def build(runtime: ModelRuntime) -> TaskAPI:
    """把一个已加载的 runtime 包装成领域 API。"""
```

- `build` 的产物必须是能力对象，不得是裸函数：

```python
class TaskAPI(Protocol):
    provides: frozenset[str]        # 这个 wrapper 提供的任务能力
    def detect(self, frame, **kw) -> list[Detection]: ...
```

- facade 必须按 `provides` 代理（§12.1，见 §12.1（07-migration-phases.md）），不得靠 `hasattr` 猜。
  理由："这个方言到底支持哪些任务"必须显式声明，缺能力时抛 `E_DIALECT_UNSUPPORTED` 而不是静默降级。
- `get_mask` 这类与某次推理结果绑定的能力必须挂在对象上。
  理由：闭包挂不住。
- wrapper 只调 `rt.preprocess()`，不得自己实现缩放与归一化。
  理由：输入准备的*实现*在 runtime（§4.2），wrapper 是调用者；这样 letterbox / NMS 在全框架只有 `ops.py` 一份实现（§1.1，见 §1.1（01-scope-and-boundaries.md））。
- wrapper 的名字不得承载语义：名字只是内置注册表的键，模型族另由 manifest 的 `family` 字段声明。

```python
BUILTIN = {
    "vision.det.raw": {
        "target": "framework.inference.wrappers.vision_det:build_raw",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "raw",
    },
    "vision.det.end2end": {
        "target": "framework.inference.wrappers.vision_det:build_end2end",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "end2end",
    },
}
```

- `vision.det.end2end` 同时服务 yolo11 与 yolov5：maixcam2 的 `.mud` 是 YOLO11，maixcam1 的 `.cvimodel` 是 YOLOv5，两者都是引擎内建后处理的端到端形态。
  理由：端到端形态的判定依据是"引擎是否自带后处理"，不是模型族；把 yolov5 排除在外会让 maixcam1 的工件在加载期必失败。
- manifest 必须通过 `wrapper: "vision.det.raw"`（内置）或 `wrapper: "mypkg.wrappers:build"`（外部扩展）指定。
- 解析规则：
  1. `"module:attr"` 形式，直接 import 该模块的该属性。
  2. 否则查内置名字表 `BUILTIN`。
  3. 都找不到，抛 `E_DIALECT_UNSUPPORTED`。
- 内置注册表的每个条目必须带元数据：`families` 用于加载期校验"模型族与方言匹配"（§7.3，见 §7.3（04-runtime-and-engine-selection.md）），`provides` 用于 facade 能力代理，`output_form` 声明输出形态（raw / end2end，§4.4）。
- 名字里不解析任何东西：`vision.det.raw` 同时服务 yolo11 与 yolov5，族差异由 `wrapper_params` 与 manifest 的 `family` 各自承担。
- 核心框架只内置 `vision_det`（`vision.det.raw` / `vision.det.end2end` 两个条目）。
  理由：分割、姿态、分类、LLM 一律不预写，等真要用时按 §3.5 的规则加。
- wrapper 必须落在 `framework/inference/wrappers/`。
- wrapper 允许出现任务的数据类型（`Detection`、`Keypoint`、`SegmentResult`），但实现里只允许通用视觉原语与通用文本处理（letterbox、归一化、decode、NMS、IoU、softmax、chat template、tokenize、detokenize），不得出现任何具体被检测对象的语义。

### §3.5 加一个新任务要做什么

状态：`[已定]`

下表固定各类扩展场景对应的改动面，核心层、协议、manifest 结构、错误码体系均不得改动。

| 场景 | 要做的事 |
|---|---|
| 加一种检测方言（如 RT-DETR） | 写一个 wrapper，把引擎适配成这个检测接口，注册名字，manifest 里改一行。名字不承载语义，模型族由 manifest 的 `family` 声明 |
| 换一个模型尺寸或归一化方式 | 改 manifest 的 `input` / `preprocess` 段，不动 wrapper |
| 加分割 / 姿态 / 分类 | 同上 |
| 加 LLM 对话 | 写一个 wrapper，把引擎适配成对话接口（内部用 `rt.engine.run()` 做生成循环），核心层零改动 |
| 换一个完全不同的推理库 | 写一个 engine adapter，登记到 `REGISTRY` |

## §4 数据流契约

状态：`[已定]`

### §4.1 三层协议

状态：`[已定]`

推理的输入侧有三层协议，各管一段。下表固定三层协议的声明位置、职责与维护方。

| 协议 | 在哪声明 | 回答什么 | 谁维护 |
|---|---|---|---|
| 平台帧协议 | HAL 的 `Camera` 协议（平台层） | 相机实际给什么帧：通道序 / dtype / 布局，另有可选的 maix.Image 原生帧 | 平台 / 相机 |
| 框架配置协议 | manifest 的 `input` 段（逻辑模型级）加工件的 `preprocess` 段 | 这个模型要什么：帧形态、目标分辨率、缩放策略、归一化参数与归属 | 模型作者 |
| 引擎输入协议 | `InferenceEngine.input_specs()`（引擎物理级） | 这个引擎能吃什么：端口名、shape、dtype、布局 | 引擎适配器 |

- 平台声明它给什么，manifest 声明模型要什么；两者不一致由前处理转换，不算报错。
- 帧协议必须仍由平台拥有：它描述相机输出，与用哪个模型、哪个引擎无关。
- "这个模型期望的帧形态"必须写在 manifest 里。
  理由：否则同一相机下每个引擎工件都要重复声明同一份期望，而模型作者改不了相机的输出。
- 现役帧协议已写在 `framework/hal/interface/camera.py`：`read()` / `read_raw()` 返回 BGR ndarray，`last_raw` 返回 maix.Image（非 maix 平台为 `None`）。
- 缺口：帧协议只写在 docstring 里，没有可查询的声明。必须补一个能力声明，让前处理不必硬编码 "BGR"：

```python
@dataclass(frozen=True)
class FrameSpec:
    color: str          # "bgr" / "rgb"
    dtype: str          # "uint8" / "float32"
    layout: str         # "hwc"
    native_image: bool  # 是否另有零拷贝原生帧
```

- 平台实现 `frame_spec()`，runtime 在规范段前读一次。
- manifest 的 `input.frame` 声明模型要的形态：

```yaml
input:
  frame: { color: rgb, dtype: uint8, layout: hwc }   # 模型期望
  size: [1280, 352]
  resize: { mode: letterbox, value: 114 }
```

- 两者不一致时，规范段按 §4.2 转换（通道序转换、dtype 扩展、布局重排）。
- 转换是数据搬运，不是契约违反；只有转换**做不到**时才报错，例如模型要 RGBA 而平台给的是 3 通道。
- 三层的关系是"声明"对"实现"：

```
框架配置协议              引擎输入协议
  目标分辨率 640x352   ->   shape (1, 3, 640, 352)
  通道序 rgb          ->   布局 nchw / nhwc
  归一化 scale/mean/std ->   dtype float32
```

- 前处理必须分两段：第一段把**帧协议**（平台）映射到**框架配置协议**（模型），第二段把框架配置协议映射到**引擎输入协议**（引擎）。
- 配置协议与引擎协议对不上时必须在加载期抛错（§7.3，见 §7.3（04-runtime-and-engine-selection.md）），不得带着矛盾跑推理。

### §4.2 前处理：产出引擎能直接吃的输入

状态：`[已定]`

- 前处理是"模型 + 引擎"这一对的事，不由任务决定，因此属于 runtime，不属于 wrapper。

```python
pre = rt.preprocess(frame)        # PreprocessResult
inputs = pre.inputs               # dict[str, Any]，逻辑名 -> 已前处理张量
```

- 前处理必须分两段：
  1. **规范预处理**（平台帧协议 → 框架配置协议），三小步：
     1. **帧形态转换**：平台帧的实际形态（`FrameSpec`）→ manifest `input.frame` 声明的形态。通道序 / dtype / 布局不一致就转换（§4.1）。
     2. **分辨率处理**：按 `input.resize.mode` 把帧变成 `input.size`（§4.3.1）。帧已等于目标尺寸时整步跳过。
     3. **归一化**：由工件的 `preprocess.norm` 三档决定谁做（§4.3）。规范预处理的产出形态随 owner 变化，见 §4.3 的三行表。
  2. **物理适配**（按引擎输入协议）：把规范张量转成引擎要的 layout 与 dtype。
     理由：这是引擎适配器的活，因为它取决于引擎（实现要点见 `inference_backends.md` §1.5，见 §1.5（08-backend-contracts.md））。

- 规范预处理的产出**必须带回几何元数据**，否则 wrapper 无法把框反算回原帧坐标。
  理由：缩放比与偏移取决于原帧分辨率，是每帧变量。

```python
@dataclass(slots=True)
class GeometryTransform:
    """原帧坐标与输入图像素坐标之间的映射，wrapper 反算坐标的依据。"""

    mode: str        # "letterbox" / "crop" / "stretch"
    scale_x: float   # 原帧 -> 输入的缩放比
    scale_y: float
    offset_x: int    # 有符号：letterbox 为左填充量，crop 为左裁切量（负）
    offset_y: int
    src_w: int       # 原帧尺寸
    src_h: int


@dataclass(slots=True)
class PreprocessResult:
    inputs: dict[str, Any]        # 逻辑名 -> 已前处理张量
    transform: GeometryTransform   # 反算坐标用的几何元数据
```

- 这两个结构每帧新建，只用 `slots=True`，不得用 `frozen=True`（§10.1 规则 1，见 §10.1（06-performance-and-equivalence.md））。

**规范段在推理路径上不得分配内存。** `ModelRuntime` 必须在 `load()` 时一次性分配全部工作缓冲（letterbox 目标缓冲、归一化 float32 缓冲、仅 ORT 需要的 NCHW 缓冲），`ops.py` 的算子一律接受 `out=` / `dst=` 目标参数并写入其中，不得隐式返回新数组。
理由：嵌入式 glibc 对大块走 `mmap`，首次写触发 page fault 与内核清零；每帧分配同时制造碎片。该约束与 CMSIS-NN 的单 arena 约定、TFLM 的 `MicroAllocator` arena 契约同构。

- 反算约定：`x_src = (x_input - offset_x) / scale_x`，`y_src` 同理。
- letterbox 的 `offset` 是填充量（正），crop 的是裁切量（负），stretch 的缩放比两轴不同。
- 前处理不得做任何解释：它不知道"框"是什么。

### §4.3 归一化归属：工件级契约，三档

状态：`[已定]`

- `norm` 不是布尔开关，必须由每个工件逐段声明。

下表固定 `preprocess.norm` 三档的含义、执行方与可验证性。

| `preprocess.norm` | 含义 | 谁执行 | 谁能验证 |
|---|---|---|---|
| `framework` | 框架的规范预处理段做 | runtime / 前处理 | 黄金帧集可比对 |
| `engine` | 框架给参数，引擎适配器执行 | 引擎适配器 | 等价性测试覆盖 |
| `baked` | 已烘焙在工件里 | 无人（框架侧） | 只能靠构建期保证 |

**`baked` 只表示"框架不执行"，不表示"设备侧不执行"。** YOLO11 的 axmodel 把预处理集成为**第一层 CPU 算子**——转换器文档原文："对于 YOLO11，因为模型集成了预处理，也就是说第一层是 CPU 处理"（`docs/doc/zh/ai_model_converter/maixcam2.md`）。因此 maix.nn 工件虽然声明 `preprocess: { resize: baked, norm: baked }`，resize 与归一化仍在**设备 CPU** 上执行，成本落在 CPU 预算而非 NPU 预算内。评估端到端引擎的 CPU 开销时必须把这一层算进去。

规范段的输出形态随 `norm` owner 变化。下表固定三档下规范段的产出与适配器职责。

| `preprocess.norm` | 规范段产出 | 适配器职责 |
|---|---|---|
| `framework` | float32，已按 `input.norm` 归一化 | 转 layout 与 dtype |
| `engine` | uint8，像素域 HWC | 转 float32 + 引擎内归一化 |
| `baked` | uint8，像素域 HWC | 转 float32，**不得**预先除 255 |

- 公式语义必须由工件的 `preprocess.formula` 显式声明，两档互不解释。

下表固定 `preprocess.formula` 的两值语义与典型用者。

| `formula` | 公式 | 典型用者 |
|---|---|---|
| `yolo` | `(x * scale - mean) / std` | `framework` 档的规范段 |
| `substract_mean_normalize` | `(x - mean) / std` | `engine` 档的 ncnn 适配器 |

- `input.norm` 的 `{scale, mean, std}` 永远是 yolo 形式（模型级）。
- `engine` 档的适配器拿到这组参数后必须按公式换算成引擎式：`mean_e = mean / scale`，`std_e = std / scale`（等价变形，可直接验算）。
- 不写 `formula` 就不猜：`norm` 非 `baked` 而缺 `formula` 抛 `E_CONTRACT_MISSING`（§8，见 §8（05-errors-and-observability.md））。
- `norm: baked` 时禁止出现 `formula`。
  理由：都烘焙了，写公式自相矛盾。
- `baked` 是最危险的一档，框架无法验证它是否正确。
- 对 rknn_lite，manifest 的 `build` 段必须记全 `mean_values` / `std_values`，框架据此校验工件声明的输入属性与配方一致，不一致直接抛错。

#### §4.3.1 分辨率策略：三种 mode

状态：`[已定]`

- 帧的分辨率与 `input.size` 不一致时的处理是模型作者的决定，必须写在 `input.resize.mode`。

下表固定三种 `resize` mode 的做法与适用场景。

| `mode` | 做法 | 适用 |
|---|---|---|
| `letterbox`（默认） | 等比缩放使整帧可见，不足处用填充值补齐 | 保留完整视野，物体不变形 |
| `crop` | 等比缩放使画面覆盖目标，超出部分裁掉 | 用分辨率换视野，不留边 |
| `stretch` | 直接拉伸到目标尺寸，不保宽高比 | 模型对宽高比不敏感 |

- 三种 mode 的产出都必须是精确的 `input.size`。
- `crop` 的裁切锚点由 `input.resize.anchor` 决定，默认 `center`。

下表固定 `crop` 各锚点的裁切位置。

| `anchor` | 裁切位置 |
|---|---|
| `center` | 从中间裁（默认） |
| `top` / `bottom` | 从上边 / 下边裁 |
| `left` / `right` | 从左边 / 右边裁 |

- `letterbox` 的填充由 `input.resize.value` 决定（缺省 114，与 YOLO 训练惯例一致）。
- `value: transparent` 表示全透明填充，**只在工件自述接受 4 通道输入时合法**，否则加载期抛 `E_OUTPUT_LAYOUT_MISMATCH`。
  理由：3 通道模型没有"透明"可言。
- `resize` 同样有两档归属：`framework`（框架做，规范段的第 2 小步）与 `baked`（已烘焙在工件里，框架整步跳过，manifest 的 `mode` 不生效）。
- 三种 mode 的实现都必须落在 `ops.py` 里，且是全框架唯一一份（§1.1，见 §1.1（01-scope-and-boundaries.md））。
- 等价性验证必须覆盖 mode。
  理由：不同 mode 产生的框位置不同，跨引擎比对前先确认两边走的是同一个 mode。

### §4.4 `engine.run()` 的契约

状态：`[已定]`

```python
def run(self, inputs: dict[str, Any]) -> dict[str, Any]: ...
```

- **输入**：必须是前处理产物。key 是逻辑名，value 是引擎能直接消费的张量（物理名已由适配器翻译）。引擎不得接收原始帧。
- **输出**：key 是逻辑名，value 有两种形态，由方言声明。

下表固定 `engine.run()` 两种输出形态的 value 内容、后处理责任方与典型引擎。

| 形态 | value 是什么 | 谁做后处理 | 例子 |
|---|---|---|---|
| raw | 原始张量 `np.ndarray`，形状按 `output_specs()` 声明 | wrapper 解码加 NMS | onnxruntime / ncnn / rknn_lite |
| end2end | 已解码的对象列表，引擎内置了后处理 | wrapper 只做格式转换 | maix.nn |

- 引擎不得解释输出的内容。
  理由：raw 形态下"这个张量怎么解释"是方言（wrapper）的事；end2end 形态下引擎已经把后处理做完，wrapper 不得重复做。
- raw 形态的解码步骤（`vision.det.raw`）：
  1. 取 `pred`，按 `wrapper_params.output_layout`（`auto` 时按形状判别）确定是 `[1, 4+nc, N]` 还是 `[1, N, 4+nc]`。
  2. 类别数 `nc` 来自逻辑模型的 `labels_file` 行数。
     理由：没有标签表就切不出类别分数。
  3. 前 4 列是框（`cx, cy, w, h`），后 `nc` 列是类别分数；`wrapper_params.score_activation` 决定是否过 sigmoid。
  4. 坐标是缩放之后的输入图像素域，按 `pre.transform` 反算回原帧坐标。
     理由：反算用的是规范段算出的 `GeometryTransform`，不是 manifest 的静态 `input.size`——缩放比与偏移随原帧分辨率变，`crop` 模式下还要算上裁切偏移。
  5. 置信度筛选与 NMS，全部向量化；解码之后必须复刻现役那道后置过滤 `score >= conf_th`（§10.1 规则 3，见 §10.1（06-performance-and-equivalence.md））。
     理由：现役 maix.nn 不完全遵守 `conf_th`，raw 路径自己筛选，两条路径的语义必须一致，否则 Phase 1 的"逐帧一致"不成立。
  6. 产出 `Detection` 列表：`class_id` 是类别索引，类别名从 `labels_file` 取。
- end2end 形态不得有上述步骤：引擎已经把解码与 NMS 做完，wrapper 只做字段映射。
- **错误**：引擎内部失败必须抛异常，由 facade 映射成错误码（§8，见 §8（05-errors-and-observability.md））；不得吞掉，不得返回空张量。

### §4.5 io 映射：逻辑名到物理名

状态：`[已定]`

- `engine.run()` 的 key 一律用逻辑名，物理名的差异由引擎适配器按 `artifact.io` 翻译。
- 这是 wrapper 与 engine 之间唯一的寻址约定。

manifest 侧：

```yaml
io:
  inputs:  { image: "data" }
  outputs: { pred: "output" }
```

wrapper 侧（永远只写逻辑名）：

```python
out = rt.engine.run({"image": blob})
pred = out["pred"]
```

- 输入固定用逻辑名 `image`。
- 输出用逻辑名映射；多输出时映射可以有多个条目，**第一个键是主输出**（YAML 映射的书写顺序即语义）。
- `io.outputs` 不得为空。端到端引擎（maix.nn）没有物理名，也必须显式声明逻辑名（`outputs: { objects: "objects" }`），此时逻辑名与物理名相同。
- wrapper 不得硬编码主输出名，必须取 `artifact.io.outputs` 的第一个键。
  理由：名字由 manifest 决定，wrapper 不猜。
- 物理值可以是名字（`"data"`、`"output0"`）或索引（`0`、`1`）。rknn_lite 是顺序寻址，只能用索引。
- 引擎适配器必须在 `load()` 时把 `artifact.io` 存下来，在 `run()` 时做双向翻译。
- 旧写法（wrapper 写死 `rt.engine.run({"images": blob})`）不得沿用。
  理由：旧写法让 wrapper 与具体引擎的物理名耦合，换引擎要改 wrapper，且 manifest 声明的 io 形同虚设；物理名只允许出现在 manifest 与适配器里。

### §4.6 闭环：从原始帧到 VisionResult

状态：`[已定]`

一次 detect 的完整链路：

```
原始帧 (maix.Image | numpy BGR)
  |  (1) 前处理   rt.preprocess(frame)         <- 框架配置协议 + 引擎输入协议
  v
引擎就绪的张量 dict
  |  (2) 推理     rt.engine.run(inputs)         <- 引擎只跑一步，不解释
  v
原始张量 dict / 已解码对象列表
  |  (3) 后处理   wrapper（解码 + NMS）          <- 方言决定怎么解释
  v
list[Detection]                                  <- 框架层数据契约
  |  (4) 组装     AIInferenceProcessor          <- modules 层
  v
VisionResult(success, result_data={"detections", "segments", "infer_fps", ...})
```

- 前四段里，(1)(2)(3) 属框架与推理层，(4) 属 modules 层。
- **推理层的产出边界是 `list[Detection]`**：推理层不知道 `VisionResult` 是什么，也不该知道。
- 现役对应位置：`AIInferenceProcessor.process()`（`modules/zw_opencv_module/processors/ai_inference_processor.py`）拿到 `list[Detection]` 后做过滤（`filter_detections`）并组装成 `VisionResult`（定义在 `modules/zw_opencv_module/processors/base.py`）。
- 重构后 modules 层这一层不动，只是 `ai.detect()` 的入参不再需要业务层自己判断 `_raw`（§4.7）。

### §4.7 帧适配：谁把 maix.Image 变成 numpy

状态：`[已定]`

现役做法是业务层自己判断：

```python
if maix_img is not None:
    objs = ai.detect(maix_img, ..., _raw=True)
else:
    objs = ai.detect(frame, ...)
```

- 现役做法是当前唯一把平台知识捅到业务层的地方（`modules/zw_opencv_module/processors/ai_inference_processor.py`）。
- 定法：原生帧的取用归平台层，形态转换归 runtime。

1. 原生帧由调用方交给 facade：`detect()` 增参 `camera=None`，调用方把已有的相机对象透传进来。
   理由：相机是平台能力对象，facade 只消费它的 `last_raw`，不自己持有相机。
2. facade 调用 runtime 之前判断要不要原生帧：`engine.capabilities().input_accepts` 含 `native_image`，且 `camera` 非 `None`、`camera.last_raw` 非 `None` 时给原生帧，否则给 numpy。
3. 规范预处理整段是否旁路，**由工件的 `preprocess.resize` / `preprocess.norm` 是否为 `baked` 决定，与帧来源无关**（§4.8）。
   理由：`baked` 表示工件自带预处理，框架不插手；判据挂在工件上，原生帧与 numpy 两条路行为一致。
4. numpy 路径下，平台帧的实际形态（`FrameSpec`）与 manifest 的 `input.frame` 不一致时，由规范段转换（§4.1）。
   理由：转换后前处理与 `ops.py` 都不需要知道 `maix.image` 存在，隔离规则不被破坏。

- `ai_inference_processor.py` 里的 `_raw=True` 分支必须在 Phase 1 随平台帧适配一起删除，改为透传 `camera=context.get("camera")`。
  理由：这是本次重构里 modules 层唯一必须改动的地方（见 §12.2（07-migration-phases.md））。

### §4.8 maix.nn 是端到端特例

状态：`[已定]`

- `maix.nn` 不吐原始张量：它内部把 resize、归一化、NPU 推理、解码、NMS 全做完，直接给检测框对象。因此：
  - 它的工件必须声明 `preprocess` 全是 `baked`；前处理阶段对它不做 letterbox 也不做归一化，直接把帧交给引擎（numpy 帧由引擎适配器转成引擎可吃的形态）。
  - 引擎适配器必须调 `maix.nn.YOLO11.detect()`，把 `maix.nn.Object` 封装后交回 wrapper。
  - wrapper 必须把结果转成框架的 `Detection`，不再跑 NMS。
- 端到端方言必须把阈值交给引擎（`run(inputs, conf_th=..., iou_th=...)`），并在映射之后按 `score >= conf_th` 再过滤一次。
  理由：现役路径显式把 `conf_th` / `iou_th` 传给 `maix.nn.detect()`，并在引擎返回后补一道 `score >= conf_th` 的后置过滤（有些模型不完全遵守传入的阈值）；两处都省掉，结果就与现役不一致。
- `maix.nn` 因此不提供"张量级 canonical 路径"。
- 等价性验证必须按 §11（见 §11（06-performance-and-equivalence.md））在端到端 `Detection` 层做。
