> 本节目标：固定 ncnn 的工件形态、加载入口、blob 寻址、归一化适配与已知坑，作为 ncnn 适配器的实现要求。

# 10 引擎适配：ncnn

来源：后端文档 §4.2 ncnn，并引 §1 / §2 / §3、架构文档附录 C.2。

下表是源文档 §4.2  的原始适配清单；表内各行内容分别落到本文件第 1–10 节对应标题下。

| 项 | 内容 |
|---|---|
| 工件 | `.param`（网络结构）加 `.bin`（权重），或 `.param.bin`（ncnn2mem 生成的二进制结构） |
| 加载 | `net.load_param()` 加 `net.load_model()`；`load_param_bin()` 单文件 [源码] |
| io | 按 blob 名；`net.input_names()` / `net.output_names()` 可自述 [源码] |
| 输入 | `ncnn.Mat(numpy)` 零拷贝（buffer protocol）[源码]；也可用 `Mat.from_pixels()` |
| 预处理 | 有 `substract_mean_normalize()` 原语；`norm: engine` 档由适配器调用，公式 `(x - mean)/std` 需由 `input.norm` 的 yolo 形式换算（`mean_e = mean / scale`，`std_e = std / scale`） [源码] |
| 设备 | `net.opt.use_vulkan_compute` / `num_threads` / fp16 / int8 存储 [源码] |
| 快路径 | `Mat.from_pixels(buffer, w, h, PixelType)` 直吃像素 [源码] |
| 坑 | Mat 要求 16 字节对齐，绑定层已处理 cstep [源码]；`extract(name, 0)` 是深拷贝、`1` 是引用 [源码] |
| GIL | [待测]，pybind11 默认持有 |

## 1. 标识与设备

- 引擎 id 必须是 `ncnn`（注册表键，见 §7.1（04-runtime-and-engine-selection.md））。
- 设备：CPU / Vulkan。
- Python 绑定：pybind11。
- 量化：fp16 / int8（量化表）。
- 角色定位：CPU 侧的性能降级选项；选 ncnn 的理由只能是"做对比"或"NPU 不支持该算子"。
  理由：ncnn 的 int8 收益主要在体积（模型小 75%），不在速度（见 §10.3（06-performance-and-equivalence.md））。

## 2. 工件形态与依赖

- 工件形态：`.param`（网络结构）加 `.bin`（权重），或 `.param.bin`（ncnn2mem 生成的二进制结构）。
- manifest 中必须同时给出 `file` 与 `bin`：`file: yolo11n.param` / `bin: yolo11n.bin`。
- 依赖：`ncnn`；只有 `framework/inference/engines/` 允许 import（见 §1.1（01-scope-and-boundaries.md））。

## 3. 加载

- 加载入口：`net.load_param()` 加 `net.load_model()`；`load_param_bin()` 单文件 [源码]。
- 加载耗时：快 [推断]。
- 加载期间持 GIL（pybind11）[待测]。
- 预载价值：低——按需加载也不痛。
- 多实例共存：可。
- 加载失败的错误码：`E_MODEL_LOAD_FAILED`（见 §6（08-backend-contracts.md））。

## 4. io 寻址

- io 按 blob 名寻址。
- `net.input_names()` / `net.output_names()` 可自述 [源码]。
- `.bin` 里没有张量名也没有形状，形状只能从 `.param` 的层参数反推；因此 manifest 必须显式声明 `io`。
  理由：`input_names()` / `output_names()` 并非所有 ncnn 版本都提供。
- 引擎适配器必须在 `load()` 时把 `artifact.io` 存下来，`run()` 时做双向翻译：逻辑名 `image` → blob 名（`"data"`），逻辑名 `pred` → 输出 blob 名。
- 多输出时按名取；主输出取 `artifact.io.outputs` 的第一个键。

## 5. 输入与快路径

- 规范输入必须是 HWC，直接包成 `ncnn.Mat`（零拷贝，不需要转置）。
  理由：ncnn 正好要 HWC；若规范输入定成 NCHW，这里每次都要转置（见 §10.1（06-performance-and-equivalence.md））。
- `ncnn.Mat(numpy)` 零拷贝（buffer protocol）[源码]。
- **`Mat.from_pixels()` 不得用于快路径**：它新建 Mat 并做像素/色彩转换，**不是零拷贝** [源码]。正确姿势是"规范段把 letterbox 结果写进自持缓冲 → 用外部数据构造 `Mat`"。
- 自持输入缓冲必须满足 16 字节对齐。numpy 的 buffer 不保证对齐，绑定层会改写 cstep 绕过，可能走非对齐 kernel；框架应分配自己的对齐缓冲。
- 快路径形态必须进入 `Capabilities.input_accepts`：`("numpy_pixels", "numpy_bgr")`（见 §3.3（02-contracts-and-dataflow.md））。
  其中 `numpy_pixels` 指框架自持的 16 字节对齐像素缓冲，不是 ncnn 的 `Mat.from_pixels`（已禁用，见上）。
- 设备选项：`net.opt.use_vulkan_compute` / `num_threads` / fp16 / int8 存储 [源码]。
- 快路径与 canonical 的逐像素一致性：[待测]，通道序与插值细节是待确认项。

## 6. 输出形态

- 输出形态：raw——原始 blob；后处理由框架做（wrapper 解码 + NMS）。
- 取输出必须用 `extract(name, 1)`（引用），不得用深拷贝。
  理由：取深拷贝（`0`）会让每帧多一次输出张量的内存拷贝，在 A53 上是几毫秒（见 §10.1 规则 2，见 §10.1（06-performance-and-equivalence.md））。
- 适配器只做搬运与寻址，不得认识 `Detection`（见 §3.2 硬规矩 1，见 §3.2（02-contracts-and-dataflow.md））。

## 7. 预处理归属

- ncnn 有 `substract_mean_normalize()` 原语 [源码]。
- `norm: engine` 档由适配器调用该原语，公式 `(x - mean)/std` 必须由 `input.norm` 的 yolo 形式换算（`mean_e = mean / scale`，`std_e = std / scale`）[源码]。
- 换算必须落在 `ops.py` 里，且是全框架唯一一份。
- `norm: engine` 档下规范段送的是 uint8 像素域，`Mat` 构造时自动扩展成 float32，再调 `substract_mean_normalize`。
- `resize` 归属：框架做（规范段第 2 小步）；三种 mode 的实现共用 `ops.py`（见 §4.3.1（02-contracts-and-dataflow.md））。
- manifest 对应声明：`preprocess: { resize: framework, norm: engine, formula: substract_mean_normalize }`。

## 8. 已知坑

- Mat 要求 16 字节对齐，绑定层已处理 cstep [源码]。`ncnn.Mat(numpy)` 对非对齐 numpy buffer 会改写 cstep 绕过，可能走非对齐 kernel；框架应提供自持的 16B 对齐输入缓冲。
- `Mat.from_pixels()` 是转换而非零拷贝 [源码]，误用会多一次整帧拷贝。
- `extract(name, 0)` 是深拷贝、`1` 是引用 [源码]。
- `.bin` 不含张量名与形状，形状只能从 `.param` 反推。
- ncnn / maix.nn / rknn 推理期间是否持 GIL：[待测]，该结论决定 §5.3 的对策选哪条（见 §5.3（08-backend-contracts.md））。
- ncnn int8 量化表与 ORT int8 的精度差：[待测]，影响等价性容差。
- `.param` 与 `.bin` 的格式约定（依据 ncnn 官方文档 `docs/developer-guide/param-and-model-file-structure.md`）：
  - `.param` 是"文本图"，即网络结构的逐层描述。第 1 行是魔数 `7767517`，第 2 行是层数与 blob 数，之后每行一层：`层类型 层名 输入数 输出数 输入blob... 输出blob... 键值对...`。参数用数字 ID 表达（`0=64` 是输出通道数，`1=3` 是卷积核宽），不是可读的键名。`ncnn2mem` 能把它转成二进制 `.param.bin`，此时不再是文本。
  - `.bin` 是权重：所有层的权重块按层顺序拼接，每块 32 位对齐，前面可选一个 flag 标识存储类型（`0` = float32，`0x01306B47` = float16，`0x01348B83` = bfloat16，其余为 int8 量化数据加量化表）。

## 9. 等价性验证要求

- ncnn 属于张量级引擎：canonical 路径输出原始张量，必须比对张量与解码后框。
- 快路径必须与自己的 canonical 路径一致。
- 等价性验证必须覆盖 resize mode。
  理由：不同 mode 产生的框位置不同，跨引擎比对前先确认两边走的是同一个 mode（见 §4.3.1（02-contracts-and-dataflow.md））。
- `Mat.from_pixels` 快路径与 canonical 的逐像素一致性：[待测]。
- 比对指标与容差按 §11 定稿（见 §11（06-performance-and-equivalence.md））。

## 10. 验收清单

- 加载失败抛 `E_MODEL_LOAD_FAILED`，不静默。
- 未知模型族抛 `E_MODEL_TYPE_UNSUPPORTED`，不兜底到通用类。
- 输出 shape 与 manifest 声明不符，抛 `E_OUTPUT_LAYOUT_MISMATCH`。
- canonical 路径在黄金帧集上与参考引擎结果一致（IoU / 类别 / 置信度容差）。
- 快路径与自己的 canonical 路径一致。
- 在 `Capabilities` 里声明 device / quant / layouts / dynamic_shape / input_accepts / releases_gil / load_holds_gil / io_binding。
- 缺依赖时抛 `E_ENGINE_UNAVAILABLE`，不静默变 stub。
- 加载耗时与是否持 GIL 已标注（供上层决定加载时机）。
- io 映射在引擎适配器内完成，wrapper 代码里不出现引擎物理名。
- 取输出必须用 `extract(name, 1)`（引用），不得深拷贝。
- 归一化换算必须复用 `ops.py` 的唯一实现，不得在适配器里另写一份。
