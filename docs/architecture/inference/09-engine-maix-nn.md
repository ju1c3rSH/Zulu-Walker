> 本节目标：固定 maix.nn 端到端特例路径的工件、加载、io、预处理归属、已知坑与验收要求。

# 09 引擎适配：maix.nn

来源：后端文档 §4.1 maix.nn（MaixCAM1 / MaixCAM2），并引 §1 / §2 / §3、架构文档 §4.8 与附录 C.3。

下表是源文档 §4.1  的原始适配清单；表内各行内容分别落到本文件第 1–10 节对应标题下。

| 项 | 内容 |
|---|---|
| 工件 | `.mud`（描述）加 `.axmodel` / `.cvimodel`（权重） |
| 加载 | `maix.nn.YOLO11(path, dual_buff=True)` 等 slot 类 |
| 模型族 | 从 MUD 的 `[extra] model_type` 读（已实现在 `maixcam2/ai.py`） |
| io | 隐式；输出已是解码后的对象 |
| 快路径 | 原生 `maix.image.Image` |
| 坑 | 模型类必须与模型族匹配，否则后处理静默出错（已修为显式失败） |
| GIL | [待测] |

## 1. 标识与设备

- 引擎 id 必须是 `maix_nn`（注册表键，见 §7.1（04-runtime-and-engine-selection.md））。
- 设备：NPU。
- 适用平台：MaixCAM1 / MaixCAM2。
- Python 绑定：maix.nn。
- 量化：int8（厂商工具链）。
- 设备能力声明必须落在 `framework/hal/platforms/<p>/capabilities.py` 的 `available_engines` 中（见 §2（01-scope-and-boundaries.md））。
- 该引擎在 `maixcam2` 上可跑、在 `linux` 上能力声明会拦住（见 §1（01-scope-and-boundaries.md））。

## 2. 工件形态与依赖

- 工件形态：`.mud`（描述）加 `.axmodel` / `.cvimodel`（权重）。
- 依赖：`maix.nn`；它是平台库，只有 `framework/inference/engines/` 允许 import（见 §1.1（01-scope-and-boundaries.md））。
- manifest 的工件声明示例（对应 §4 io 与 §7 预处理归属）：

```yaml
maix_nn:
  file: yolo11n.mud
  wrapper: "vision.det.end2end"
  wrapper_params: {}
  preprocess: { resize: baked, norm: baked }
  io:
    inputs: { image: "image" }
    outputs: { objects: "objects" }
```

## 3. 加载

- 加载入口：`maix.nn.YOLO11(path, dual_buff=True)` 等 slot 类。
- 模型族必须从 MUD 的 `[extra] model_type` 读（已实现在 `maixcam2/ai.py`）。
- 模型类必须与模型族匹配，否则后处理会静默出错。
  理由：静默出错不可接受；该行为已修为显式失败。
- 加载耗时：200–500ms（flash 重载）。
  真机实测（MaixCAM2，YOLO11n 640×480）：**15–25ms**，比估计低一个数量级；预载的价值因此不在"加载慢"，而在"加载期持 GIL 冻结所有 Python 线程"。
- 加载期间持 GIL（pybind11）。
- 端到端推理耗时（真机实测，YOLO11n 640×480，同一张帧重复 30 次）：中位 **8.6 ms**（原生帧）/ **9.6 ms**（numpy 路径），约 115 FPS。
  理由：numpy 路径与原生帧路径的差距只有 1 ms，说明 `cv2image` 的转换不是瓶颈；但这不代表本项目 1280×352 输入的预算（见 §10.3（06-performance-and-equivalence.md））。
- 预载价值：高——切换最慢，最该预载；且加载期持 GIL 会冻结所有 Python 线程，必须在启动期完成（见 §6.2.1（04-runtime-and-engine-selection.md））。
- 多实例共存受加速器内存池限制。
- 加载失败的错误码：`E_MODEL_LOAD_FAILED`（见 §6（08-backend-contracts.md））。

## 4. io 寻址

- io 寻址是隐式的（单模型）。
- 输出已是解码后的对象，引擎没有物理名；逻辑名仍由 manifest 显式声明：`io.inputs: { image: "image" }`、`io.outputs: { objects: "objects" }`，逻辑名与物理名相同。
- 引擎适配器不得自述物理名：没有可自述的名字；适配器按 manifest 声明的逻辑名给输出命名。
- wrapper 取 `artifact.io.outputs` 的第一个键作为主输出，不得写死名字（§4.5，见 §4.5（02-contracts-and-dataflow.md））。
- 引擎适配器必须接受运行选项，并把其中的阈值转交给 `maix.nn.detect()`（§3.2，见 §3.2（02-contracts-and-dataflow.md））。
  理由：`maix.nn.detect()` 的默认阈值是 `conf_th=0.5` / `iou_th=0.45`，与现役的 `0.3` / `0.4` 不同；不显式下传就会少一个框。
- `output_specs()` 的具体返回内容：待补（源文档未给出 maix.nn 的 `TensorSpec` 列表）。

## 5. 输入与快路径

- 快路径：直接吃 `maix.image.Image`；[源码] 已存在。
- **输入图像的格式标记必须是 `FMT_RGB888`**：引擎会校验格式，拿到 `FMT_BGR888` 会直接抛 "image format not match, input_type: RGB888"（真机实测）。
  理由：numpy 路径必须把 BGR 帧翻转成 RGB 再用 `cv2image(..., bgr=False)` 构造，否则颜色序反了、模型结果也不对。
- 快路径形态必须进入 `Capabilities.input_accepts`：`("native_image", "numpy_bgr")`（见 §3.3（02-contracts-and-dataflow.md））。
- 规范预处理整段是否旁路由工件的 `preprocess.resize` / `preprocess.norm` 是否为 `baked` 决定，与帧来源无关（见 §4.7（02-contracts-and-dataflow.md））。
  理由：判据挂在工件上，原生帧与 numpy 两条路行为一致，框架不插手 baked 工件的预处理。
- facade 必须按 `input_accepts` 与 `camera.last_raw` 是否为 `None` 决定给原生帧还是 numpy；相机由 `detect()` 的 `camera` 入参提供。

## 6. 输出形态

- 输出形态：end2end——已解码的对象列表，引擎内置了后处理。
- wrapper 只做格式转换，不得重复做后处理。
  理由：对已带 NMS 的图再跑一次 NMS 会打乱框且不报错。
- 阈值口径按平台不同：maixcam2 现役 `conf_th=0.3` / `iou_th=0.4`，maixcam1 现役 `0.48` / `0.5`；端到端方言的阈值由引擎侧执行，迁移时必须按平台对齐，不得一刀切。
- 引擎适配器必须调 `maix.nn.YOLO11.detect()`，把 `maix.nn.Object` 封装后交回 wrapper。
- 阈值必须由方言下传给引擎，映射之后按 `score >= conf_th` 再过滤一次；两处都不能省（§4.8，见 §4.8（02-contracts-and-dataflow.md））。
- wrapper 必须把结果转成框架的 `Detection`，不再跑 NMS。
- 该方言的 wrapper 声明：`wrapper: "vision.det.end2end"`，`provides: {"detect"}`，`output_form: "end2end"`，`families: {"yolo11", "yolov5"}`（见 §3.4（02-contracts-and-dataflow.md））。
  理由：maixcam1 的 `.cvimodel` 是 YOLOv5，同样走端到端形态。
- "把已解码的对象转成原始张量不现实，所以 maix.nn 走端到端特例路径，框架对它不做后处理，wrapper 直接消费引擎的解码结果并转成 `Detection`"。
  理由：这是"引擎自带后处理"时 wrapper 的样板——任务接口照常暴露，只是解释输出这一步换成直接吃引擎的成品。

## 7. 预处理归属

- `preprocess` 必须全为 `baked`：`{ resize: baked, norm: baked }`。
- 前处理阶段对 maix.nn 不做 letterbox 也不做归一化，直接把原生帧交给引擎。
- `baked` 档禁止出现 `formula`（见 §4.3（02-contracts-and-dataflow.md））。
- **`baked` 不等于零成本**：axmodel 把预处理集成为第一层 CPU 算子，resize 与归一化仍在设备 CPU 上执行（转换器文档："对于 YOLO11，因为模型集成了预处理，也就是说第一层是 CPU 处理"）。端到端省掉的是 Python 侧的转置与 NMS，不是 CPU 侧的预处理本身。
- **mud 的四个 cache 属性直接决定每帧 DDR 读写量，框架不得改写**：`input_cache` / `output_cache`（输入输出是否走缓存内存，多次读输出时更快）、`input_cache_flush`（跑模型前是否把 cache 刷到 DDR；首层是 NPU 算子时必须为 `true`，YOLO11 因首层是 CPU 处理故为 `false`）、`output_cache_inval`（跑完后是否把输出 buffer 置无效以保证直接从 DDR 读；末层是 CPU 算子时可置 `false` 以减少耗时）。
  manifest 是否显式承载这四项：[待定]——当前 schema 无对应字段。
- `baked` 是最危险的一档，框架无法验证它是否正确。

## 8. 已知坑

- 模型类与模型族不匹配时，后处理会静默出错。
  状态：已修为显式失败。
- 推理期间与加载期间是否持 GIL：持有（pybind11 默认）[待测]。
- mud 是否连 resize / letterbox 一起烘焙：**已确认是**。
  真机证据：mud 的 `[extra]` 段自带注释 "we already fused preprocess in axmodel, so mean and scale not used here"；引擎把任意尺寸的帧按 `Fit.FIT_CONTAIN` 缩放到模型输入尺寸（640×480）后直接送图。
  结论：`preprocess.resize` 与 `preprocess.norm` 都可以标 `baked`，框架不插手。
- **加载后的第一次推理不返回任何检测**（真机实测，YOLO11n 640×480）：第一帧返回空，第二帧起结果稳定；**零帧预热不能清除该状态**——预热帧之后的那一帧仍然是空的。
  后果：`warmup: true` 必须用一张真实帧，零帧预热等于没预热。
- 首次 `detect()` 有额外的 NPU 首次调度与内存页分配开销；预热（`warmup: true`）与预载是两件事，默认关闭。
- end2end 路径不得重复做 NMS。

## 9. 等价性验证要求

- maix.nn 不吐原始张量，无法提供张量级 canonical 路径。
- 等价性必须在 `Detection` 层比对：同一张黄金帧，maix.nn 出的框与 onnxruntime 加 `vision_det` 出的框对比。
- 不要求 maix.nn 吐原始张量。
  理由：要求它吐张量会迫使 mud 走慢路径，得不偿失。
- 比对指标与容差按 §11 定稿（见 §11（06-performance-and-equivalence.md））。
- 快路径一致性要求对 maix.nn 以 §2 的 `Detection` 级对齐为准（见 §2（08-backend-contracts.md））。

## 10. 验收清单

- 加载失败抛 `E_MODEL_LOAD_FAILED`，不静默。
- 未知模型族抛 `E_MODEL_TYPE_UNSUPPORTED`，不兜底到通用类。
- 输出 shape 与 manifest 声明不符，抛 `E_OUTPUT_LAYOUT_MISMATCH`。
- canonical 路径在黄金帧集上与参考引擎结果一致（IoU / 类别 / 置信度容差）；maix.nn 走 `Detection` 级对齐。
- 在 `Capabilities` 里声明 device / quant / layouts / dynamic_shape / input_accepts / releases_gil / load_holds_gil / io_binding。
- 缺依赖时抛 `E_ENGINE_UNAVAILABLE`，不静默变 stub。
- 加载耗时与是否持 GIL 已标注（供上层决定加载时机）。
- io 映射在引擎适配器内完成，wrapper 代码里不出现引擎物理名。
- 不得出现 `Detection` 以外的模型族混入 `vision.det.end2end` 的 `families`（本方言 `families: {"yolo11", "yolov5"}`）。
