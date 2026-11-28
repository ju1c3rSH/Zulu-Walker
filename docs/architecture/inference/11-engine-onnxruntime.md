> 本节目标：固定 onnxruntime 作为参考实现的适配要求：工件、加载、tensor 寻址、布局转置与已知坑。

# 11 引擎适配：onnxruntime

来源：后端文档 §4.3 onnxruntime，并引 §1 / §2 / §3。

下表是源文档 §4.3  的原始适配清单；表内各行内容分别落到本文件第 1–10 节对应标题下。

| 项 | 内容 |
|---|---|
| 工件 | `.onnx` |
| 加载 | `InferenceSession(path, sess_options, providers)` |
| io | 按 tensor 名；`get_inputs()` / `get_outputs()` 自述 [源码] |
| 设备 | `providers=["CPUExecutionProvider", "CUDAExecutionProvider", ...]` |
| 量化 | fp16（onnxruntime 转换）/ int8（QDQ、QOperator） |
| 快路径 | 无，canonical 即快路径 |
| GIL | `run()` 释放 [源码]；session 初始化不释放 [GitHub #27063] |
| 坑 | `run_with_iobinding()` 历史上不释放 GIL（#11246，PR #11248 修复），必须用普通 `run()`，或按版本验证 |

## 1. 标识与设备

- 引擎 id 必须是 `onnxruntime`（注册表键，见 §7.1（04-runtime-and-engine-selection.md））。
- 设备：CPU / CUDA / TensorRT EP。
- Python 绑定：pybind11。
- 量化：fp16（onnxruntime 转换）/ int8（QDQ、QOperator）。
- 角色定位：参考实现——其它引擎的等价性都以它为准。

## 2. 工件形态与依赖

- 工件形态：`.onnx`。
- manifest 对应声明：`file: yolo11n.onnx`、`wrapper: "vision.det.raw"`、`io: { inputs: { image: "images" }, outputs: { pred: "output0" } }`。
- 依赖：`onnxruntime`；只有 `framework/inference/engines/` 允许 import（见 §1.1（01-scope-and-boundaries.md））。

## 3. 加载

- 加载入口：`InferenceSession(path, sess_options, providers)`。
- 加载耗时：中；持 GIL [源码]。
- session 初始化不释放 GIL [GitHub #27063]。
- 预载价值：中。
- 多实例共存：可。
- 加载失败的错误码：`E_MODEL_LOAD_FAILED`（见 §6（08-backend-contracts.md））。

## 4. io 寻址

- io 按 tensor 名寻址。
- `get_inputs()` / `get_outputs()` 可自述 [源码]。
- 引擎适配器必须在 `load()` 时把 `artifact.io` 存下来，`run()` 时做双向翻译：逻辑名 `image` → tensor 名（`"images"`），逻辑名 `pred` → 输出 tensor 名（`"output0"`）。
- 多输出时按名取；主输出取 `artifact.io.outputs` 的第一个键。
- `io_binding` 在 `Capabilities` 中取值为 `"named"`。

## 5. 输入与快路径

- 快路径：无，本来就是 numpy；canonical 即快路径。
- 规范输入为 HWC，必须做一次 HWC→NCHW 转置。
  理由：onnxruntime 是四个引擎里唯一一个需要转置的引擎。
- 转置属于"布局选错的固定税"（见 §10.1（06-performance-and-equivalence.md））。
- `input_accepts` 必须是 `("numpy_bgr",)`（见 §3.3（02-contracts-and-dataflow.md））。

## 6. 输出形态

- 输出形态：raw——原始 tensor；后处理由框架做（wrapper 解码 + NMS）。
- 必须按名取输出。
- 适配器只做搬运与寻址，不得认识 `Detection`（见 §3.2 硬规矩 1，见 §3.2（02-contracts-and-dataflow.md））。

## 7. 预处理归属

- 归一化由框架的规范预处理段完成（`preprocess.norm: framework`）。
- 规范段产出已是 float32 归一化域，适配器只转 layout。
- `resize` 归属：框架做。
- manifest 对应声明：`preprocess: { resize: framework, norm: framework, formula: yolo }`。

## 8. 已知坑

- `run_with_iobinding()` 历史上不释放 GIL（#11246，PR #11248 修复）。
  约束：必须用普通 `run()`，或按版本验证。
- `io_binding` 历史有坑 [源码]，列为已知风险项。
- session 初始化不释放 GIL，会在加载期冻结所有 Python 线程；加载必须由主循环发起（见 §6.3（04-runtime-and-engine-selection.md））。

## 9. 等价性验证要求

- onnxruntime 是张量级引擎：canonical 路径输出原始张量，必须比对张量与解码后框。
- onnxruntime 路径是参考实现：其它引擎的等价性都以它为准。
- 快路径与 canonical 是同一条路径（canonical 即快路径），无额外一致性要求。
- 等价性验证必须覆盖 resize mode。
  理由：不同 mode 产生的框位置不同，跨引擎比对前先确认两边走的是同一个 mode（见 §4.3.1（02-contracts-and-dataflow.md））。
- ncnn int8 量化表与 ORT int8 的精度差：[待测]，影响等价性容差。
- 比对指标与容差按 §11 定稿（见 §11（06-performance-and-equivalence.md））。

## 10. 验收清单

- 加载失败抛 `E_MODEL_LOAD_FAILED`，不静默。
- 未知模型族抛 `E_MODEL_TYPE_UNSUPPORTED`，不兜底到通用类。
- 输出 shape 与 manifest 声明不符，抛 `E_OUTPUT_LAYOUT_MISMATCH`。
- canonical 路径在黄金帧集上与参考引擎结果一致（IoU / 类别 / 置信度容差）。
- 在 `Capabilities` 里声明 device / quant / layouts / dynamic_shape / input_accepts / releases_gil / load_holds_gil / io_binding。
- 缺依赖时抛 `E_ENGINE_UNAVAILABLE`，不静默变 stub。
- 加载耗时与是否持 GIL 已标注（供上层决定加载时机）。
- io 映射在引擎适配器内完成，wrapper 代码里不出现引擎物理名。
- 必须使用普通 `run()`，不得使用 `run_with_iobinding()`（或按版本验证后再用）。
