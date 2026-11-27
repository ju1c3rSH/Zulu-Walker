> 本节目标：固定 rknn-toolkit-lite 的离线构建、顺序寻址、烘焙归一化与版本比对要求，作为该引擎适配器的实现要求。

# 12 引擎适配：rknn-toolkit-lite

来源：后端文档 §4.4 rknn-toolkit-lite，并引 §1 / §2 / §3、架构文档 §5.2 / §4.3。

下表是源文档 §4.4  的原始适配清单；表内各行内容分别落到本文件第 1–10 节对应标题下。

| 项 | 内容 |
|---|---|
| 工件 | `.rknn`（二进制） |
| 构建 | 离线，x86 主机加 `rknn-toolkit2`：`config(mean_values, std_values, target_platform)` 到 `load_onnx` 到 `build(do_quantization, dataset)` 到 `export_rknn` [源码] |
| 运行 | 板端 `rknn-toolkit-lite2`（aarch64 wheel）加 `librknnrt.so` [源码] |
| io | 隐式，按顺序；多输出按顺序取，因此 manifest 的 io 只能用索引 |
| 输入 | NHWC float32；官方示例由用户自己 `cvtColor` 加 `resize` [源码] |
| 设备 | `init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)` [源码] |
| 快路径 | 无 |
| 坑 | runtime 版本必须与 toolkit 版本匹配，否则 "load failures or wrong results" [源码] |
| GIL | [待测] |

## 1. 标识与设备

- 引擎 id 必须是 `rknn_lite`（注册表键，见 §7.1（04-runtime-and-engine-selection.md））。
- 设备：NPU（`core_mask` 选核）。
- Python 绑定：C API 的 Python 包装。
- 量化：int8（厂商工具链）。
- 落地条件：延后，需 RK3588 硬件与工具链版本匹配（见 §13（07-migration-phases.md））。

## 2. 工件形态与依赖

- 工件形态：`.rknn`（二进制）。
- 依赖：板端 `rknn-toolkit-lite2`（aarch64 wheel）加 `librknnrt.so` [源码]。
- manifest 对应声明：`runtime: { package: "rknn-toolkit-lite2", version: "2.3.0" }`。
- `rknn_lite` 必须被隔离脚本单独列出。
  理由：正则里的 `rknn\b` 匹配不到 `import rknn_lite`，因为下划线不构成词边界（见 §1.2（01-scope-and-boundaries.md））。

## 3. 加载

- 构建是离线步骤：x86 主机加 `rknn-toolkit2`，流程为 `config(mean_values, std_values, target_platform)` → `load_onnx` → `build(do_quantization, dataset)` → `export_rknn` [源码]。
- 运行时初始化：`init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)` [源码]。
- 加载耗时：快 [推断]。
- 加载期间持 GIL（C API 包装）[待测]。
- 预载价值：低。
- 多实例共存：按 `core_mask` 分核。
- 版本比对必须发生在加载期：不符则抛 `E_MODEL_LOAD_FAILED`（见 §6（08-backend-contracts.md））。

## 4. io 寻址

- io 寻址是隐式的、按顺序的；多输出按顺序取。
- manifest 的 io 只能用索引：`inputs: { image: 0 }`、`outputs: { pred: 0 }`。
- 引擎适配器必须按 `artifact.io` 的索引取输出。
- 不得依赖 tensor 名或 blob 名自述。

## 5. 输入与快路径

- 快路径：无，本来就是 numpy NHWC；canonical 即快路径。
- 规范输入是 HWC，加一个 batch 维即 NHWC，不需要转置，只差一次 dtype 扩展。
- 输入要求：NHWC float32。
  理由：官方示例由用户自己 `cvtColor` 加 `resize` [源码]。
- `input_accepts` 必须是 `("numpy_bgr",)`（见 §3.3（02-contracts-and-dataflow.md））。

## 6. 输出形态

- 输出形态：raw——原始 tensor（按顺序）；后处理由框架做（wrapper 解码 + NMS）。
- 适配器只做搬运与寻址，不得认识 `Detection`（见 §3.2 硬规矩 1，见 §3.2（02-contracts-and-dataflow.md））。

## 7. 预处理归属

- 归一化已在转换期烘焙（`preprocess.norm: baked`）。
- 规范段送出去的是 uint8 像素域 `[0, 255]`，绝不能预先除 255。
  理由：否则被二次归一化且不报错。
- `resize` 归属：框架做（官方示例自己 `cv2.resize`）[源码]。
- manifest 对应声明：`preprocess: { resize: framework, norm: baked }`；`baked` 档禁止出现 `formula`。
- manifest 的 `build` 段必须记全 `mean_values` / `std_values`，框架据此校验工件声明的输入属性与配方一致，不一致直接抛错。

## 8. 已知坑

- runtime 版本必须与 toolkit 版本匹配，否则 "load failures or wrong results" [源码]。
  理由：不匹配的表现可能是错结果，比加载失败更危险。
- 规范段若预先除过 255，RKNN 会再除一次，数值差 255 倍且不报错。
- rknn runtime 与 toolkit 版本比对的可靠途径：[待测]——`get_sdk_version()` 能否拿到足够信息。
- 推理期间与加载期间是否持 GIL：持有（C API 包装）[待测]。

## 9. 等价性验证要求

- rknn_lite 属于张量级引擎：canonical 路径输出原始张量，必须比对张量与解码后框。
- 快路径与 canonical 是同一条路径（canonical 即快路径），无额外一致性要求。
- 等价性验证必须覆盖 resize mode。
  理由：不同 mode 产生的框位置不同，跨引擎比对前先确认两边走的是同一个 mode（见 §4.3.1（02-contracts-and-dataflow.md））。
- MaixCAM2 上 mud 的 int8 与 onnx fp32 的框偏差：[待测]，是现役模型的等价性基线。
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
- 版本比对必须在加载期完成，不符抛 `E_MODEL_LOAD_FAILED`。
- 规范段不得预先除 255。
- 另一条可选路径是 C API（`rknn_api.h`，支持 zero-copy input buffer），官方称生产环境显著更快 [源码]；Python 绑定适合原型，量产建议走 C API。

### 附：C API 零拷贝路径（可选）

- C API 入口：`rknn_api.h`，支持 zero-copy input buffer。
- 官方称该路径在生产环境显著更快 [源码]。
- 定位：Python 绑定适合原型，量产建议走 C API。
- 该路径的 io 寻址、版本比对与验收细则：待补（源文档未展开）。
