> 本节目标：固定推理层对外契约的改动面、modules 层的改动面与分阶段落地顺序，作为实施计划。

# 07 对外契约与迁移

来源：架构文档 §12 对外契约与迁移、§13 分阶段落地。整体状态：`[已定]`。

编号说明：本章三个子节已按上级 §12 重编为 §12.1 / §12.2 / §12.3，跨文件引用同步更新。

## §12 对外契约与迁移

状态：`[已定]`

### §12.1 协议拆分

状态：`[已定]`

现役 `AIInference` Protocol 把检测与分割 / 分类绑在一起：

```python
class AIInference(Protocol):
    # registry 与状态属性
    models / active_model / model_type / model_family / loaded / input_width / input_height
    labels / model_path / last_error_code / error_count / degraded
    # 生命周期
    add / remove / switch / load / unload
    # 上下文管理
    __enter__ / __exit__
    # 推理
    detect / segment / classify / get_mask
```

- wrapper 机制只预置检测（§3.4，见 §3.4（02-contracts-and-dataflow.md））。
- 若 facade 只实现 `detect()`，它就不满足这个 Protocol，`isinstance(facade, AIInference)` 在运行时直接失败（`@runtime_checkable` 按属性存在性检查）。
- 定法：拆协议。
  - `AIInference`（基础协议）：registry 与状态属性 + 生命周期 + `detect()`。
  - `AIInferenceSegment` / `AIInferenceClassify` / `AIInferenceMask`（可选能力协议）：`segment()` / `classify()` / `get_mask()`。
- facade 必须实现基础协议，并额外提供 `segment` / `classify` / `get_mask` 三个方法：active wrapper 产出的能力对象的 `provides` 集合含对应能力就代理，否则抛 `E_DIALECT_UNSUPPORTED`。
- 不得返回空列表，不得静默降级。
- 能力判定必须读 `provides`（§3.4，见 §3.4（02-contracts-and-dataflow.md）），不得靠 `hasattr` 猜。
- `model_family` 升入基础协议：值来自 manifest 的 `family` 字段，不再从模型文件反查。
  理由：模型族已经是 manifest 的一等字段（§5.2，见 §5.2（03-manifest.md）），协议层不该再猜。
- `infer_timestamp_ns` 不进协议：它属于可观测性面，由 facade 暴露并计入 `EngineStats`（§9，见 §9（05-errors-and-observability.md））。
- `DetectKwargs` / `ClassifyKwargs` 两个 TypedDict 是 `detect` / `segment` / `classify` 的 kwargs 契约，拆分后留在基础协议侧；`detect` 新增的 `camera` 入参与 `conf_th` / `iou_th` 一并由 `DetectKwargs` 承载。
- `add()` 必须接得下 manifest：现役签名 `add(nick_name, model_path, model_type="auto")` 放不下 `manifest:`，Phase 1 改为 `add(nick_name, model_path, manifest_path=None, model_type="auto")`；`manifest_path` 非空时必须与 `model_path` 指向同一逻辑模型（§5.4，见 §5.4（03-manifest.md））。
- 这条改动落在 `framework/hal/interface/ai.py`，是本次重构里唯一需要动既有对外协议的地方，属于一次性破坏性变更，必须在 Phase 1 完成。

### §12.2 对上层的改动

状态：`[已定]`

- `modules` 层只有一处必须改：`modules/zw_opencv_module/processors/ai_inference_processor.py` 里的 `_raw=True` 分支（帧适配下沉后删除，见 §4.7（02-contracts-and-dataflow.md））。
- 其余 modules 代码零改动。
  理由：业务层只用 `detect()`，不接触引擎、方言、io。

### §12.3 迁移阶段

状态：`[已定]`

见 §13。

## §13 分阶段落地

状态：`[已定]`

下表固定各 Phase 的内容与风险等级，作为实施排期依据。

| Phase | 内容 | 风险 |
|---|---|---|
| -1 | 引擎事实 Spike：装 onnxruntime 与 ncnn，同一份 onnx 跑通 canonical 路径并比对输出；实测各引擎 GIL 行为；把"待测"变成"已测" | 无（不改框架代码） |
| 0 | 引擎协议 + manifest 结构 + wrapper 机制骨架 + `ops.py`（numpy + cv2）+ 错误码新增（纯新增，不碰现役路径） | 无 |
| 1 | maixcam2 拆成 engine(maix.nn) + wrapper；`ai.py` 协议拆分；删除业务层 `_raw` 分支；`Detection` 改 `slots=True` 的普通 dataclass。行为与现役逐帧一致 | 低，需真机回归 |
| 2 | maixcam1 同样处理（注意 `.cvimodel`、YOLOv5 输出约定与 `0.48` / `0.5` 的阈值口径） | 低 |
| 3 | onnxruntime 后端 + 等价性验证（黄金帧集） | 中 |
| 4 | ncnn 后端 | 中 |
| 5 | rknn_lite 后端（延后，需 RK3588 硬件与工具链版本匹配） | 中 |

落地状态（2026-11）：

| Phase | 状态 | 交付物 |
|---|---|---|
| 0 | 已完成 | `framework/inference/`（contract / manifest / ops / registry / runtime / facade / wrappers / engines）、四个错误码与 `ManifestError`、隔离脚本改 AST 扫描、manifest 离线校验器 |
| 1 | 未开始 | 平台 `ai.py` 转调 facade、协议拆分、`Detection` 改 `slots=True`、删除业务层 `_raw` 分支 |

- 每个 Phase 结束时都必须能跑现有测试且真机行为不变。
- Phase 3 之前不得碰任何跨引擎等价性承诺。
  理由：那时还没有第二个引擎可比。
