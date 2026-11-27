> 本节目标：固定推理层的错误码体系、异常映射规则与统计暴露口径，使失败可分类、可观测。

# 05 错误码与可观测性

来源：架构文档 §8 错误码、§9 可观测性。整体状态：`[已定]`。

## §8 错误码

状态：`[已定]`

- 沿用 `framework/hal/interface/ai.py` 已有体系，新增四个错误码。

下表固定新增错误码的异常类、触发时机与行为。

| 错误码 | 异常类 | 触发时机 | 行为 |
|---|---|---|---|
| `E_ENGINE_UNAVAILABLE` | `ModelConfigError` | 启动期选引擎 | 抛，硬失败 |
| `E_CONTRACT_MISSING` | `ManifestError`（新增） | manifest 缺必需字段 / 缺该引擎的工件声明 / 工件文件名与配置对不上 | 抛，硬失败 |
| `E_DIALECT_UNSUPPORTED` | `ModelConfigError` | wrapper 无法解析 | 抛，硬失败 |
| `E_SWITCH_TIMEOUT` | `AIError` | 切换等待超时 | 抛，硬失败 |

已有错误码必须保持不变：`E_MODEL_NOT_LOADED` / `E_MODEL_TYPE_UNSUPPORTED` / `E_MODEL_LOAD_FAILED` / `E_OUTPUT_LAYOUT_MISMATCH` / `E_INFER_FAILED` / `E_INVALID_INPUT`。

- `E_CONTRACT_MISSING` 必须用新增的 `ManifestError`，不得复用 `ContractError`。
  理由：`ContractError` 的语义是"输出与声明的布局不符"，用它承载"manifest 缺字段"会让两类问题混在一起。
- 分级不变：配置 / 契约类硬失败抛异常；运行期单帧类计数加降级继续，状态通过 `last_error_code` / `error_count` / `degraded` 可见。

错误码总表（10 个）：

| # | 错误码 | 来源 | 异常类 | 触发时机 | 行为 |
|---|---|---|---|---|---|
| 1 | `E_MODEL_NOT_LOADED` | 已有 | 待补 | 待补 | 待补 |
| 2 | `E_MODEL_TYPE_UNSUPPORTED` | 已有 | 待补 | 待补 | 待补 |
| 3 | `E_MODEL_LOAD_FAILED` | 已有 | 待补 | 待补 | 待补 |
| 4 | `E_OUTPUT_LAYOUT_MISMATCH` | 已有 | 待补 | 待补 | 待补 |
| 5 | `E_INFER_FAILED` | 已有 | 待补 | 待补 | 待补 |
| 6 | `E_INVALID_INPUT` | 已有 | 待补 | 待补 | 待补 |
| 7 | `E_ENGINE_UNAVAILABLE` | 新增 | `ModelConfigError` | 启动期选引擎 | 抛，硬失败 |
| 8 | `E_CONTRACT_MISSING` | 新增 | `ManifestError` | manifest 缺必需字段 / 缺该引擎的工件声明 / 工件文件名与配置对不上 | 抛，硬失败 |
| 9 | `E_DIALECT_UNSUPPORTED` | 新增 | `ModelConfigError` | wrapper 无法解析 | 抛，硬失败 |
| 10 | `E_SWITCH_TIMEOUT` | 新增 | `AIError` | 切换等待超时 | 抛，硬失败 |

待补说明：源文档只逐条给出了四个新增错误码的异常类、触发时机与行为；六个已有错误码的异常类与分级未在源文档中列出，本表相应单元格记为「待补」，不在本次重构中自行裁决。

## §9 可观测性

状态：`[已定]`

- 统计必须上移到 facade，按 `(engine, model)` 维度：

```python
@dataclass
class EngineStats:
    engine_id: str
    model_name: str
    infer_count: int = 0
    infer_avg_ms: float = 0.0
    infer_fps: float = 0.0
    error_count: int = 0
    last_error_code: str = ""
    load_ms: float = 0.0
```

- 现役 `maixcam2` 后端的私有字段 `infer_fps` / `infer_avg_ms` 必须迁到这里，对外形状不变。
- facade 必须额外暴露 `stats_by_engine()`。
