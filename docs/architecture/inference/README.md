> 本节目标：给出推理层设计文档的阅读顺序、文件清单与全局承重不变量速查，作为 `docs/architecture/inference/` 的入口索引。

# 推理层设计文档（拆分版）

源文档为两份定稿稿：`docs/architecture/inference_architecture.md`（架构定稿，§1–§14 + 附录 A/B/C）与 `docs/architecture/inference_backends.md`（引擎接入，§0–§8）。本目录是二者的拆分版本，技术内容与源文档一致，只做两件事：按小节拆成独立文件，并把行文改写为 plan 陈述语气。

状态标注约定：

- `[已定]` — 已定稿结论。
- `[待实现]` — 设计已定，代码尚未落地。
- `[待测]` — 需上机或专项验证后才能确认。

## 阅读顺序

| 步 | 文件 | 读完应当掌握 |
|---|---|---|
| 1 | `01-scope-and-boundaries.md` | 推理层的职责边界、目标目录结构与"明确不做"清单 |
| 2 | `02-contracts-and-dataflow.md` | 数据结构、引擎协议、wrapper 机制与数据流契约 |
| 3 | `03-manifest.md` | manifest schema v3、字段校验与存储布局 |
| 4 | `04-runtime-and-engine-selection.md` | 运行时对象、生命周期、预载语义与引擎选择规则 |
| 5 | `05-errors-and-observability.md` | 错误码体系、异常映射与统计口径 |
| 6 | `06-performance-and-equivalence.md` | 性能预算、算子开销与等价性验证口径 |
| 7 | `07-migration-phases.md` | 对外契约改动与分阶段落地计划 |
| 8 | `08-backend-contracts.md` | 引擎适配器的统一内部形式、双路径、差异矩阵、GIL 事实与接入验收标准 |
| 9 | `09-engine-maix-nn.md` … `12-engine-rknn-lite.md` | 四个引擎各自的工件、加载、io、预处理归属、已知坑与验收清单 |
| 10 | `13-appendix.md` | 改动点清单、术语表、wrapper 实现草案与性能参考 |

## 文件清单

下表列出本目录全部文件、其收编的源章节与一句话职责。

| 文件 | 收编的源章节 | 职责 |
|---|---|---|
| `README.md` | 新写 | 索引：阅读顺序、文件清单与全局不变量 |
| `01-scope-and-boundaries.md` | 架构文档 §1、§2、§14 | 固定推理层的范围、依赖边界与目录结构 |
| `02-contracts-and-dataflow.md` | 架构文档 §3、§4 | 定义核心数据结构、引擎协议与三层数据流契约 |
| `03-manifest.md` | 架构文档 §5 | 定义 manifest schema v3 与工件声明的校验规则 |
| `04-runtime-and-engine-selection.md` | 架构文档 §6、§7 | 定义运行时对象生命周期、预载语义与引擎注册选择 |
| `05-errors-and-observability.md` | 架构文档 §8、§9 | 定义错误码体系、异常映射与统计暴露 |
| `06-performance-and-equivalence.md` | 架构文档 §10、§11 | 固定性能预算、算子开销与等价性验证口径 |
| `07-migration-phases.md` | 架构文档 §12、§13 | 固定对外契约改动与分阶段落地计划 |
| `08-backend-contracts.md` | 后端文档 §0、§1、§2、§3、§5、§6 | 固定引擎适配器的统一形式、双路径、差异矩阵与验收标准 |
| `09-engine-maix-nn.md` | 后端文档 §4.1 | 固定 maix.nn 端到端特例路径的接入要求 |
| `10-engine-ncnn.md` | 后端文档 §4.2 | 固定 ncnn 的工件、加载、io 与归一化适配要求 |
| `11-engine-onnxruntime.md` | 后端文档 §4.3 | 固定 onnxruntime 的参考实现路径要求 |
| `12-engine-rknn-lite.md` | 后端文档 §4.4 | 固定 rknn-toolkit-lite 的烘焙归一化与索引寻址要求 |
| `13-appendix.md` | 架构文档 附录 A/B/C；后端文档 §7、§8 | 收录改动点清单、术语表、wrapper 实现草案与性能参考、风险清单 |

## 全局不变量（承重约束速查）

以下每条均为承重约束，违反其一即视为设计偏离。

1. **三轴分离**：平台、引擎、工件互相独立；平台不等于引擎。〔§1，见 §1（01-scope-and-boundaries.md）〕
2. **依赖边界按库的性质划**：`framework/inference/engines/` 是唯一允许 import 第三方推理库的目录；`ops.py` 只用 `numpy` 与 `cv2`，不得 import 平台库或引擎库。〔§1.1，见 §1.1（01-scope-and-boundaries.md）〕
3. **通用算子只有一份**：letterbox、归一化、NMS、IoU、softmax 的实现只存在于 `ops.py`。〔§1.1、§14，见 §1.1 与 §14（01-scope-and-boundaries.md）〕
4. **引擎只跑一步**：命名字典进、命名字典出；引擎不做后处理，不认识 `Detection`，不认识任务类型。〔§3.2，见 §3.2（02-contracts-and-dataflow.md）〕
5. **wrapper 不做输入准备**：wrapper 只调 `rt.preprocess()`；缩放与归一化由"模型 + 引擎"契约决定，属于 runtime。〔§3.4，见 §3.4（02-contracts-and-dataflow.md）〕
6. **推理层产出边界是 `list[Detection]`**：推理层不认识 `VisionResult`；组装 `VisionResult` 属 modules 层。〔§4.6，见 §4.6（02-contracts-and-dataflow.md）〕
7. **归一化三档 + 公式两值**：`norm` 必为 `framework` / `engine` / `baked` 之一；`formula` 必为 `yolo` / `substract_mean_normalize` 之一；`norm: baked` 时禁止出现 `formula`。〔§4.3，见 §4.3（02-contracts-and-dataflow.md）〕
8. **分辨率三 mode**：`letterbox` / `crop` / `stretch`，三者共用 `ops.py` 的同一套原语；等价性验证必须覆盖 mode。〔§4.3.1，见 §4.3.1（02-contracts-and-dataflow.md）〕
9. **io 只用逻辑名**：wrapper 侧只出现逻辑名 `image` 与逻辑输出名；物理名只出现在 manifest 与引擎适配器里。〔§4.5，见 §4.5（02-contracts-and-dataflow.md）〕
10. **运行期只有一个引擎**：`engine` 必填，不做 `auto`、不做 `prefer` 候选链、不静默降级成 stub。〔§7.2，见 §7.2（04-runtime-and-engine-selection.md）〕
11. **加载期必须校验完**：任何不匹配都在启动期抛错，不拖到推理时才发现。〔§7.3，见 §7.3（04-runtime-and-engine-selection.md）〕
12. **配置/契约类硬失败**：抛异常；运行期单帧类计数并降级继续，状态由 `last_error_code` / `error_count` / `degraded` 暴露。〔§8，见 §8（05-errors-and-observability.md）〕
13. **核心层不认识任务语义**：核心层不出现"框""图像""token"这类词。〔§14，见 §14（01-scope-and-boundaries.md）〕
14. **构建离线**：模型转换工具链不进运行时框架，也不进仓库；设备只加载工件。〔§1.1、§2.1、§14，见 §1.1 与 §2.1（01-scope-and-boundaries.md）〕
