> 本节目标：收录改动点清单、术语表、wrapper 实现草案、引擎性能参考与风险未验证项，作为设计定稿的附录。

# 13 附录

来源：架构文档 附录 A 改动点清单、附录 B 术语、附录 C wrapper 实现草案；后端文档 §7 性能参考、§8 风险与未验证项。

## 附录 A 改动点清单

状态：`[待实现]`

下表固定本次重构涉及的每个文件与对应改动。

| 文件 | 改动 |
|---|---|
| `framework/inference/` | 新增（本文 §2） |
| `framework/hal/interface/ai.py` | 协议拆分为基础协议 + 可选能力协议；新增 4 个错误码与 `ManifestError`；`detect` / `segment` / `classify` 增参 `camera=None`；`Detection` 改 `slots=True` 的普通 dataclass（§10.1 规则 1） |
| `framework/hal/interface/__init__.py` | 新增：统一导出 AI 契约（错误码、异常类、Detection 等） |
| `framework/hal/platforms/<p>/ai.py` | 降级为平台侧薄适配：转调 `framework/inference/facade.py`，只保留平台帧形态与错误状态；`maix.nn` 的 `draw_seg_mask` 彩色叠加移出引擎路径（属显示层职责） |
| `framework/hal/platforms/<p>/capabilities.py` | 新增：平台可用引擎声明 |
| `framework/hal/platforms/<p>/camera.py` | 实现 `frame_spec()` 声明实际帧形态；原生帧取用规则（§4.1 / §4.7） |
| `modules/zw_opencv_module/processors/ai_inference_processor.py` | 删除 `_raw=True` 分支，改为透传 `camera=context.get("camera")` |
| `scripts/check_platform_isolation.py` | EXEMPT 改目录列表并加入 `inference/engines/`；FORBIDDEN 加入引擎库（含 `rknn_lite`、`llama_cpp`） |
| `deploy/maixcam/models/*/manifest.yaml` | 加 `family` / `input.frame` / `input.resize` / `preprocess.formula` 段 |
| `scripts/check_manifest.py` | 按 schema v3 补齐校验项（family / frame / resize / formula / build 段） |
| `project_config.yaml` | `ai.engine`（必填，单一引擎）/ `ai.preload_budget_ms` 段，模型条目加 `manifest`（精确路径）/ `preload` / `warmup` |

## 附录 B 术语

状态：`[已定]`

下表固定推理层各术语的唯一指代。

| 术语 | 指什么 |
|---|---|
| 平台 platform | 代码运行所在的机器与其硬件 API 集合 |
| 引擎 engine | 执行一张计算图的运行时 |
| 逻辑模型 | 语义层面的"一个模型"，对应一份 manifest |
| 工件 artifact | 被某个引擎加载的具体模型文件（可含预处理与后处理） |
| 方言 dialect | 输出怎么解释，由工件上的 wrapper 键标识；键不承载语义 |
| 模型族 family | 逻辑模型所属的模型族（`yolo11` 等），逻辑模型级声明，与工件自述交叉校验 |
| 能力集 provides | wrapper 产出的能力对象声明的任务能力集合，facade 据此代理 |
| 规范段 preprocess | runtime 里的规范预处理阶段，产出 `PreprocessResult`（inputs + transform） |
| 几何变换 GeometryTransform | 原帧坐标与输入图像素坐标之间的映射（mode / scale / offset），wrapper 反算坐标的依据 |
| wrapper | `build(runtime) -> 领域 API` 的函数，可内置可自行实现；任务适配层，把底层通用的 Engine 适配成某个具体任务的接口，不是 Python 装饰器 |
| 引擎适配器 engine adapter | `framework/inference/engines/<e>.py`；把命名字典搬运成引擎原生格式并取回原生输出 |
| 规范输入 canonical | 规范段的产出（HWC，帧形态与分辨率已处理；数值域随 `norm` owner 变） |
| io 映射 | 逻辑名到引擎物理名 / 索引的映射，声明在工件上 |
| manifest | 描述工件与其契约的声明文件 |
| runtime | engine + artifact + wrapper 的组合单元 |

## 附录 C wrapper 实现草案

状态：`[待实现]`

- wrapper 的契约只有一句：`build(runtime) -> 领域 API`。
- 本附录给出内置 `vision_det` 的形状，以及引擎适配器怎么把逻辑名翻译成物理名。

### C.1 内置 wrapper：`vision.det.raw`

状态：`[待实现]`

```python
# framework/inference/wrappers/vision_det.py
from __future__ import annotations

import numpy as np

from framework.inference import ops
from framework.inference.runtime import ModelRuntime


class RawDetAPI:
    """raw 形态的检测接口：自己解码、自己做 NMS。"""

    provides = frozenset({"detect"})

    def __init__(self, rt: ModelRuntime) -> None:
        self._rt = rt
        self._main_out = next(iter(rt.artifact.io.outputs))   # 主输出名由 manifest 定
        # 阈值默认取现役行为，manifest 可用 wrapper_params 覆盖
        self._conf_th = float(rt.params.get("conf_th", 0.3))
        self._iou_th = float(rt.params.get("iou_th", 0.4))

    def detect(self, frame: np.ndarray, **kw) -> list:
        conf_th = kw.pop("conf_th", self._conf_th)
        iou_th = kw.pop("iou_th", self._iou_th)
        pre = self._rt.preprocess(frame)        # 前处理：模型+引擎的契约（§4.2）
        out = self._rt.engine.run(pre.inputs)   # 推理：引擎只跑一步（§4.4）
        pred = out[self._main_out]              # 只用逻辑名
        return ops.decode_yolo(pred, self._rt.artifact, pre.transform,
                               conf_th, iou_th, **kw)   # 后处理：方言


def build_raw(rt: ModelRuntime) -> RawDetAPI:
    return RawDetAPI(rt)
```

要点：

- `rt.engine.run({"image": blob})` 里只有逻辑名。物理名（`"data"`、`"output"`）由适配器按 `artifact.io` 翻译，wrapper 不知道引擎是什么。
- `rt.preprocess(frame)` 负责前处理（帧形态转换、分辨率处理、归一化），wrapper 不碰；`ops.decode_yolo` 负责后处理，两者都在 `ops.py` 里，是全框架唯一一份实现（§1.1，见 §1.1（01-scope-and-boundaries.md））。
- 框的反算必须用 `pre.transform`，不得用 manifest 的静态 `input.size`。
  理由：缩放比与偏移随原帧分辨率变（§4.2，见 §4.2（02-contracts-and-dataflow.md））。
- 解码必须全部向量化：置信度筛选、topk、NMS 都在 numpy 里，Python 只做最后的对象包装（§10.1 规则 3，见 §10.1（06-performance-and-equivalence.md））。
- `Detection` 用 `slots=True` 的普通 dataclass，不用 `frozen`（§10.1 规则 1，见 §10.1（06-performance-and-equivalence.md））。
- **阈值必须与现役对齐**：默认 `conf_th=0.3` / `iou_th=0.4`（不是 0.25 / 0.45），并且 raw 路径要复刻现役那道后置过滤 `score >= conf_th`。
  理由：现役 maix.nn 不完全遵守 conf_th，raw 路径是自己做筛选，语义必须一致，否则 Phase 1 的"逐帧一致"不成立。

### C.2 引擎适配器：逻辑名到物理名

状态：`[待实现]`

```python
# framework/inference/engines/ncnn.py
class NcnnEngine:
    engine_id = "ncnn"

    def load(self, artifact, **opts) -> None:
        self._net = ncnn.Net()
        self._net.load_param(artifact.param_path)
        self._net.load_model(artifact.bin_path)
        self._in_name = artifact.io.inputs["image"]    # 逻辑名 -> blob 名
        self._out_name = next(iter(artifact.io.outputs.values()))   # 主输出
        if artifact.preprocess.norm == "engine":
            self._mean, self._std = convert_norm_params(artifact)   # yolo -> 引擎式

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        blob = inputs["image"]                          # HWC，规范段产出
        mat = ncnn.Mat(blob)                            # 零拷贝，不转置
        if self._artifact.preprocess.norm == "engine":
            mat.substract_mean_normalize(self._mean, self._std)
        self._net.input(self._in_name, mat)
        ret = self._net.extract(self._out_name, 1)      # 1 = 引用，不是深拷贝
        return {"pred": np.asarray(ret)}                # 逻辑名 -> 张量
```

要点：

- 规范输入必须是 HWC，ncnn 正好要 HWC，**一次转置都不需要**。若规范输入定成 NCHW，这里每次都要转置（§10.1，见 §10.1（06-performance-and-equivalence.md））。
- `norm: engine` 档规范段送的是 uint8 像素域，`Mat` 构造时自动扩展成 float32，再调 `substract_mean_normalize`；参数由 `input.norm` 的 yolo 形式换算（`mean_e = mean / scale`，`std_e = std / scale`），换算在 `ops.py` 里，是全框架唯一一份。
- `extract(name, 1)` 取的是引用。取深拷贝（`0`）会让每帧多一次输出张量的内存拷贝，在 A53 上是几毫秒（§10.1 规则 2，见 §10.1（06-performance-and-equivalence.md））。
- 适配器只做搬运与寻址，不得认识 `Detection`（§3.2 硬规矩 1，见 §3.2（02-contracts-and-dataflow.md））。

### C.3 maix.nn 的 end2end wrapper

状态：`[待实现]`

```python
# framework/inference/wrappers/vision_det.py
class End2endDetAPI:
    """maix.nn 已经把解码和 NMS 做完，这里只做格式转换。"""

    provides = frozenset({"detect"})

    def __init__(self, rt: ModelRuntime) -> None:
        self._rt = rt
        self._main_out = next(iter(rt.artifact.io.outputs))
        self._conf_th = float(rt.params.get("conf_th", 0.3))
        self._iou_th = float(rt.params.get("iou_th", 0.4))

    def detect(self, frame, **kw) -> list:
        conf_th = float(kw.pop("conf_th", self._conf_th))
        iou_th = float(kw.pop("iou_th", self._iou_th))
        objects = self._rt.engine.run(
            {"image": frame}, conf_th=conf_th, iou_th=iou_th
        )[self._main_out]
        return [_to_detection(o) for o in objects if o.score >= conf_th]


def build_end2end(rt: ModelRuntime) -> End2endDetAPI:
    return End2endDetAPI(rt)
```

- 差异只有两处：`rt.engine.run` 收到的是帧而不是规范张量（baked 工件的规范段整段旁路，§4.7，见 §4.7（02-contracts-and-dataflow.md）），输出的是已解码对象而不是原始张量。
- 输出逻辑名取 `artifact.io.outputs` 的第一个键，与 raw 方言同一套规则（§4.5，见 §4.5（02-contracts-and-dataflow.md））。
- 阈值必须交给引擎并在映射后再过滤一次，两处都不能省（§4.8，见 §4.8（02-contracts-and-dataflow.md））。
- wrapper 对上层的接口必须完全一样。
  理由：这正是"任务适配层"的意义。

## §7 性能参考（后端文档）

状态：`[已定]`（设备真值 `[待测]`）

下表固定各引擎的性能参考数据与来源，用于判断量级。

| 引擎 | 场景 | 数据 | 来源 |
|---|---|---|---|
| maix.nn | YOLO11n 640×640 | 113 FPS（标称） | Sipeed 官方 wiki |
| maix.nn | 本项目输入 1280×352 | 待测（面积约为 640×640 的 1.1 倍） | Phase -1 实测 |
| ncnn | YOLO11n，RPi 4（A72 @1.8 GHz） | 409 ms/帧（2.4 FPS） | 第三方实测 |
| ncnn | YOLO11n，RPi 5（A76 @2.4 GHz） | 63.4 ms 纯推理 / 67.2 ms 含预处理（14.9 FPS） | 同上 |
| ncnn | int8 vs fp32，RPi 5 | 仅快约 4%（单线程），多线程反而更慢 | 同上 |
| onnxruntime | YOLO11n，RPi 5 | 331 ms/帧（3.0 FPS） | 同上 |

- 板级推论：MaixCAM2 的 A53 @1.2 GHz 比上表里最弱的 RPi 4（A72 @1.8 GHz）还弱一档，ncnn 跑 YOLO11n 预计 150–250 ms/帧，即 4–7 FPS。
  理由：与 NPU 的差距在一个数量级以上，选 ncnn 的理由只能是"对比调试"或"NPU 不支持该算子"。
- 这些数字都来自别的板子，只能当量级参考。
- 设备上的真值必须用 `scripts/bench_inference_ops.py` 与 `scripts/bench_python_overhead.py` 在板子上重测（见 §10.4（06-performance-and-equivalence.md））。

## §8 风险与未验证项（后端文档）

状态：`[待测]`

下表固定尚未验证的风险项及其影响范围。

| 项 | 状态 |
|---|---|
| ncnn / maix.nn / rknn 推理期间是否持 GIL | [待测]，决定 §5.3 的对策选哪条 |
| ncnn int8 量化表与 ORT int8 的精度差 | [待测]，影响等价性容差 |
| rknn runtime 与 toolkit 版本比对的可靠途径 | [待测]，`get_sdk_version()` 能否拿到足够信息 |
| ncnn `Mat.from_pixels` 快路径与 canonical 的逐像素一致性 | [待测]，通道序与插值细节 |
| MaixCAM2 上 mud 的 int8 与 onnx fp32 的框偏差 | [待测]，现役模型的等价性基线 |
| mud 是否连 resize / letterbox 一起烘焙 | [待测]，决定 `preprocess.resize` 能否标 `baked` |

- 这些必须上机实测，不能靠文档推断。
- 测试用例必须写进 `tests/`。
