> 本节目标：固定推理层的性能预算、算子开销基线与跨引擎等价性验证口径，作为性能取舍的判定依据。

# 06 性能预算与等价性验证

来源：架构文档 §10 性能预算与语言开销、§11 等价性验证。整体状态：`[已定]`（数值待真机复测）。

## §10 性能预算与语言开销

状态：`[已定]`

- 推理层的 Python 胶水本身不是瓶颈；真正要盯的是它带来的数据搬运和对象分配。
- 下列数字由仓库里的两个基准脚本测出（x86 / Python 3.13.7 / numpy 2.5.1 / OpenCV 5.1.0）。
- 设备上的数字用同样的脚本在 A53 上重测过，两组都记在下面；**两组差一个数量级**，任何一条性能结论都必须以设备组为准。

### §10.1 语言特性开销（每帧）

状态：`[已定]`（x86 与 MaixCAM2 真机双组实测）

下表固定每帧语言构造开销的实测值与用途判定。

| 构造 | µs/op | 说明 |
|---|---|---|
| dict 字面量（2 项） | 0.055 | io 契约的每帧成本 |
| 函数 / 闭包调用 | 0.029 | wrapper 的 `detect` 调用 |
| kwargs 打包（2 项） | 0.040 | `conf_th` / `iou_th` 透传 |
| 锁 acquire + release | 0.058 | facade 的切换锁 |
| `isinstance` + runtime_checkable Protocol | 0.209 | 启动期一次，运行时不做 |
| dataclass 普通实例 | 0.318 | 每个 `Detection` |
| dataclass `slots=True` 实例 | 0.213 | 比普通 dataclass 便宜 33% |
| dataclass `frozen=True` 实例 | 0.422 | 比普通 dataclass 贵 33%（`object.__setattr__` 路径） |
| 100 个 `Detection`（普通 dataclass） | 27.6 | 每帧的框数 |
| 100 个 `Detection`（slots） | 19.4 | 省 30% |

- 结论一（约束）：框架自己的语言构造加起来不到 0.2 µs/帧，占 60 fps 预算（16.67 ms）的万分之一。
- 结论一对应的硬规则：不得为了性能牺牲契约——dict 化的 io、能力对象化的 wrapper、统一的 kwargs 透传必须全部保留。
- 结论二（约束）：真正贵的是每帧的对象分配与数据搬运。

设备实测（MaixCAM2 / aarch64 / Python 3.13.2 / numpy 2.2.6）：

| 构造 | µs/op | 说明 |
|---|---|---|
| dict 字面量（2 项） | 1.159 | io 契约的每帧成本 |
| 函数 / 闭包调用 | 0.687 | wrapper 的 `detect` 调用 |
| kwargs 打包（2 项） | 1.030 | `conf_th` / `iou_th` 透传 |
| 锁 acquire + release | 1.228 | facade 的切换锁 |
| `isinstance` + runtime_checkable Protocol | 5.301 | 启动期一次，运行时不做 |
| dataclass 普通实例 | 6.049 | 每个 `Detection` |
| dataclass `slots=True` 实例 | 4.422 | 比普通 dataclass 便宜 27% |
| dataclass `frozen=True` 实例 | 11.169 | 比普通 dataclass 贵 85% |
| 100 个 `Detection`（普通 dataclass） | 573.4 | 每帧的框数 |
| 100 个 `Detection`（slots） | 385.2 | 省 33% |

| 操作 | µs/op | 说明 |
|---|---|---|
| 取张量视图 | 1.712 | 免费 |
| 拷贝 2.8 MB 张量 | 653.0 | 一次 `np.array` 就吃掉一帧预算的 3.9% |
| HWC→NCHW 连续化（2.6 MB） | 9615.4 | 布局选错的固定税，单这一条就超出一帧预算 |
| 向量化 topk 解码 | 35.1 | 推荐做法 |
| Python 循环解码 100 框 | 494.5 | 比向量化贵 14 倍 |

- 设备组的结论一（**推翻 x86 组**）：同样的构造在 A53 上贵约 20 倍，选中的语言开销合计 **613 µs/帧**，占 60 fps 预算的 **3.7%**——不再是"万分之一"。
  理由：x86 是乱序执行 + 强推测执行，A53 是双核 1.2 GHz 的顺序发射内核，解释器每条字节码的成本差一个数量级。
  结论：**任何"语言开销可忽略"的说法只在 x86 成立，设备上必须按 0.6 ms/帧 计入预算**。
- 设备组的结论二：HWC→NCHW 连续化 9.6 ms 一条就超过一帧预算，这比"慢多少倍"更能说明问题——规范路径的布局转换必须是 `out=` 写进预分配缓冲，不能靠 `np.ascontiguousarray` 现造。
- 设备组的结论三：`slots=True` 的收益从 x86 的 33% 稳定到 27%，`frozen=True` 的代价从 33% 涨到 85%；热路径禁用 `frozen` 这条硬规则在设备上更值得坚持。
  理由：`frozen` 走 `object.__setattr__` 路径，而这条路径在弱内存序核上要多付一次内存屏障级别的开销。

下表固定每帧数据搬运与解码方式的实测开销。

| 操作 | µs/op | 说明 |
|---|---|---|
| 取张量视图 | 0.095 | 免费 |
| 拷贝 2.8 MB 张量 | 441.6 | 一次 `np.array` 就吃掉一帧预算的 2.6% |
| HWC→NCHW 转置（2.6 MB） | 570 | 布局选错的固定税 |
| NCHW→HWC 转置 | 857 | 布局选错的固定税 |
| 向量化 topk 解码 | 2.9 | 推荐做法 |
| Python 循环解码 100 框 | 22.0 | 比向量化贵 7.5 倍 |

落到设计上的三条硬规则：

1. **热路径对象不得用 `frozen=True`**。`frozen` 只给加载期配置（`ArtifactRef`、`Capabilities`）用；每帧产生的 `Detection` 用普通 dataclass 加 `slots=True`。
2. **引擎适配器必须返回视图，不得返回拷贝**。ncnn 的 `extract(name, 1)` 取的是引用；每帧一次 2.8 MB 的 `np.array()` 在 A53 上就是几毫秒。
3. **解码必须向量化**。置信度筛选、topk、NMS 的 IoU 计算都在 numpy 里做完，Python 循环只负责把最终结果包成对象。

### §10.2 算子开销（letterbox / NMS / 布局）

状态：`[已定]`（x86 实测，设备值 `[待测]`）

下表固定 letterbox / NMS / 布局类算子在 numpy 与 OpenCV 下的实测开销与倍数判定。

| 算子 | numpy | OpenCV | 倍数 |
|---|---|---|---|
| letterbox（1280×352 → 640×352） | 1.42 ms（最近邻）/ 13.13 ms（双线性） | 0.11–0.33 ms | 13×（最近邻）/ 120×（双线性） |
| NMS（300 框，IoU 0.45） | 3.33 ms | 0.096 ms | 35× |
| 归一化（uint8→float32） | 1.71 ms | 1.24 ms | 1.4× |
| HWC→NCHW 打包 | 0.60 ms | 0.16 ms | 3.6× |

- 这组数字是 §1.1 那条依赖边界按"库的性质"划的直接理由：把 `cv2` 挡在 `ops.py` 外面，letterbox 慢 13 倍、NMS 慢 35 倍，而且纯 numpy 的双线性 letterbox 还要额外付插值精度。

设备实测（MaixCAM2 / aarch64 / numpy 2.2.6 / OpenCV 4.12.0，源 1280×352 → 640×352）：

| 算子 | numpy | OpenCV | 倍数 |
|---|---|---|---|
| letterbox（1280×352 → 640×352） | 16.48 ms（最近邻）/ 112.05 ms（双线性） | 2.34 ms | 7×（最近邻）/ 48×（双线性） |
| NMS（300 框，IoU 0.45） | 65.63 ms | 1.38 ms | 47× |
| 归一化（uint8→float32） | 10.38 ms | 9.34 ms | 1.1× |
| HWC→NCHW 打包 | 4.77 ms | 2.12 ms | 2.2× |
| uint8→float32 转换 | 0.92 ms | — | — |
| HWC→NCHW 转置 | 4.93 ms | — | — |
| NCHW→HWC 转置 | 12.28 ms | — | — |

- 设备组的推论一（**改变结论**）：numpy 版 NMS 在 A53 上是 65.6 ms，**单条就超过一帧 60 fps 预算**。
  理由：x86 上只有 3.33 ms，看着无所谓；A53 上是慢 20 倍。规范路径（张量引擎）必须用 `cv2.dnn.NMSBoxes`，`ops.py` 允许 `cv2` 这条边界在设备上是硬约束，不是口味。
  注意这只针对张量引擎：端到端引擎（maix.nn）的 NMS 在 NPU 运行时内部完成，不走 `ops.py`。
- 设备组的推论二：letterbox 的倍数差从 x86 的 13× 收敛到 7×，说明 OpenCV 在 A53 上的优势主要来自**避免 Python 侧循环**而不是算子本身快；双线性那栏 48× 才是真实差距。
- 设备组的推论三：归一化在两边几乎持平（1.1×），归一化没有"必须上 cv2"的理由——它的成本来自浮点除法本身。

### §10.3 引擎侧的外部基线

状态：`[已定]`（外部数据 + 本项目真机实测已回填）

下表固定引擎侧的外部性能基线与来源，用于判断量级。

| 项 | 数据 | 来源 |
|---|---|---|
| MaixCAM2 NPU | 3.2 TOPS@INT8 / 12.8 TOPS@INT4 | Axera 官方规格 |
| MaixCAM2 CPU | 双核 Cortex-A53 @ **1.2 GHz**，带 NEON | Sipeed / Axera 官方规格 |
| YOLO11n 640×640 | 官方标称 113 FPS | Sipeed 官方 wiki |
| MaixCAM2 实测 YOLO11n 640×480 | 纯推理中位 **8.6 ms**（原生帧）/ **9.6 ms**（numpy 路径）；模型加载 20 ms | 本项目真机实测（官方 zoo `yolo11n.mud`，VNPU axmodel） |
| ncnn / YOLO11n CPU | RPi 4（A72 @1.8 GHz）409 ms/帧；RPi 5（A76 @2.4 GHz）63 ms/帧（纯推理）/ 67 ms（含预处理） | 第三方实测 |
| ncnn int8 收益 | RPi 5 上仅比 fp32 快约 4%（单线程），多线程反而更慢 | 同上 |

对本项目的推论（约束）：

- 本项目输入是 1280×352（450 k px），约为 640×640 的 1.1 倍；NPU 侧理论上仍在百帧量级，但**必须在真机实测**，不得拿 113 FPS 直接当预算。
- MaixCAM2 的 A53 @1.2 GHz 比上表里最弱的 RPi 4（A72 @1.8 GHz）还弱一档；ncnn 在这块板子上跑 YOLO11n 预计 150–250 ms/帧，即 4–7 FPS。
  理由：与 NPU 差一个数量级以上——§2 里"CPU 是性能降级"的判断有数据支撑。
- ncnn 的 int8 收益主要在体积（模型小 75%），不在速度。
  理由：选 ncnn 的理由只能是"做对比"或"NPU 不支持该算子"，不得是"int8 更快"。
- 真机实测把量级钉死了：640×480 的 YOLO11n 在 NPU 上纯推理约 8.6 ms（约 115 FPS），与官方标称的 113 FPS 同量级，说明 NPU 侧不是瓶颈。
  但这**不是本项目的预算**：实测模型是官方 zoo 的 640×480，本项目输入 1280×352，且还要叠加取帧、搬运与协议打包。
  结论：性能不构成选型约束，但真机帧预算仍须按本项目分辨率与管线开销重测。

### §10.4 复现与待测

状态：`[已测]`（x86 与 MaixCAM2 真机各跑过一轮）

```bash
python scripts/bench_inference_ops.py --width 1280 --height 352 --target 640 352
python scripts/bench_python_overhead.py --frames 300
```

- 两个脚本都必须能在设备上直接跑（`bench_inference_ops.py` 在没有 cv2 时自动跳过 OpenCV 那一组）。
- 设备上的数字才是预算依据；x86 的数字只用于判断量级。
- 复现口径：设备侧用 `python3` 直接跑脚本，不加任何环境包装；本轮设备环境为 Python 3.13.2 / numpy 2.2.6 / OpenCV 4.12.0 / aarch64。

## §11 等价性验证

状态：`[已定]`（用例 `[待测]`）

- 跨引擎对比不得"跑通就行"，必须量化。

下表固定等价性验证的基准集、指标与覆盖范围。

| 项 | 定稿 |
|---|---|
| 黄金帧集 | `tests/fixtures/golden/<model_id>/frames/*.npy` 加 `expect.json`（Phase 3 交付） |
| 基线来源 | 现役引擎（MUD / maix.nn）在真机上的输出 |
| 比对指标 | 类别一致率、框 IoU、中心误差（px）、置信度差 |
| 容差 | 按 (引擎, 工件) 对分别定；量化方式不同则容差不同 |
| 覆盖范围 | 每条路径：规范路径（canonical）、native 快路径 |
| 工具 | 扩展现有调参页做双引擎叠加对比 |

比对层级必须按引擎能力分两档：

1. 张量级引擎（onnxruntime / ncnn / rknn_lite）：canonical 路径输出原始张量，比对张量与解码后框。
2. 端到端引擎（maix.nn）：不要求它吐原始张量，在 `Detection` 层比对。同一张黄金帧，maix.nn 出的框与 onnxruntime 加 `vision_det` 出的框对比。

- 真机已跑通的一条：MaixCAM2 上 YOLO11n 640×480 的端到端路径，与现役 `hal/platforms/maixcam2/ai.py` 在同一张帧上**逐位一致**——原生帧路径与 numpy 路径各 4 框，类别、分数、坐标全同。
  差异只有一处：`Detection.label` 现役留空，新内核按 manifest 的标签表填类别名；业务层读的是 `ai.labels` 而不是 `det.label`，行为不受影响。
- 测试用例必须写进 `tests/`，把"待测"变成"已测"。
