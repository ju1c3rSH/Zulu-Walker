> 本节目标：固定运行时对象、生命周期、启动期预载语义与引擎注册选择规则，确保运行期行为可预测。

# 04 运行时与引擎选择

来源：架构文档 §6 运行时对象与生命周期、§7 引擎注册与选择。整体状态：`[已定]`。

## §6 运行时对象与生命周期

状态：`[已定]`

### §6.1 三个对象

状态：`[已定]`

下表固定三个运行时对象的职责与生命周期。

| 对象 | 职责 | 生命周期 |
|---|---|---|
| `ModelRuntime` | engine 实例 + artifact + wrapper 产出的领域 API + 统计 | 加载时创建，卸载时销毁 |
| `ModelRegistry` | `nick_name -> ModelRuntime`（可处于未加载态） | 全程 |
| `AIFacade` | 实现 `AIInference` 基础协议 | 全程 |

`ModelRuntime` 必须向 wrapper 暴露：

```python
rt.engine       # 引擎实例
rt.artifact     # 工件契约（input / io / preprocess / wrapper / build）
rt.params       # wrapper_params
rt.stats        # 统计
rt.preprocess   # frame -> PreprocessResult（inputs + transform）
```

- `rt.preprocess` 必须是 wrapper 唯一的输入准备入口。
  理由：wrapper 调它，不自己实现缩放与归一化（§3.4，见 §3.4（02-contracts-and-dataflow.md））。
- wrapper 的产物是**能力对象**（§3.4，见 §3.4（02-contracts-and-dataflow.md）），由 runtime 持有；facade 通过它对外暴露任务接口。

### §6.2 单活 + 按需预加载

状态：`[已定]`

- 同一时刻只允许有一个 active runtime（沿用现役 `switch()` 语义）。
- `add(nick, ...)` 只登记不加载。
  理由：现役 `Machine.create()` 的 `add() -> switch(active)` 序列语义不变。
- `switch(nick)`：目标已加载则原子换指针（O(1)）；未加载则需要 `load()`。
- 框架不得自动预加载；需要时必须由配置显式声明 `preload: true`。

#### §6.2.1 启动期预载

状态：`[已定]`

- 预载只有一个意义：把"运行期切换模型时的加载成本"挪到启动期，让 `switch()` 退化成原子指针交换，从而绕开 §6.3 的"加载期持 GIL 冻结所有 Python 线程"问题。
- 必须**按逻辑模型预载，不按引擎工件预载**。
  理由：一个逻辑模型可以挂着多个引擎工件（maix_nn / onnxruntime / ncnn），它们是同一份权重的不同实现；运行期只有一个引擎（§7.2），所以只有那个引擎的工件会被装载，manifest 里其它引擎的工件是离线转换产物，不进运行时。
- 只有运行期会切换到的模型才值得预载。
  理由：全程只用一个模型时，预载只是把几百毫秒从首次推理挪到启动，没有实质收益，还白占一份加速器内存。
- 必须**串行加载，不并行**。
  理由：四个引擎的 `load()` 全程持 GIL（onnxruntime 的 session 初始化不释放 GIL；ncnn / maix.nn 是 pybind11 默认；rknn-lite 是 C API 包装），持 GIL 意味着 Python 层调度不动，并行加载零收益——能并行的只有引擎自己内部的 C++ 线程，那与"同时装两个模型"无关。
- 时机约束是顺序，不是平台：预载必须在所有需要连续占用 GIL 的后台线程启动之前做完。
  理由：平台层的 UART 接收线程由 `start_receiver()` 创建，调用点在 UART 模块的 `start()` 里（晚于 `Machine.create()` 返回），因此与预载不重叠；linux 平台的相机采集线程在 `CameraHub.open()` 时就起来了，与预载重叠，后果是加载期间丢几帧，视觉管道对丢帧不敏感，与 UART 字节丢失不同级；预载留在 `Machine.create()` 内即满足这个约束。
- 失败语义：**任一预载失败即拒绝启动，并输出详细日志**。
  理由：不区分 active 与非 active——预载是配置里明确声明的意图，声明了却做不到就是配置错误，静默降级成"运行期再试"会把问题推迟到比赛中途。
- 失败日志必须包含：失败的 nick、模型路径、选中的引擎、失败阶段（选引擎 / 读工件 / 建 runtime / 建 wrapper）、底层异常与错误码。
- 现役 `Machine.create()` 把 AI 初始化包在 `except Exception` 里并置 `ai = None`，这在预载语义下不可接受。
  理由：失败必须冒到启动期，让"AI 不可用"是显式的启动失败，而不是运行期才发现没有检测。
- **不做资源准入判断**：不得预先按 `memory_hint` / `load_cost_hint` 估算显存或内存是否够，直接加载，资源不足交给引擎自己报错。
  理由：基于提示值做预测，猜错的代价是"该加载的没加载"，比加载失败更难查。
- **预载预算**：`preload_budget_ms` 是启动期预载的**总**预算（累加），不是每个模型各自的上限；超预算就放弃剩余预载并记 warning。
  理由：预载是优化，不该把启动拖垮。
- **预热是另一件事**：预载解决"加载慢"，不解决"首帧慢"。
  理由：maix.nn 第一次 `detect()` 还有额外的 NPU 首次调度与内存页分配；`warmup: true` 会在加载后跑一次预热，把这笔开销也挪到启动期；循迹场景首帧慢几十毫秒无差别，默认关闭。
  真机实测（MaixCAM2，YOLO11n 640×480）：加载后的**第一次推理不返回任何检测**；**零帧预热不能清除该状态**——预热帧之后的那一帧仍然是空的。
  结论：预热必须用一张真实帧；零帧预热既没 warms up，还白占一次推理预算。

配置形态：

```yaml
ai:
  engine: "maix_nn"           # 必填：运行期唯一的引擎
  preload_budget_ms: 3000
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_1280x352/steelball_1280x352.mud"
      manifest: "/root/models/steelball_1280x352/manifest.yaml"
      preload: true
      warmup: false
```

下表固定各引擎的加载特性，决定预载值不值得。

| 引擎 | load 耗时 | load 期 GIL | 多实例共存 | 预载价值 |
|---|---|---|---|---|
| `maix_nn` | 200–500ms（flash 重载） | 持有（pybind11） | 受加速器内存池限制 | 高——切换最慢，最该预载 |
| `onnxruntime` | 中（session 初始化不释放 GIL） | 持有 | 可 | 中 |
| `ncnn` | 快 | 持有（pybind11） | 可 | 低——按需加载也不痛 |
| `rknn_lite` | 快 | 持有（C API 包装） | 按 `core_mask` 分核 | 低 |

### §6.3 并发与安全点

状态：`[已定]`

1. facade 必须持一把 `threading.Lock` 加一个在途推理计数。
2. `switch()` 需要加载时，必须等待在途推理结束，最长 `switch_timeout_ms`（默认 2000ms）。
3. 超时必须抛 `E_SWITCH_TIMEOUT`。
   理由：不得静默卡死，也不得丢一次推理。
4. 加载只能由主循环发起。
   理由：加载期持 GIL 会冻结所有 Python 线程。
5. 框架必须提供 `Machine.safe_point()` 钩子，app 层可在此实现"允许阻塞"的窗口。

### §6.4 卸载

状态：`[已定]`

- `remove(nick)` / `unload()` 必须释放引擎资源（ORT session、ncnn net、NPU 句柄）。
- 卸载正在使用的 runtime 必须等待在途推理结束（同 §6.3 的锁）。

## §7 引擎注册与选择

状态：`[已定]`

### §7.1 注册表（不用 entry_points）

状态：`[已定]`

`framework/inference/engines/__init__.py`：

```python
REGISTRY: dict[str, EngineFactory] = {
    "maix_nn": _load_maix_nn,
    "ncnn": _load_ncnn,
    "onnxruntime": _load_onnxruntime,
    "rknn_lite": _load_rknn_lite,
}
```

- 每个 `_load_*` 内部必须 try / except import，失败返回 `None`（该引擎标记不可用），注册表条目保留。
  理由：这样"装了但 import 失败"能被区分于"没装"。
- 不得使用 `entry_points`。
  理由：MaixCAM2 的 `app.yaml` 打包与 pip 插件机制耦合很深，首期没有外部引擎的实际需求；目录加注册表可审计、可离线、可被隔离脚本检查。
- 注册表登记的是**全部可用引擎**，但运行期只装载配置指定的那一个（§7.2）。
  理由：完整清单仍然有用——它回答"这台设备上装了什么、哪个 import 失败了"，也支撑离线转换工具链选择目标格式。

### §7.2 引擎与工件的选择

状态：`[已定]`

运行期只有一个引擎，必须显式指定，不做候选链：

```yaml
ai:
  engine: "maix_nn"           # 必填：maix_nn | ncnn | onnxruntime | rknn_lite
  preload_budget_ms: 3000     # 启动期预载的总预算（累加），超预算放弃剩余预载
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_1280x352/steelball_1280x352.mud"      # 精确的工件文件
      manifest: "/root/models/steelball_1280x352/manifest.yaml"            # 精确的 manifest 文件
      preload: false           # 可选：true 则启动期加载，switch() 退化为指针交换
      warmup: false            # 可选：加载后跑一张零帧预热
```

规则：

1. `engine` 必填，取值必须在 `REGISTRY` 里；引擎不在注册表，或 import 失败，抛 `E_ENGINE_UNAVAILABLE`（启动期硬失败）。
2. 不得做 `auto`，不得做 `prefer` 候选链，不得做"这个引擎不行就换一个"。
   理由：换引擎等于换加速器（NPU ↔ CPU），性能差一个数量级以上；静默降级到 CPU 引擎会得到一个"能跑但慢十倍"的系统，而现场没人知道速度为什么变了；宁可启动失败。
3. 不得提供按模型覆盖引擎的入口——所有模型用同一个引擎。
   理由：引擎是运行期属性，不是模型属性。
4. manifest 里必须存在该引擎的工件声明，且 `file` 与 `model` 指向同一个文件；缺声明或文件对不上，抛 `E_CONTRACT_MISSING`。
5. 绝不静默退化成 stub。

迁移提醒：

- 现役 `framework/hal/machine.py` 的 `Machine.create()` 只认 `nick_name` / `model` / `model_type` / `active` 四个键（见该文件 81 行与 155-176 行）。
- **Phase 1 落地前，`project_config.yaml` 的 ai 段必须保持旧格式不动**，否则启动即失败；上面的目标形态由 facade 在 Phase 1 接管解析。
- 现役 `model` 已经是精确文件路径（`/root/models/steelball_1280x352.mud`），落地时必须把它移进 manifest 同目录并补上 `manifest:` 键。

### §7.3 加载期校验

状态：`[已定]`

- 加载时必须全部校验完，任何一项不匹配都在启动期抛错，不得拖到推理时才发现。

下表固定加载期校验项与对应的失败错误码。

| 校验项 | 失败错误码 |
|---|---|
| 引擎在当前平台可用 | `E_ENGINE_UNAVAILABLE` |
| manifest 里存在该引擎的工件声明 | `E_CONTRACT_MISSING` |
| manifest 的 `artifacts.<engine>.file` 与配置的 `model` 指向同一文件 | `E_CONTRACT_MISSING` |
| 工件文件存在且可读 | `E_MODEL_LOAD_FAILED` |
| 工件声明的布局 / 尺寸与引擎 `input_specs()` 一致 | `E_OUTPUT_LAYOUT_MISMATCH` |
| 实际输出张量 shape 与声明的方言一致 | `E_OUTPUT_LAYOUT_MISMATCH` |
| labels 数量与工件一致 | `E_OUTPUT_LAYOUT_MISMATCH` |
| 模型族与方言匹配（manifest 的 `family` ∈ wrapper 条目的 `families`） | `E_MODEL_TYPE_UNSUPPORTED` |
| runtime 版本与构建工具链版本匹配 | `E_MODEL_LOAD_FAILED` |
| 构建配方与工件的输入尺寸 / 布局一致 | `E_OUTPUT_LAYOUT_MISMATCH` |
