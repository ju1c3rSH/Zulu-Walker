# 桌面端推理后端（ONNX Runtime / NCNN）

> 本节目标：说明桌面端（普通 Linux / Windows）如何接入推理内核、配置形态、依赖矩阵、GPU 加速与等价性验证口径。
> 设计真相源见 `inference/（`10-engine-ncnn.md、`11-engine-onnxruntime.md、`08-backend-contracts.md）。

## 1. 现状

桌面端与真机共用同一套推理契约，但走的是两条不同的路径：

| | 真机（MaixCAM2） | 桌面端（Linux / Windows） |
|---|---|---|
| 平台包 | `framework/hal/platforms/maixcam2/` | `framework/hal/platforms/linux/`（Windows 复用同一包） |
| 引擎 | `maix.nn`（端到端方言） | `onnxruntime` / `ncnn`（张量级方言） |
| 后端实现 | `MaixCam2AI` 直连 `maix.nn` | `LinuxAI` → `AIFacade` → 推理内核 |
| 工件 | MUD 文件 | manifest + `.onnx` / `.param` + `.bin` |

真机路径**保持不变**：只多了一个被忽略的 `manifest_path` 关键字，行为逐帧一致。桌面端则第一次真正用上了推理内核。

## 2. 配置形态

`configs/project_config.desktop.yaml` 是桌面端的配置：

```yaml
platform: linux
uart_defaults:
  port: "auto"          # 解析本机系统串口; 找不到就没有串口, 照常启动
ai:
  engine: "onnxruntime" # 必填, 单一显式引擎, 不做 auto / 候选链
  switch_timeout_ms: 2000
  frame: { color: bgr, dtype: uint8, layout: hwc }
  models:
    - nick_name: "yolo11n"
      model: "deploy/models/yolo11n/yolo11n.onnx"
      manifest: "deploy/models/yolo11n/manifest.yaml"
  active: "yolo11n"
```

- `ai.engine` **必填**。配置了引擎而引擎起不来 = **启动失败**（不静默降级）。没有 `ai.engine` 的配置走平台自有后端（真机路径）。
- `ai.models[].manifest` 指向逻辑模型的契约；`ai.models[].model` 是**精确的工件文件**，不是目录。
- 换引擎只改 `ai.engine` 一行，工件声明都在 manifest 里。

## 3. 启动

```bash
# 推荐：显式指定配置
python run.py main --config configs/project_config.desktop.yaml

# 等价：环境变量
ZWULU_CONFIG=configs/project_config.desktop.yaml python run.py main
```

Windows（PowerShell）：

```powershell
$env:ZWULU_CONFIG="configs/project_config.desktop.yaml"; python run.py main
```

**串口**：`port: "auto"` 由平台层（`framework/hal/platforms/linux/uart.py`）解析本机系统串口（Linux 优先 `/dev/ttyUSB*`、`/dev/ttyACM*`；Windows 用 `COM3` 及以上）。解析不到时 UART 缺席，**应用照常启动**——串口缺失不是致命错误。

## 4. 依赖矩阵与 GPU

| 后端 | 安装 | GPU | 说明 |
|---|---|---|---|
| onnxruntime | `pip install onnxruntime` | CPU | 基线 |
| onnxruntime | `pip install onnxruntime-gpu` | CUDA | 需 CUDA 12/13 + cuDNN 9 |
| onnxruntime | `pip install onnxruntime-directml` | DirectML | Windows |
| ncnn | `pip install ncnn` | CPU | wheel 为 CPU 构建 |
| ncnn | 源码编译 `--vulkan=on` | Vulkan | 见下 |

**ORT 的三个包装进同一个 `onnxruntime` 模块**，互相覆盖且 pip 不会自动换干净。安装顺序：先 `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml`，再装目标包。

**EP 静默回退**：请求了 `CUDAExecutionProvider` 但机器上没有时，ORT **不报错**，只是悄悄退回 CPU。适配器在 `load()` 里比对"请求的 EP"与"实际生效的 EP"，不一致就记 WARNING，并把实际设备写进 `capabilities().device`。

**ncnn Vulkan 源码编译**：

```bash
git clone --recursive https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake -DNCNN_VULKAN=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release -j
cd ../python && python setup.py install --vulkan=on
```

前置条件：安装 **Vulkan SDK** 并把 `VULKAN_SDK` 指向它（Windows / Linux 均需）；Windows 侧需 VS2022 + `vcvarsall.bat`。

**请求了不可用的加速后端一律硬失败**（`E_ENGINE_UNAVAILABLE`），绝不静默退回 CPU。

## 4.5 runtime.version 契约

manifest 的每个工件都必须在 `runtime` 段写 `version`：这是**工件被验证过的运行期版本**。适配器在 `load()` 时把它与已安装库的版本逐段比对，不一致抛 `E_MODEL_LOAD_FAILED`（加载失败，不是降级）。

比对规则：**固定里写了几个段，就比对几个段**；固定没写的段不参与比对。

| 固定 | 接受 | 拒绝 |
|---|---|---|
| `1.26` | `1.26.0`、`1.26.3` | `1.27.0`、`1.25.7` |
| `1.26.0` | `1.26.0` | `1.26.3` |
| `1.0.20261020`（ncnn） | `1.0.20261020` | `1.0.20261101` |

- 不写 `version` = 不声明版本，跳过比对；但 `scripts/check_manifest.py` 把缺 `version` 的 `runtime` 块判为**错误**，所以仓库内 manifest 必须写。
- 库不报版本 = 无法验证的固定 = 加载失败。
- `runtime.version` 与 `build.version` 同时存在时必须相等（`check_manifest.py` 的 `check_version_agreement`）。

实现见 `framework/inference/engines/version.py`，测试见 `tests/test_inference_runtime_version.py`。

## 5. 工件与转换工具链

工具链**离线**，不进运行时框架、不进仓库产物；配方记在 manifest 的 `build` 段，产物才部署。

| 步骤 | 工具 | 产物 | 脚本 |
|---|---|---|---|
| 1 | `torch.onnx.export` / ultralytics | `.onnx`（raw head，不带 NMS） | `scripts/export_onnx.py` |
| 2 | `onnx2ncnn` / `pnnx` | `.param` + `.bin` | `scripts/convert_ncnn.py` |
| 3 | Pulsar2 (Docker) | `.mud`（真机） | — |

**★ 转换后必须回填 manifest 的 `io`**：ncnn 的 `.bin` 既不含张量名也不含形状，manifest 是 io 映射的唯一声明源。`scripts/convert_ncnn.py` 转换后会回显 `input_names()` / `output_names()`，按回显结果写：

```yaml
io: { inputs: { image: "data" }, outputs: { pred: "output" } }
```

产物不进 git（`.gitignore` 已排除 `.onnx/.param/.bin/.mud/.pt`），manifest 与 labels 保留在仓库里。

**本机可用的转换器与它们各自的结果**（2026-11 实测）：

| 转换器 | 输入 | 结果 |
|---|---|---|
| `onnx2ncnn` | onnx | **不可用**：ncnn 只发布运行库，转换器要源码编译（本机没有 cmake / MSVC / g++） |
| `pnnx`（PyPI 包，PATH 上有 `pnnx.exe`） | onnx | **翻译不了**：ONNX 导入器对 YOLO11 头部的 `Expand` 报 `todo Expand` 后崩溃（Windows 异常码 `0xc0000409`） |
| `pnnx` | TorchScript（ultralytics 导出） | **能转但不等价**：图能加载能跑，输出却是 21×8400，而模型是 84×8400（头部 4 个分支的 Concat 被翻译错） |
| `pnnx` | pnnx 自己的 ONNX 中间产物 | **不可用**：pnnx 的 ONNX 导入器拒绝自己的产物（`Missing opset in the model`） |

`scripts/convert_ncnn.py` 现在会把这几种失败**分别**报出来（不支持的算子、崩溃的异常码、非零退出码），不会再"看起来成功"。

**结论**：真实 YOLO11n 的 ncnn 工件要用 `onnx2ncnn`（或能正确翻译该头部的更新版 pnnx）产出。在此之前 ncnn 侧只有微型模型对这一级别的证据。

## 6. 等价性验证

`scripts/check_engine_equivalence.py` 做**双档**比对：

- **档一（张量级）**：两个引擎的原始输出张量——最大绝对误差、均值绝对误差、余弦相似度；
- **档二（Detection 级）**：各自 `decode_yolo` 之后的框——类别集合、匹配框 IoU、置信度差。

onnxruntime 是参考实现。

| 指标 | fp32 | fp16 |
|---|---|---|
| 张量最大绝对误差 | ≤ 1e-3 | ≤ 5e-2 |
| 张量余弦相似度 | ≥ 0.9999 | ≥ 0.999 |
| 匹配框 IoU | ≥ 0.90 | ≥ 0.90 |
| 置信度绝对差 | ≤ 0.02 | ≤ 0.02 |

必须覆盖 `resize` 的 `letterbox` 与 `stretch` 两种 mode：不同 mode 产生的框位置不同，跨引擎比对前要先确认两边走的是同一个 mode。

```bash
python scripts/check_engine_equivalence.py --manifest deploy/models/yolo11n/manifest.yaml --frames assets/golden
python scripts/check_engine_equivalence.py --synthetic   # 微型模型对, 不需要转换工具链
```

**微型模型对**（`scripts/tiny_model.py`）是同一个卷积在两种方言下的产物，权重共享，因此是最紧的跨引擎自检：不需要 C++ 工具链就能验证"预处理 → 引擎 → 张量 → 解码"整条链路。

## 7. 已知坑

| 坑 | 后果 | 对策 |
|---|---|---|
| ORT 请求了 EP 但没生效 | 以为在用 GPU 实则 CPU | `load()` 比对请求与实际，记 WARNING |
| ncnn `substract_mean_normalize` **乘**第二参数 | 数值差一个倒数 | 适配器传 `1/std`，公式换算仍由 `ops.convert_norm_params` 唯一实现 |
| ncnn 该原语要求**每通道**一个值 | 长度不等于通道数时只处理第一个通道，其余是垃圾 | 适配器按通道数广播 |
| ncnn `extract(name, 1)` 返回引用 | 拷贝移出 extractor 作用域就读到已释放内存 | 拷贝在 `with` 块内完成 |
| ncnn Mat 要求 16 字节对齐 | 走非对齐 kernel | `ops.alloc_aligned_buffer()` 自持对齐缓冲 |
| ncnn 的 `.param` blob 名不确定 | io 对不上，报 `E_OUTPUT_LAYOUT_MISMATCH` | 转换后回填 manifest |
| ncnn `load_model` 失败只返回错误码 | 缺权重文件却像加载成功 | 适配器检查返回码并抛 `E_MODEL_LOAD_FAILED` |
| ORT session 初始化持 GIL | 加载期冻结所有 Python 线程 | 加载只经 `switch()` 在主循环发起 |
| ORT 规范输入是 HWC，模型要 NCHW | 每次推理一次转置 | 适配器转置；三个引擎里只有 ORT 需要 |

## 7.5 本机验证状态

| 项 | 状态 | 证据 |
|---|---|---|
| ORT 适配器加载真实 YOLO11n | 通过 | `bus.jpg` 检出 5 个目标、`zidane.jpg` 3 个，`last_error_code` 为空 |
| 桌面主程序端到端启动 | 通过 | `python run.py main --config configs/project_config.desktop.yaml`：开摄像头、持续出检测框 |
| 无串口启动 | 通过 | 同上；串口缺席只打印 `UART creation failed (non-fatal)`，应用照常运行 |
| ncnn 适配器（微型模型对） | 通过 | 与 ORT 的最大绝对误差 1.5e-05 |
| ncnn 适配器（真实 YOLO11n） | **未验证** | 本机没有 `onnx2ncnn`（需源码编译），`pnnx` 的 ONNX 导入器遇 `Expand` 崩溃、TorchScript 路线产出**形状错误**的图（21×8400 ≠ 84×8400）；ncnn 的 io 绑定目前只有微型模型对这一级别的证据 |
| ORT ↔ ncnn 真实模型等价性 | **未验证** | 同上；跨引擎等价性目前只有微型模型对级别的证据，真模型的等价性要等转换工具链就位后补 |

**未验证项的处理方式**：转换工具链缺失是**离线工具链**问题，不进运行时框架，也不靠放宽工件要求来绕过——真 `.param`/`.bin` 产出后，`scripts/convert_ncnn.py` 会回显 blob 名，回填 manifest 的 `io` 即可接入现役路径，代码零改动。

## 8. 相关文件

| 文件 | 作用 |
|---|---|
| `framework/inference/engines/onnxruntime.py` | ORT 适配器（参考实现） |
| `framework/inference/engines/ncnn.py` | ncnn 适配器 |
| `framework/hal/platforms/linux/ai.py` | `LinuxAI`（facade 支撑的桌面后端） |
| `framework/hal/platforms/linux/uart.py` | 系统串口解析 |
| `framework/inference/ops.py` | `alloc_aligned_buffer`（唯一一份对齐缓冲） |
| `framework/inference/engines/version.py` | 固定版本与已安装版本的比对（唯一一份） |
| `scripts/tiny_model.py` | 微型模型对（两种方言，共享权重） |
| `scripts/check_engine_equivalence.py` | 双档等价性验证 |
| `scripts/export_onnx.py` / `scripts/convert_ncnn.py` | 离线转换工具链 |
| `configs/project_config.desktop.yaml` | 桌面端配置 |
| `deploy/models/yolo11n/` | 公开 YOLO11n 的 manifest 与 labels |
