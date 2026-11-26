# 运行时 UI 的跨平台统一（MaixCAM2 / MaixCAM1 / 通用 Linux / Windows）

> 生成日期：2026-11-26
> 目的：把「MaixCAM 与通用 Linux/Windows 的运行时 UI 尽量统一」这件事，从感觉变成一份可执行的边界定义 + 落地路线。
> 证据来源：仓库各分支实读（`git show`/`git grep`）、`F:\MaixPy-main` 上游源码与官方文档、MaixPy 官方 wiki。
> 关联文档：`docs/architecture/preview_and_tuning.md`、`docs/architecture/thread_tick_topology.md`、`docs/retrospective/framework_issues.md`（D1~D6 决策）。

---

## 0. 结论先行

1. **「统一」的现实边界不是「一套代码四端跑」，而是「一份 UI 语义代码 + 每端一层薄后端」。** 绘制与输入必须在设备端（`maix.image` / `maix.touchscreen`）与桌面端（cv2 / 鼠标键盘）之间分叉，这是硬件与内存约束决定的，不是工程偷懒。
2. **仓库里最完整的一套显示抽象已经写好了，但不在 main 上**：`feat/maixcam1-port` / `EOD2026-NearCam` 分支的 `framework/display/`（单一 `show()` 出口 + tap 扇出 + 策略单一事实源 + 平台能力声明 `caps()` + seq 代数号帧槽 + 慢格式告警 + show 开销统计）。**回收它比从零重写划算得多。**
3. **真正的空洞是「画布层」与「输入层」**：
   - 画布层（Canvas 协议 / OverlayRenderer / compositor）—— D2 定了但**任何分支都没有实现**，main 上 ~500 行绘制仍趴在 `vision_manager.py` 里；
   - 输入层（`InputSource`）—— 协议已定义（`framework/hal/interface/sink.py`）但**零实现**，触摸在三个分支里各自直连 `maix.touchscreen`。
4. **main 分支上 MaixCAM2 的按钮图标是「死 UI」**：`set_exit_icon` / `get_calib_button_rect` / `get_fill_light_button_rect` 全仓无调用者，main 也没建 touchscreen——图标画出来但点不动。
5. **通用 Linux 上所有 overlay 静默消失**：`vision_manager.py` 里 11 处函数内 `import maix.image` 用 `except ImportError: return` 兜底（`:517 :566 :661 :676 :691 :781 :792 :974 :994 :1008 :1026 :1046`），在非 maix 平台不是报错而是消失。

---

## 1. 现状矩阵：四个目标各自怎么做 UI

### 1.1 出图链路

| 目标 | 显示通道 | 像素容器 | 合成方式 | 节流 |
|---|---|---|---|---|
| **MaixCAM2**（main） | `MaixLcdSink.flush()` → `maix.display.Display.show()` | 平台原生 `maix.image.Image`（`FMT_RGB888`，`last_raw.copy()`） | `vision_manager._update_display_frame()` 每帧全量重绘 | 两处：VisionManager 重建窗口 + Sink `interval_ms` |
| **MaixCAM1 / Pro**（main 只有 HAL） | 同上，`platforms/maixcam1/display.py`（552×368 触屏面板） | 平台原生 `maix.image.Image` | main 未接；完整 UI 只在 `feat/maixcam1-port` | 同上 |
| **通用 Linux**（main） | `LinuxDisplay.show()` → `cv2.imshow` + `cv2.waitKey(1)` | BGR ndarray | 同上，但 11 处 maix 绘制静默失效 → **实际只有裸画面** | 无 |
| **Windows** | 无正式支持；仅老分支 `feature-windows-pc-support` 裸 `cv2.imshow`（不经 HAL） | BGR ndarray | 无 | 无 |

### 1.2 输入链路（现状最大的空洞）

| 目标 | 触摸 | 按键 | 鼠标/键盘 | 抽象 |
|---|---|---|---|---|
| MaixCAM2（main） | ✗ 无 | ✗ 无 | ✗ 无 | 只有 `app/main.py` 里的 TODO 注释 |
| release/2026H | `touchscreen.TouchScreen()` 直读 + `maix.image.resize_map_pos_reverse(..., Fit.FIT_CONTAIN, x, y)` 映射帧坐标 | ✗ | ✗ | 无 HAL，app 层直连 maix |
| feat/maixcam1-port / EOD | 同上（`DebugPageFactory._ensure_touchscreen`） | ✓ `maix.key.Key(cb)`，KEY_OK **释放沿**循环切页 + 0.5s 防抖 | ✗ | 无 HAL |
| 通用 Linux / Windows | ✗ | ✗ | 仅 `cv2.waitKey` 取 'q'/ESC，且**退出语义藏在 `show()` 返回值里** | 无 |

> `InputSource` 协议已存在于 `framework/hal/interface/sink.py`（`poll(cmds: CmdQueue)`），**全仓无任何实现类**。

### 1.3 UI 组织方式

| 分支 | 组织方式 |
|---|---|
| main | 无页面概念；每帧全量重绘 + if/else 命中测试 |
| release/2026H | 触摸三按钮（退出 / 标定 / 补光），if/else 直调，**无 KEY_OK**（更正：`framework_issues.md` ARCH-01 描述的 "main.py:86 触摸屏" 属于本分支，不是 main） |
| feat/maixcam1-port / EOD | **五页状态机** `app/page_sm.py`（MAIN + EXPLOSIVE/TARGET/RESCUE/LASER，走 `framework/state_machine` 引擎）+ `app/page_router.py`（`KeyCycleRouter`）+ 三个触屏自绘 tuner（`threshold_tuner.py` / `laser_calib_tuner.py` / `rescue_tuner.py`）+ `display/hud.py`（生产 HUD） |

### 1.4 调参 UI：两套零共享的体系

| 体系 | 入口 | 渲染 | 适用 |
|---|---|---|---|
| detector debug | `run.py debug <name>` | cv2 trackbar + 桌面窗口（`ParamPanel` / `BaseDebugWindow`） | 离线调参、阈值实验 |
| 设备端 tuner | 主循环 KEY_OK 切页 | 触屏自绘（滑条 / 组按钮 / SAVE-RESET-Exit） | 现场调参 |

`preview_and_tuning.md` §1 对此描述准确。**并且 `ParamDef` 在仓库里有两份重复实现**（`modules/zw_opencv_module/debug/param_panel.py` 与 `modules/zw_opencv_module/detectors/_shared/param_def.py`），字段命名还不一致（`display_name` vs `display`、`is_odd` vs `odd`）。

---

## 2. MaixPy 能力边界（已核实，F:\MaixPy-main）

| 能力 | API | 对统一方案的意义 |
|---|---|---|
| 显示 | `maix.display.Display(open=True)`（老固件降级 `Display()`）、`.show(img, fit=...)`、`.width()/.height()`、`.set_backlight()` | 设备端唯一出图口。CAM2 只支持 2.4″ 640×480 电容触摸；CAM/Pro 支持 552×368 等多款 |
| 画布 | `maix.image.Image` + `draw_rect/draw_string/draw_circle/draw_line/draw_image`、`string_size`、`load_font`/`set_default_font` | 设备端绘制后端。⚠ 颜色必须 `COLOR_*` 或 `Color.from_rgb()`，RGB 元组会抛异常并被绘制兜底吞掉 → **界面静默空白**（多个 tuner 头注释都记了这条） |
| 坐标映射 | `image.resize_map_pos` / `resize_map_pos_reverse` + `image.Fit.FIT_CONTAIN` | 帧↔屏映射的唯一正解，但**目前三处分支各写一份** |
| 触摸 | `maix.touchscreen.TouchScreen().read() -> (x, y, pressed)` | 设备端输入源 |
| 按键 | `maix.peripheral.key.Key(cb)`，`cb(key_id, state)`；默认 user/ok 键会 `set_exit_flag(True)` | 设备端输入源（页面切换） |
| USB HID | `maix.hid.Hid(DeviceType.DEVICE_KEYBOARD/MOUSE/TOUCHPAD)` | 板子反过来给 PC 充当键鼠/触摸板——「设备当屏、PC 当主机」的双向链路 |
| 国际化 | `maix.i18n.Trans` + `maixtool i18n` 扫描；中文字库 `/maixapp/share/font/SourceHanSansCN-Regular.otf` | UI 文案分层，避免中文字体开销进设备 |
| 退出 | `maix.app.need_exit()` / `set_exit_flag(True)` | 单一退出语义 |

**两条硬约束（影响选型）**：

- `setup.py` 的 `board_names = ["linux", "maixcam", "maixcam2"]` —— 有 **linux manylinux 轮子**，**没有 Windows 轮子**；且 C++ 内核不在本仓库（`components/maix/` 只有 pybind 绑定文件），桌面 Linux 想跑 `maix.*` 得自己编译内核，**不作为方案**。
- MaixCAM1 只有 **128MB 用户内存**，官方与本仓注释都明确「rich 这种纯 Python 大包不值得为禁用功能付这份内存」→ **设备端不可能装 Qt/PySide6**。

---

## 3. 统一方案：四层契约

```
┌─ L4 页面层 ─────────────────────────────────────────────┐
│ Page（状态机）/ UI 语义：几何、布局、命中、动作 → Cmd   │
│   纯逻辑，零像素 API，桌面与设备共用同一份代码            │
├─ L3 输入层 ─────────────────────────────────────────────┤
│ InputSource: Touch / Key / Mouse+Keyboard → 统一事件     │
│   事件坐标一律换算到「UI 空间」；动作一律进 CmdQueue      │
├─ L2 画布层 ─────────────────────────────────────────────┤
│ Canvas（7 原语）+ OverlayRenderer / Compositor          │
│   MaixCanvas(maix.image)  |  CvCanvas(cv2)              │
├─ L1 呈现层 ─────────────────────────────────────────────┤
│ FrameSink / PresenterService（单一 show() 出口 + 节流）  │
│   MaixLcdSink | CvWindowSink | JpegStreamSink | NullSink │
├─ L0 像素层 ─────────────────────────────────────────────┤
│ 像素容器 = 平台原生（D1）：maix.Image(RGB888) | ndarray    │
└──────────────────────────────────────────────────────────┘
```

### L4 页面层：一份语义代码

- 页面/调参页的**几何、布局、命中、参数 schema** 全部写成平台无关的纯逻辑（延续 `framework/ui_state.py` 的 `HitRegion` / `CooldownGate` 风格，已有单测）。
- 页面切换复用 `app/page_sm.py` 的状态机骨架（已在 EOD 分支写好）。
- **UI 只产出命令，不直接改业务状态**：按钮点击 → `Cmd("CALIB")` / `Cmd("TOGGLE_LED")` / `Cmd("EXIT")` → 主循环消费。这同时解掉 SCHED-02（无队列直调）与 DISP-04（退出语义藏在 show() 返回值）。

### L3 输入层：三个后端，一个协议

```python
class InputSource(Protocol):        # 已有，空实现待补
    def poll(self, cmds: CmdQueue) -> None: ...
```

| 后端 | 平台 | 产出 |
|---|---|---|
| `MaixTouchSource` | maixcam1/2 | `maix.touchscreen` → Press/Release/Move/Click |
| `MaixKeySource` | maixcam1/2 | `maix.key.Key` 回调 → KeyDown/KeyUp（切页、退出） |
| `CvInputSource` | Linux/Windows | `cv2.setMouseCallback` + `waitKey` → Mouse/Move/Click/Key |

**坐标统一的关键**：定义一个 **UI 空间**（逻辑面板尺寸，如 640×480），所有按钮矩形在 UI 空间里定义；渲染时由 Canvas 的 transform 映射到实际帧坐标；输入事件反向映射回 UI 空间。设备端 letterbox 用 `image.resize_map_pos`（FIT_CONTAIN），桌面端 1:1。这样同一份页面代码在 640×480 面板、1280×352 相机帧、1024×768 桌面窗口上都能用。

### L2 画布层：D2 定了但没人实现

7 个冻结原语（沿用 D2）：`copy / line / rect / circle / string / blit / size`（+ 必要的 `text_size`）。

- `MaixCanvas`：`maix.image` 委托，保留 sprite 缓存等硬件路径优化（参考 `feat/text-render-accel` 的 `_make_rgb_text_sprite`，避免每帧 `draw_string` 与 RGBA8888 alpha 慢路径）。
- `CvCanvas`：`cv2` 委托，与现有 `BaseDebugWindow._draw_params_info` 同一套画法（`cv2.putText` / `cv2.rectangle`）。
- **护栏**：① 原语集冻结，禁止后端把原生对象泄漏出去；② 金帧一致性测试（两后端渲染同一场景，断言结构等价而非像素全等，因为字体/抗锯齿不同）；③ sprite 缓存属后端私有优化，不进协议。

代价是每个新效果写两份委托（~10~20 行）——这是选「平台原生像素直通硬件 VO」必须付的固定租金。

#### 3.1 第三条路线：单 cv2 后端（`image2cv` 零拷贝），以及为什么不做默认

MaixPy 的 `draw_*` 本身就是 OpenCV 在 `Image._data` 上包一个 `cv::Mat` 视图（见 `maixpy_drawing_backend.md` §4），而 `image.image2cv(img, ensure_bgr=False, copy=False)` **直接借用**同一块内存（官方 `docs/doc/zh/vision/opencv.md`：「更改转换后的图像也会影响到转换前的图像」）。因此在设备端用 cv2 画 HUD 在原理上成立，桌面与设备就能共用**一套**绘制实现，而不是两份委托。

| 维度 | 单 cv2 后端 | 双后端（A） |
|---|---|---|
| 绘制代码量 | 1 份 | 2 份（每原语 ~10 行） |
| 设备端几何图元成本 | ≈ 与 `draw_rect` 相同（同为一次 cv2 调用；MaixCDK 内部就是 `cv::rectangle` 打在 `cv::Mat(_data)` 视图上，见 `maix_image.cpp:1372`） | 最低（pybind 直调，省一层 Python 包装） |
| 文本 | `cv2.putText` 只有 Hershey / ASCII；设备上的 cv2 是否带 contrib 的 `cv2.freetype` **未验证** | `maix.image.draw_string` + TTF（`SourceHanSansCN`） |
| 通道序 | `image2cv(copy=False)` 强制 `ensure_bgr=False`，设备端拿到的是 **RGB** 视图，而 cv2 按键位解释颜色 → CvCanvas 必须显式声明 `channel_order` | 由后端各自吸收，语义层永远只传 `(r,g,b)` |
| 生命周期 | **安全**：`image2cv(copy=False)` 的 numpy 视图以 `py::cast(img)` 为 base（`convert_image.hpp`），视图存活即源 `Image` 存活；反过来 `cv2image(array→Image, copy=False)` 才是危险方向 | 无跨语言借用 |
| 设备端额外能力 | 丢失 `resize` / `to_format` / sprite 的 maix 路径 | 保留 |

结论（2026-11-26 修正）：**几何可以、也应该只做一套 cv2 实现**——两端本来就在跑同一个 OpenCV，只是入口不同（pybind 封装 vs Python 绑定），像素结果一致。真正剩下的分叉只有**文本**（FT2 vs Hershey）和一个必须显式声明的**通道序**参数。因此目标形态是「一份 Canvas 协议 + 一个 `CvCanvas` 负责几何 + 一个 `TextRenderer` 策略」，而不是「两个完整后端」。待真机验证两件事：设备 `cv2` 是否带 `freetype`（有则文本也可统一）、以及逐调用开销差。

### L1 呈现层：回收 EOD 的 `framework/display/`

main 现有 `SinkGroup` + `MaixLcdSink`，EOD 分支有 `PresenterService` + taps。**两者不要并存**，建议以 EOD 版本为主回收进 main，理由：

| 能力 | main (SinkGroup) | EOD (PresenterService) |
|---|---|---|
| 单一 `show()` 出口 | ✓ | ✓ |
| tap 扇出（推流/空实现） | ✗（SinkGroup 能塞，但无协议约束） | ✓ `Tap` Protocol + Lcd/Jpeg/Null |
| 策略单一事实源 | 部分（interval 在 sink 上） | ✓ `PresentPolicy` |
| 平台能力声明 `caps()` | ✗ | ✓ 三后端统一 |
| seq 代数号帧槽 | 身份比较 | ✓ `DisplaySource` |
| 慢格式告警 | ✗ | ✓ `gate.py` |
| show 开销统计 | ✗ | ✓ `metrics.py`（含 p95） |
| 自持线程 | ✗（符合 D4） | ✗（EOD 另有一条 `_display_loop`，反而违规） |

`PixelFormat` 里 **BGR888 被显式标为「慢路径」**（MaixCAM1 `show()` 只白名单 RGB888，其余格式在 show 内部整帧 `to_format(RGB888)`）——这条信息目前只存在于分支里，应该上提到主线文档。

### L0 像素层：D1 保留，不动

设备端 `maix.Image` 直通硬件 VO / 硬 JPEG 编码；桌面端 ndarray。跨平台代码只准经 Canvas 触碰像素。

---

## 4. 调参 UI 的统一（preview_and_tuning.md 的未竟之功）

| 件 | 现状 | 目标 |
|---|---|---|
| **ConfigSchema** | 两份重复 `ParamDef`（字段名还不一致） | 抽一份到 `framework/`，统一 `display/is_odd/scale/step` |
| **ParamPanel** | 只有 cv2 trackbar 后端；触屏 tuner 的滑条是**手写硬编码列表**（`threshold_tuner.py:33 SLIDERS = [...]`） | 两个后端共享同一 schema：`CvTrackbarPanel` / `TouchSliderPanel` |
| **内容签名热重载** | `(st.st_size, sha1)` **已在 EOD 业务层落地**（`gks_task_processor.py:58-70`），但没抽成框架原语；tuner 侧注释仍写「mtime 热重载」→ 口径不一致 | 上提为 `framework/` 原语，改写 tuner 注释 |
| **渲染基座** | `_text/_rect/_in` 在三个 tuner 里各写一份 | 落到 Canvas 层 |

**mtime 同 tick 失效**（实测 300 次写 187 次 mtime 一致）这条坑值得单独立一个框架级原语 + 单测。

---

## 5. 落地路线（按「回收优先」排序）

| 阶段 | 内容 | 产出 | 风险 |
|---|---|---|---|
| **P0 止血** | 给 `vision_manager` / tuner 的绘制加护栏：`framework/`、`modules/` 除平台目录外禁止 `import maix.image`（已有 `scripts/check_platform_isolation.py` 可扩展）；修「按钮无调用者」的死 UI | 桌面端 overlay 不再静默消失 | 低 |
| **P1 回收** | 把 EOD 的 `framework/display/`（types/policy/gate/sources/metrics/service + taps）搬进 main，替换/兼容 `SinkGroup`+`MaixLcdSink`；三后端补 `caps()` | 单一 `show()` 出口 + 策略单一事实源 | 中（有分支冲突，但代码基本逐字） |
| **P2 画布** | 定义 Canvas 7 原语 + `MaixCanvas`/`CvCanvas`；把 `vision_manager` 的 11 处绘制与三个 tuner 的渲染基座迁上去；`hud.py` 一并迁 | 同一份绘制语义跑两端 | 中高（金帧一致性测试是主要工作量） |
| **P3 输入** | 实现 `MaixTouchSource`/`MaixKeySource`/`CvInputSource`；UI 空间坐标统一；动作一律进 CmdQueue | 触摸/按键/鼠标键盘等价 | 中 |
| **P4 页面** | `app/page_sm.py` 骨架进 main；UI 语义与 tuner 抽到 `framework/` 或 `app/display/`；按 D5 分层（能力→平台包，可复用→框架，本届编排→app） | 页面化 UI | 中 |
| **P5 调参统一** | ConfigSchema + 双后端 Panel + 内容签名热重载上提 | 一套参数定义，两种交互形态 | 中 |
| **P6 收尾** | 统一退出函数（disable WDT → flush 日志 → stop manager → exit），`Cmd("EXIT")` 单一出口；删掉 `linux/display.py` 里藏在 show() 的 'q' | 退出路径可审计 | 低 |

---

## 6. 需要明确的取舍

1. **「一套代码四端跑」不可达，只可达「一份语义 + 四套薄后端」。** 别向「GUI 框架统一」的期待去设计，否则会被 D1/D4（原生像素 + 线程预算 ≤4）反复打脸。
2. **设备端不上 Qt/PySide6**（128MB 内存 + 无 aarch64 轮子 + 线程预算）。桌面端要不要额外的「原生 GUI 皮肤」（PySide6/DearPyGui）是**桌面专属体验**，不进入设备端统一抽象，最多作为 `CvWindowSink` 之外的另一个可选 sink。
3. **cv2 `imshow` 必须留在主线程**（HighGUI 约束）——这与 D4「主循环 tick 里 `flush()` 泵」天然吻合；反过来说，**EOD 分支那条独立 `_display_loop` 显示线程是违规的，回收时不要带进来**。
4. **`waitKey` 只能取到按键、取不到鼠标**：`cv2.setMouseCallback` 需要 `cv2.waitKey` 持续泵事件循环——这意味着桌面端的鼠标事件只能在主循环泵里被 poll，与 D4 兼容；但要接受「鼠标事件延迟一帧」。
5. **不要为了统一而牺牲设备端性能**：D1/D3 的理由（零拷贝直通 VO、双核不新增线程）在 EOD 分支已经被验证过。

---

## 6.5 桌面端的「面板模拟器」：让 PC 窗口看起来就是那块屏

设备端的观感来自四件被操作系统默认剥夺的东西：无边框、无系统光标、固定比例、手指而非指针。桌面窗口把四件都还回来即可，不需要换 GUI 框架：

| 观感来源 | 设备端 | 桌面端做法 |
|---|---|---|
| 无边框 | 面板本身没有窗口 | SDL `NOFRAME`（pygame）或 `WND_PROP_FULLSCREEN`（cv2）；`F` 全屏 |
| 固定比例 | 640×480 / 552×368 | 窗口 = 面板 × **整数** 缩放，内容四周黑边居中，不做非整数重采样 |
| 触摸而非指针 | 只有 press/move/release | 隐藏系统光标，画触摸环；`TapGate` 只在「短按且不移动」时产出 tap |
| 同一画面 | `Display.show()` 呈现合成好的面板帧 | 呈现同一张面板帧，窗口只是 1x/2x 的查看器 |

关键点：**UI 必须画在面板空间，而不是相机帧空间**。当前设备端在 1280×352 的帧上画 HUD，再由 `show(fit=FIT_CONTAIN)` 把整帧压到 640×176（缩放 0.5），于是字号被迫写 2 倍、按钮坐标随传感器分辨率漂移；改成在 640×480 的面板上画，桌面与设备就是同一张图，输入也退化成两次减法（`Panel.to_frame`），不再需要 `resize_map_pos_reverse`。

已落地的最小实现（P2 的第一刀，仅桌面可跑，未接生产路径）：

| 文件 | 内容 |
|---|---|
| `framework/ui/canvas.py` | `Canvas` 协议（rect/line/circle/text/text_size）、`(r,g,b)` 入口、字号→行高校准 |
| `framework/ui/panel.py` | `Panel.fit`（FIT_CONTAIN 同规则）、`to_panel`/`to_frame` 坐标换算、`TapGate`（slop + 最长按压） |
| `framework/hal/platforms/linux/canvas.py` | `CvCanvas`：BGR ndarray 上委托 cv2，颜色只在这里转一次 |
| `framework/hal/platforms/linux/panel_window.py` | `DesktopPanelWindow`：pygame 无边框 kiosk（隐藏光标 + 触摸环 + 整数缩放），cv2 兜底 |
| `scripts/ui_panel_demo.py` | 参考实现：合成场景 → 面板合成 → HUD → 窗口；`--png` 离屏出图便于比对 |
| `tests/test_ui_panel.py` | fit/映射/tap 判定/颜色序的契约测试 |

设备端接入方式不变：把 `present()` 换成 `Display.show(panel_image)`，`CvCanvas` 换成 `MaixCanvas`（`maix.Image` + `Color.from_rgb`），其余代码逐字复用。

---

## 7. 文档与代码现状的偏差（做统一前建议先对齐）

| 文档说法 | 代码现状 |
|---|---|
| `preview_and_tuning.md` §2：ConfigSchema / ParamPanel 双后端 / 内容签名热重载 / 渲染基座「应进通用框架」 | **均未进框架**；内容签名只活在 EOD 业务层，ParamDef 两份重复 |
| `framework_issues.md` D2：Canvas 协议 7 原语 + MaixCanvas/CvCanvas | **任何分支都没有实现** |
| `framework_issues.md` MOD-01：绘制迁 `app/display/compositor.py` | main 只有 `framework/ui_state.py`，`compositor.py` 不存在，~500 行绘制仍在 `vision_manager.py` |
| `framework_issues.md` ARCH-01：新增 `Touch(Protocol)` | 未实现；三处触摸各自直连 `maix.touchscreen` |
| P4 遗留项「Touch→InputSource」 | `InputSource` 协议存在，**零实现** |
| `framework_issues.md` D4：呈现由主循环 flush 泵出，sink 不自持线程 | main 符合；**EOD/maixcam1-port 不符**（仍有 `_display_loop` 显示线程） |
| EOD 的 `framework/display/` 整套 | **未登记在任何架构文档**——最完整的一版显示抽象处于「文档外」状态 |
| `AGENTS.md`「四页 KEY_OK 路由」 | 实际 `PAGE_CYCLE` 是 5 项（MAIN + EXPLOSIVE/TARGET/RESCUE/LASER）；且该路由只在 `feat/maixcam1-port` / `EOD2026-NearCam`，**不在 release/2026H** |
| `PROJECT_CONTEXT.md` 的显示描述 | 与代码基本一致，但 main 上触摸屏**不可交互** |

---

## 8. 参考

- **MaixPy 绘制后端的真实机制**：`docs/architecture/maixpy_drawing_backend.md`（MaixCDK `maix_image.cpp` / `maix_display.cpp` 实读：OpenCV 零拷贝视图绘制、FT2 文本、show() 格式白名单与慢路径）
- MaixPy 官方文档（本机 `F:\MaixPy-main\docs\doc\zh\`）：`vision/touchscreen.md`、`vision/display.md`、`vision/image_ops.md`、`vision/opencv.md`、`peripheral/key.md`、`peripheral/hid.md`、`gui/i18n.md`、`basic/app.md`
- MaixPy 上游源码：`setup.py`（board_names / manylinux 轮子）、`projects/app_mouse_simulator/main.py`（触屏当鼠标的实现范式）、`examples/vision/touchscreen/*`
- 仓库内：`docs/retrospective/framework_issues.md`（DISP-01~06、ARCH-01、MOD-01、D1~D6）、`docs/architecture/preview_and_tuning.md`、`docs/architecture/thread_tick_topology.md`
- 分支资产：`feat/maixcam1-port` 与 `EOD2026-NearCam` 的 `framework/display/`、`app/page_sm.py`、`app/page_router.py`、`modules/zw_opencv_module/display/hud.py`、`modules/zw_opencv_module/debug/*_tuner.py`