# C3 + C4 实现规格（Presenter 收敛 + 推流迁出视觉线程）

## 目标（一句话）
全系统只有一个对象能调 `display.show()`：`PresenterService`。两份重复的
`_maybe_show` 与 `_ThrottledPageDisplay` 全删；推流不再由视觉线程发起。

## 为什么现在做
- `MainUiShell._maybe_show` 与 `DebugPageFactory._maybe_show` 是同一套策略的两份拷贝
  （enabled / interval / frame-skip / 计时 / fmt 探针），语义靠注释保证一致。
- `_ThrottledPageDisplay` 是"缺一层抽象"的补丁（只拦 show 的代理）。
- 推流由 `vision_manager._capture_sink` 在视觉线程里发起，违反"显示侧负责出门"。

## 新增文件
```
framework/display/service.py    PresenterService
framework/display/policy.py     PresentPolicy
framework/display/metrics.py    Metrics（预分配环形缓冲）
framework/display/taps/base.py  Tap 协议
framework/display/taps/lcd.py   LcdTap
framework/display/taps/jpeg.py  JpegTap（复用现有 JpegStreamer）
framework/display/taps/null.py  NullTap
framework/display/sources.py    DisplaySource（seq 计数 + 按需创建）
```

## 核心接口
```python
class PresentPolicy:
    enabled: bool = True
    interval_ms: int = 0          # 0 = 每 _DISPLAY_SHOW_SKIP 帧
    hud_interval_ms: int = 0

class PresenterService:
    def __init__(self, display, taps: list[Tap], policy: PresentPolicy): ...
    def submit(self, frame, *, source: str = "main") -> bool:
        """唯一出口。按 policy 限流；向每个 attached tap 分发最多一次。
        线程安全（内部单锁）。零 attached tap 时不编码。"""
    def stats(self) -> dict: ...
    # 兼容垫片：让三个 tuner 的 disp.show(img) 继续可用
    def show(self, frame, *a, **k): ...   # == submit(frame)
    def width(self): ...; def height(self): ...

class Tap(Protocol):
    name: str
    def attached(self) -> bool: ...
    def offer(self, frame, seq: int) -> None: ...   # 不得阻塞
```

## 分三步落地（一个 commit 内完成，但按序验证）

### 步 1 — 引入 PresenterService，接 MAIN 路径
- `MainUiShell._display_loop` → `presenter.submit(frame, source="main")`
- 删除 `MainUiShell._maybe_show` / `_note_show_perf` / `_cmm_pressure_throttled` /
  `_display_enabled` / `_show_interval_s` / `_display_counter` / `_show_*` 字段
  （策略全搬进 PresentPolicy；CMM 压力检查搬进 service）
- `set_display_policy()` 保留为薄封装，转发到 `presenter.set_policy(...)`
  （`main()` 与既有测试都调它）

### 步 2 — 调试页改用同一个 Presenter
- `DebugPageFactory.step_active` 里 `page.step(self._page_display(ts), ...)` 改为
  `page.step(self._presenter, ...)`（presenter 实现 show/width/height，直接当 disp 用）
- 删除 `_ThrottledPageDisplay`、`DebugPageFactory._maybe_show` / `_note_show_perf` /
  `_reset_show_clock` / `_display_enabled` / `_show_interval_s` / `_display_counter` /
  `_last_show_time` / `_last_show_tag`
- **保留** `_note_page_iter` / `_flush_page_iter`（调试页迭代归因，与 show 节流无关）
- 页面进/出清相位 → `presenter.reset_phase()`

### 步 3 — 推流迁出视觉线程（C4）
- 删除 `VisionManager.set_capture_sink` 与 `_update_display_frame` 里的 sink 调用
- 新增 `JpegTap`：包住现有 `modules/zw_wifi_stream/streamer.py::JpegStreamer`
  （它的 `push_frame` 已按 fps 节流、`to_jpeg` 编码、max_width 降采样 —— 直接复用，
  不要重写编码逻辑）
- `_init_streamer` 改为：若 streamer 存在，把它作为 JpegTap 注册到 presenter；
  若 `stream.enabled=false`，注册 NullTap
- `vision_manager` 不再持有 capture_sink

## 必须保住的行为（既有测试钉住的）
1. `display_enabled=False` → 一次都不 show（但 `_display_frame` 照常每帧发布）
2. `interval>0` → 按时间限流；`interval==0` → 每 `_DISPLAY_SHOW_SKIP` 帧
3. 缺键/None → 不改变当前值
4. **只拦 show，不拦 step**：调试页 touch/识别每帧照跑（`test_debug_page_display.py`）
5. show 抛异常不冒泡（否则 step_active 会把页面踢回 MAIN）
6. `cam.read() is None` → 安静退让
7. 进页首帧必上屏（相位重置）
8. `competition_mode` → 跳过 HUD 与推流，但 `_display_frame` 每帧发布
9. 激光退出兜底关帧、wdt disarm 路径不变
10. `[PERF] lcd show` / `[PERF] page show` 行格式尽量保持（用户对照用），可加 `src=` 字段

## 硬约束
- 注释 **brief**（1 行 + �� 标签），不要小作文
- 不修改既有测试文件；如某测试钉的是错误语义（如已修的 capture_sink 计数），
  在报告里明确指出并说明为何改
- 热路径零额外分配；f-string 只在真要打印时拼
- `main.py` 净减行数（这是这次重构的验收指标之一）

## 验收
- `python -m pytest tests/ -q` 全绿（当前 323）
- 新增测试：PresenterService 限流/分发/异常隔离、JpegTap attached 语义、
  source 按需创建
- 报告 `main.py` 行数变化
