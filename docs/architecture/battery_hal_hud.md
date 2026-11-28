# 电量 HAL 与主屏显示：设计

> 2026-11-02 · 修订 2026-11-03（约束改为"实时性不是需求"，主用 MaixCAM-Pro）
> 分支 `feat/explosive-dock-trim`（`platform: maixcam1` = MaixCAM-Pro）
> 前置事实来源：`F:\2026工科赛\docs\调研-MaixPy读取电池电量.md`（源码级，未上机）
>
> **状态（2026-11-03 晚）**：**全链路已落地并上机** —— HAL、`battery_sampler`、
> 采集节拍（`modules/zw_battery_module`，主循环 `monotonic` 判据，无线程）、
> HUD 第二行右对齐、`project_config.yaml` 的 `battery:` 段。
> 读数耗时已在 Pro 上实测（§10.1：一次完整读中位 1.02ms，加 `is_charging` 后 ~1.30ms），
> 其余未标"实测"者仍为推断。
>
> **2026-11-03 实机反馈修正**：控制台"看不见 `[BAT]`"有两个原因，都已修 ——
> ① 原来只在百分比变化时打印，电量稳定 = 永久静默；现改成**每拍一行**（10s 一行，
> 同时落 `logs/debug.log` 与控制台），只对无读数限流 5 分钟。
> ② `logger.warning` 只进 `logs/app.log`（`basicConfig` 只挂 FileHandler），
> 失败原因在控制台完全不可见。**修法不是"改 log_print"，而是把原因塞进读数**：
> `BatteryReading.reason` → `BatterySnapshot.reason` → 模块那一行。HAL/采样器一律不打日志，
> `[BAT]` 只剩主循环里这一个出口（见 §7.5）。
>
> **不做独立 `logs/battery.log`**（2026-11-03 决策）：`logging.FileHandler` 是**调用方线程
> 同步 write+flush**（实测 39µs/行），而 `log_print` 只入队（9µs，写盘在 log-writer 线程）。
> 主循环里已经有一次可能阻塞的 I2C 读，再开第二条同步 I/O 路径不值得。存档就用
> `debug.log`，详见 `docs/architecture/logging_pipeline.md`。

## 0. 现状

设计过两版：第一版给电量单开监控线程，第二版挂到显示线程。两版都没落地——
线程是多余的生命周期，显示线程会让数据源耦合到"当前在看哪一页"。
**最终选了第三条：主循环慢节拍**（§2），采集与"当前在看哪一页"解耦，也不新增线程。

已实现的边界：

| 层 | 状态 | 说明 |
|---|---|---|
| `Battery` / `BatteryReading` / `NullBattery` | 已落地 | 一个同步 `read()`，返回 `percent=-1` 表示无电池 |
| `maixcam1/maixcam2/linux/mock` | 已落地 | Pro 走 `pmu`，CAM2 走 sysfs，其余返回 -1 |
| `Machine.battery` | 已落地 | create 阶段构造（Pro 上 open i2c-4，~2.4ms），失败不致命 |
| `BatterySampler` | 已落地 | 读一次 + 抗跳变 + 发布 `frozen` 快照 |
| 采集节拍 | 已落地 | `modules/zw_battery_module`，主循环调用，`time.monotonic()` 判据 |
| HUD 上屏 | 已落地 | 第二行右对齐，无 I/O，见 §7 |

## 1. 约束（读码确认）

- HUD 画在视觉线程：`VisionManager._process_loop` → `_update_display_frame` →
  `hud.draw()`（`vision_manager.py:356`）。同线程每帧跑 find_blobs / YOLO。
  节流 100ms = 10Hz（`vision_manager.py:348-351`），且跳过绘制的帧连发布一起跳过。
- 主循环 2ms（`framework/module_manager.py:16`），串行跑 `module.loop()` →
  `coordinator.loop()`（DATA_STREAM 推流）→ `tick_callback` → `display_callback`，
  并在循环开头喂 WDT。
- 显示线程 5ms 轮询（`app/main.py:475-487`），只在 MAIN 页工作；`presenter.submit`
  持自己的锁调 `show()`，实测 2.3~6.2ms/次。
- 协议 v2.1 payload 固定 10B，没有电量字段。电量不上报，只上屏 + 打日志。

## 2. 读在哪：五个位置，三个指标

实时性取消后，判据从"每秒开销"变成"卡住时谁跟着停"。

| 位置 | 正常代价 | 卡死（I2C 总线超时）时 | 判定 |
|---|---|---|---|
| 视觉线程每帧读 | 每帧 +0.3~1.2ms，直接吃识别预算 | 推流归零心跳 → Brain 判 stale → 重订阅风暴 | 否 |
| 视觉线程每 10s 读 | 每 10s 一帧 +1.2ms | 同上，整条视觉流水线停 | 否 |
| **主循环每 10s 读（monotonic 判据）** | **实测 1.02ms，10s 一次** | **DATA_STREAM 推迟约 1ms（50ms 帧间隔的 2%）；卡到秒级才可能碰 WDT 窗口** | **选定（`zw_battery_module`）** |
| 独立线程 | 一次唤醒/10s | 只有它自己停 | 过度设计 |
| 显示线程每 10s 读 | 实测 1.02ms，10s 一次 | 屏幕停更；视觉、推流、WDT 全部照常 | 否（数据源耦合"在看哪一页"） |

两条候选的差别不在性能，在两个工程属性上：

- **主循环**：一直在跑，数据源不依赖"当前在看哪一页"；代价是卡死时落在控制路径上。
  10s 一次、单次毫秒级这条量级下，DATA_STREAM 只推迟一个 2ms tick 内的抖动。
- **显示线程**：卡死时爆炸半径最小（只冻结屏幕）；代价是 `_display_loop` 非 MAIN 页
  直接 `continue`，离开主屏就不再采样，以后调试页想显示电量、或想写日志，拿到的都是
  停更的值。

2026-11-03 在 Pro 上量到了（§10）：一次完整读（电量 + 电压）**中位 1.02ms，max 1.19ms**。
摊到 10s 是 0.010% 占空比；落在主循环上就是把 2ms 的一拍拉长到 3ms，10s 一次。
这个量级不构成选择依据 —— 两条候选都安全，区别只剩下面那个工程属性。

顺带一条不管选谁都成立的约束：**`poll()` 不能放进 `PresenterService.submit()` 内部**。
那个函数持 `presenter._lock` 调 `show()`，而锁是 MAIN 页和调试页共用的——把可能阻塞的
第三方调用塞进持锁区，一旦卡死，主循环调 `submit` 时会永久阻塞。

## 3. 数据形状：只留两个量（已实现）

`framework/hal/interface/battery.py`：

```python
@dataclass(frozen=True, slots=True)
class BatteryReading:            # HAL 只报数，不判断够不够用
    percent: int = -1            # 0~100；-1 = 无电池 / 不支持 / 读取失败
    millivolts: int = -1         # -1 = 不可用

@runtime_checkable
class Battery(Protocol):
    def read(self) -> BatteryReading:
        """阻塞读一次；失败或无电池返回 percent=-1，不抛异常。"""
```

Pro 上每个量都是一次独立的 I2C 事务：`0xA4` 电量、`0x34/0x35` 电压。所以字段不是免费的：

- `percent` 必读（要显示）。
- `millivolts` 读（低电判据比百分比可靠，调研 §2.3；也进日志）。
- `charging` 读（STATUS2，实测 0.28ms）。屏幕要按充电态换色，这 27% 付得起；一次完整读
  因此从 1.02ms 变成约 1.30ms。Pro 用 `is_charging()`，CAM2 用 `status` 是否 `Charging`/
  `Full`；判定不出来给 `None`，屏幕按普通电量处理。
- `present` 不设：`percent < 0` 已经是"没有有效读数"，两个字段表达同一件事早晚不一致。

时间戳不在 `BatteryReading` 里，在采样器的 `BatterySnapshot` 里（见 §4）。HAL 不该知道
"什么时候读的"——那是调用方的事。`frozen` 是为了跨线程传引用不被改写；`slots=True`
是因为 CAM1 用户内存实测已占 80%。

`NullBattery` 与 Protocol 同文件（`interface/` 下唯一的实现类，破例换平台工厂统一返回
非 None、HUD/采样器少两个 None 分支）：

```python
class NullBattery:
    def read(self) -> BatteryReading:
        return BatteryReading()          # 用户要的"默认实现返回 -1"
```

## 4. 采样器：没有线程、也没有定时器（已实现，无调用者）

`framework/hal/battery_sampler.py`，与 `camera_hub.py` 同层——它管 HAL 对象的调用策略，
和 CameraHub 管相机是同一种职责。

```python
@dataclass(frozen=True, slots=True)
class BatterySnapshot:          # 渲染路径读它
    percent: int = -1
    millivolts: int = -1
    level: str = "charging" | "ok" | "low" | "unknown"，写线程算好
    ts: float = 0.0             # 读者据此判陈旧，无需额外锁

class BatterySampler:
    def poll(self) -> None:     # 读一次、判一次、发布一次。节拍由调用方给
    def snapshot(self) -> BatterySnapshot:      # 一次引用读，无锁
```

**节拍不在这个类里**，这是有意的：把它放进来就等于提前决定了"谁多久调一次"。
主循环、显示线程、独立线程三种放法都能直接调 `poll()`，位置定了只改调用点。

`snapshot()` 没有锁：单写者、单读者、写的是 `frozen` 对象的一次引用赋值，CPython 下
不会撕裂。这是整套设计里唯一需要想清楚的并发点。

`poll()` 是全 try 包裹的：平台实现不该抛，抛了也不能让调用方的循环停住。
`last_ms` 记录每次读的墙上耗时，调用方可以据此判断"总线是不是坏了"（见 §10）。

零 sleep 测试从这里来：`poll()` 是"给什么电池就读什么"，脚本化假电池 + 直接调
`poll()` 就能断言等级、跳变、异常保持，不需要线程也不需要时钟。

### 4.1 异常值规则

只剩一条：**不合理的下降要连续两次确认，上升立即采纳。**

判据就是一行 `prev > 0 and new <= prev - 20`。读不到 0 的情况也在里面（`prev >= 20` 时
`0 <= prev - 20` 成立），不需要单独规则。

理由不是经验法则，是调研 §3.5 的明确语义：Pro 的 `get_bat_percent()` I2C 失败分支
`return false` → Python 收到 **0**，与"真的没电"不可区分。若直接采纳，屏幕会在一次
总线抖动后显示 0% 并把颜色刷红，一个周期后才恢复——比不显示更糟。连续两次说明要么
真的耗尽了，要么总线一直坏，两种都该如实显示。

小变化（80→78）立即采纳：单次噪声最多让黄色提前一个周期出现，无害。

20 这个阈值有物理依据：一个周期内掉 20 个点意味着电芯已短路，那时设备早就断电了。
边界（正好 20 个点）算"不合理"，要确认两次——这条被测试钉住，见
`test_drop_of_exactly_twenty_points_needs_confirmation`。


## 5. GIL：线程选择只保护正常情况

这一条容易被忽略，但它决定了上面"10s"这个数字的重要性。

MaixCDK 是 pybind11 绑定。pybind11 默认**不释放 GIL**，除非显式 `call_guard<
py::gil_scoped_release>()`。`maix.ext_dev` 的 I2C 读有没有释放，我手上没有 C++ 源码可查
（`F:\MaixPy-main` 只有 Python 层），所以这是**推断，未验证**。

两种情况的后果不同：

- 释放了 GIL：这 1ms 只阻塞调用它的那条线程，位置选择就按 §2 的影响面来挑。
- 没释放：这 1ms 是**进程级**停顿，视觉线程和主循环一起等。这时放哪个线程都一样。

所以真正起保护作用的是**周期**，不是线程：1.2ms / 10s = 0.012%，即使全程持 GIL 也在
测量噪声以下。第一版把它设计成 2s（现在是 10s），假设的是"GIL 已释放、只要避开热线程"；
现在这个假设对不对都不影响结论。

验证办法在 §10：临时把 `poll_interval_ms` 调到 500（20 倍频），看 `[PERF] mainloop ...
gil_wait=` 与 `[PERF] vision ... gil_wait=` 是否随读次数线性抬升。抬 = 没释放。

## 6. HAL 的平台实现

### 6.1 maixcam1（Pro，本次目标）

```python
class MaixCam1Battery:
    """AXP2101 用户态 I2C。句柄常驻，构造一次。"""

    def __init__(self) -> None:
        self._pmu = None
        self._dead = False

    def read(self) -> BatteryReading:
        if self._dead:
            return BatteryReading()
        if self._pmu is None:
            try:
                from maix.ext_dev import pmu   # 老固件没有 ext_dev -> 退化为无电池
                self._pmu = pmu.PMU()         # i2c-4 open + 芯片 ID 校验
            except Exception:
                self._dead = True             # 一次失败即放弃，不再重试刷日志
                return BatteryReading()
        try:
            pct = int(self._pmu.get_bat_percent())
            mv = int(self._pmu.get_bat_vol() or 0)
        except Exception:
            return BatteryReading()
        if pct < 0:
            return BatteryReading()           # -1 = 没插电池
        return BatteryReading(percent=min(100, pct),
                              millivolts=mv if mv > 0 else -1)
```

三处细节，都是调研里写明的坑：

- `mv if mv > 0 else -1`：`AXP2101::init()` 没使能 ADC，`get_bat_vol()` 可能恒 0。
  0mV 不是有效读数，报 -1 比报 0 诚实（0 会被下游当"电芯已死"）。
- `self._dead` 而不是无限重试：`pmu.PMU()` 抛异常意味着板子上没有这颗 PMIC 或 I2C 打不开，
  每 10s 重试一次的收益是零，代价是日志。
- 句柄常驻：MaixCDK 侧是带引用计数的进程内单例，重复构造只打 warning 并共用，但每次
  构造仍要跑一遍芯片 ID 校验（一次 I2C 事务）。每 10s 重造 = 白花一倍事务。

一条**给未来的人**的约束：`pmu.PMU()` 只许在这里构造。别的地方（比如以后调试页想显示
电量）要读，必须走 `machine.battery` / `sampler.snapshot()`。两个 PMU 对象在 C 层是同一份
`i2cdev`，但没有锁，并发读会在字节层面交错。

### 6.2 maixcam2

sysfs 直读，官方口径照抄：

```python
_DIR = "/sys/class/power_supply/cw2015-battery"
min_poll_s = 8.0                  # 内核 CW2015 驱动 8s 才更新一次

def read(self):
    try:
        cap = int(_read_text("capacity"))          # 0~90
    except (OSError, ValueError):
        return BatteryReading()                    # 节点不存在 = 没装电池 / 驱动没起
    try:
        mv = int(_read_text("voltage_now")) // 1000
    except (OSError, ValueError):
        mv = -1                                    # 拿不到电压不影响百分比
    return BatteryReading(percent=min(100, cap * 100 // 90), millivolts=mv)
```

与"调用官方函数"的偏差：`pmu.PMU()` 的 maixcam2 分支内部就是读这个 `capacity` 再 ×100/90，
**百分比是同一个数**。差别只有两点——官方封装把 B29 复用成 GPIO，且 `get_bat_vol()`
直接 `return 0`。当前工程没占用 B29，所以那不是冲突，是一个无谓的引脚副作用：哪天有人
用 B29 接按键，症状会是"电量读得挺好，那个功能突然不工作"。

没做 `driver: pmu` 开关：一条没人走的配置分支比"改 6 行代码换实现"更难维护。真要用官方
函数，把 `read()` 里那几行换掉即可，调用方不用动。

### 6.3 linux / mock

`create_battery()` 返回 `NullBattery()`，两行。

## 7. 显示（已实现，见 `display/hud.py::_draw_battery`）

### 7.1 位置

第二行右对齐：`y = _BAR_H + 2 = 32`，`scale=1.2`，`x = img.width() - 8 - 文本宽`。

第一行右侧已被 `T:FOUND`（x = w-240）与 `FPS`（x = w-130）占满，中间只剩约 130px，
`BAT:100%` 在 scale 1.5 下正好 96px——塞得进去，但没有余量。第二行左侧最长约 390px
（TARGET：`CODES:[…]  ORDER:[…]  ERR:+1234`），右侧 400~632 全空。IDLE 时第二行只有左边
一行字，电量照样可见，赛前点检正需要这个。

### 7.2 绘制路径

```python
def _draw_battery(self, img) -> None:
    if self._battery_fn is None:
        return
    snap = self._battery_fn()                     # 无锁无 IO
    if snap.percent != self._bat_pct:             # 每次变化才重算
        self._bat_pct = snap.percent
        self._bat_text = "" if snap.percent < 0 else f"BAT:{snap.percent}% {mv/1000:.2f}V"
        self._bat_w = self._measure(self._bat_text, 1.2) if self._bat_text else 0
    if not self._bat_text:                        # 无电池/未采到: 不画
        return
    stale = time.monotonic() - snap.ts > self._bat_stale_s
    color = self._c["gray" if stale else _BAT_COLORS.get(snap.level, "gray")]
    # _BAT_COLORS: charging 绿 / low(<20%) 红 / ok 白 / unknown 灰
    self._text(img, img.width() - 8 - self._bat_w, _BAR_H + 2,
               self._bat_text, color, scale=1.2)
```

每 HUD 帧：1 次 bound method 调用 + 1 次 int 比较 + 1 次 `monotonic()` + 1 次
`draw_string`。文本和 `string_size` 只在百分比变化时做（全场 ≤100 次）。台架实测
`[PERF] hud 9.6Hz avg=6.98ms`（§10.3）—— 新增一行是其中约 1/20 的绘制量，10Hz 下
每秒约 +2~3.5ms，占视觉线程 wall time 的 0.2~0.35%，`[FPS] pipeline` 看不出差别。

无电池时画不画是个取舍：画 `BAT:--` 便于点检，但也会让"没装电池"看起来像"功能坏了"。
默认不画（`percent < 0` 直接 return），靠日志告知：启动一行
`[BAT] poll=10s 每拍一行（首拍即采）driver=MaixCam1Battery`，之后**每 10s 一行**
`[BAT] 87% 7980mV ok read=1.03ms`（同时进控制台与 `logs/debug.log`）。

每拍都打是必须的，2026-11-03 实机踩过：原来只在 `percent != _last_logged` 时打印，
电量稳定 = 开机两行之后永久静默。现场无法区分"10s 采一次、只是没变"与"`loop()` 根本
没被调"——后者会让整块 HUD 静默死掉，而屏幕上什么都看不出来；存档里也断了大段。
`percent=-1` 也照打，**没有第二套策略**（代价是没装电池的板子 6 行/分，要安静就
`battery.enabled: false`）。

`battery_stale_s` 取定值 **30s**（不是 `max(30, 3*interval)`）：正常 10s 一更，30s 没刷新
说明采样卡住/总线坏了，数字转灰。退避到 60s 时数据**确实已经旧了**，那时转灰是正确表现，
跟着放大阈值反而会把陈旧读数当新鲜的显示。

第二行最长出现在 TARGET + `THR:MISSING`（57 字符 ≈ 547px，右端到 553），会压到右对齐的
电量行（起点 498）。所以 `_draw_line2` 缓存当前文本宽度，撞车时电量退化成 `BAT:87%`，
再撞就整行不画 —— 宁可少一个数字也不叠成一团。宽度只在文本变化时量一次。

调用点：`DisplayHUD.draw()` 里 `_draw_line2` 之后。**不能塞进 `_draw_line2`**——它在
IDLE 时提前 return（`hud.py:177-179`），而 IDLE 恰恰是最需要看电量的时刻。

### 7.3 为什么不在显示线程里画

最自然的想法是"既然读放在显示线程，画也放过去，视觉线程彻底不碰电量"。不行。

`_update_display_frame` 发布的是 `camera.last_raw` 的引用，不是拷贝。显示线程若在
`submit()` 里往这张图上画，视觉线程下一轮 `read_raw()` 可能拿回同一块缓冲（CAM1 只有
双缓冲），于是"读一帧"和"画上一帧"落在同一块内存上——撕裂，而且是偶发的、随帧率比例
变化的撕裂。HUD 之所以画在视觉线程，就是为了保证"画"和"读"是同一线程的先后顺序。

所以：**数据可以在任意一侧产出，像素只能在视觉线程合成**，两者只通过那个 `frozen`
快照交汇。这条不随 §2 的选择改变。

### 7.4 不这么做

- 不进 `coordinator.get_info()`：那是协议状态（订阅/锁存/推流），每帧重建 dict。把硬件
  塞进去，等于让"谁在读 I2C"不可追踪，且 coordinator 要多持有一个 HAL 对象。
- 不走 DebugConsole：`debug_console_enabled: false`（当前默认）时 `_push_coordinator_status`
  直接 return，键根本不存在，HUD 的 `_draw_debug` 画不出东西。
  顺带一个既有问题：`hud.py:209` 每帧 `from utils.debug_console import DebugConsole`，
  该模块顶层 `import rich`；默认配置下没人 import 过它，所以**第一次画 HUD 就会把 rich
  拉进内存**——与 `project_config.yaml` 里"禁用时连 rich 都不导入"的说法不符。修法是先判
  `log_util.debug_console_active()`。电量不能挂在这条链上。

### 7.5 日志出口：一个地方，一条路径

```
[10s] zw_battery_module._poll_if_due
        └─ snap = sampler.poll()      ← 返回新发布的快照；没发布（读异常/跳变被拒）返回 None
              └─ if snap is not None: log_print(_format(snap))
                    └─ 队列 ──► log-writer 线程 ──► logs/debug.log + 控制台
```

三条约束，都是"换个人改就会破坏"的那种：

1. **HAL 与采样器不打日志**。HAL 报事实（`BatteryReading.reason`），采样器报状态
   （`percent/level`），格式只属于出口。否则同一个 `-1` 会有两句话，而控制台
   只看得到其中一句。
   例外：采样器对 `read()` 抛异常、跳变被拒这两个**即时事件**仍就地打一行 —— 它们
   没有对应的快照（一个是提前 return，一个是故意不发布），没地方挂。
2. **写盘的不是主循环**。`log_print` 只做 `Queue.put_nowait`（实测 9.1µs），真正的
   `open/write/flush/close` 在 `log-writer` 线程。这条是全局约定，不要为任何单个功能
   在主循环里加 `logging.FileHandler`（它是调用方线程同步写，实测 39.1µs/行）。
3. **判据是"有没有新快照"，不是"值变没变"**（2026-11-03 定）。`poll()` 把"是否发布了
   快照"作为返回值交给调用方，模块因此不需要 `_last_logged` / 限流时间戳这类状态，
   也不会在读失败时把旧值当新值重打一遍。8640 行/天 ≈ 730KB/天。

- 不在 HUD 里算阈值/去抖/电压换算：那些是策略，属于采样器。
- 管道本身的细节（闸门、writer 线程、成本、缺口）见 `docs/architecture/logging_pipeline.md`。

## 8. 配置（随 §2 的位置一起定，现在还没读）

```yaml
battery:
  enabled: true
  poll_interval_ms: 10000    # CAM2 会被 min_poll_s=8000 抬到 8s；Pro 无下限
  low_percent: 20
```

缺整段 = 当前行为（不显示）。`enabled: false` 时不建采样器。

## 9. 失败模式

| 现象 | 表现 | 兜底 |
|---|---|---|
| 板子没装电池 / Pro 返回 -1 | 不画 | `percent < 0` 分支 |
| 老固件无 `maix.ext_dev` | 同上 | ImportError 捕获一次，`_dead=True` |
| I2C 偶发失败返回 0 | 不闪 0% | 不合理的下降要连续两次 |
| I2C 总线卡住（i2c-4 与 RTC/IMU 共用） | 取决于 §2 的选择 | 快照 30s 不更新 → HUD 变灰 |
| AXP2101 的 ADC 未使能 | 日志里 `mv=-1` | 只影响日志，不影响百分比 |
| 采样器异常 | 快照停更 | `read()` 整体 try，日志只记一次 |
| 日志里没有 `[BAT]` | 模块没进部署包，或 init 抛异常 | 之前这条是**静默**的：`ModuleManager.load` 只 `traceback.print_exc()`，走 stderr，而控制台的 `[HH:MM:SS]` 行走 stdout —— 失败完全看不见。现在失败会打 `[MODULE] xxx 加载失败`，main.py 还会打 `[BAT] 未接入` |

## 10. 验证

### 10.1 已做（2026-11-03，MaixCAM-Pro 实机，`python scripts/battery_probe.py`）

```
device_id=maixcam_pro
[AXP2101] Find AXP2101 PMU, chip version: 0x4a.
pmu.PMU() 构造         2.375 ms
get_bat_percent()      value=9   min 0.474  中位 0.500  p90 0.564  max 0.592 ms
get_bat_vol()          value=3361  min 0.495  中位 0.523  p90 0.562  max 0.607 ms
is_charging()          value=False  min 0.271  中位 0.279  p90 0.348  max 0.383 ms
```

由此定下三个数：

- **一次完整读（电量 + 电压）= 中位 1.02ms，最坏 1.19ms。** 之前按总线时钟推的
  0.3~1.2ms 落在这个区间的上半段，推断偏乐观但没出圈。
- **构造 2.375ms 是一次性的**，懒建放在第一次 `read()` 里，所以第一次采样是 3.4ms，
  之后回到 1.02ms。10s 节拍下不值得为它做预热。
- **单次 I2C 事务 ≈ 0.26ms**（`is_charging()` 只要一次事务，0.28ms；`get_bat_percent()`
  要两次 —— 先读 STATUS1 判在位、再读 `0xA4`，0.50ms）。按 39 个时钟周期算，400kHz 的
  线上时间是 0.098ms，所以**其中约 0.18ms 是 i2c-dev 的系统调用与驱动固定开销**。
  这条决定了优化方向：想再快，减少事务数比提高总线时钟有效（把百分比的两次读合成
  一次 `I2C_RDWR` 的 msgs 列表能省约 0.18ms）—— 但在 1.02ms/10s 面前不值得做。

第二次跑（插上 USB 充电，90s 后）: 构造 2.447ms；`percent` 中位 0.503、`vol` 0.509、
`is_charging` 0.290ms，读数 `8% / 3422mV / charging=True`。**两次运行中位数差 <5%**，
所以 1.02ms 可以当定值用，不必再按区间估。

另外 `get_bat_vol()` **不是恒 0**（Pro 的 ADC 是开着的，3361→3422mV），调研 §2.2 里
"maixcam2 分支恒 0" 的说法只适用于 CAM2；`is_charging()` 也确实会随插电从 False 翻到
True，要加充电指示的话代价是已知的 0.29ms。

两次读数放一起有个反常：插电后电压 3361→3422mV、`is_charging()` 由 False 变 True，但
百分比 9→8。AXP2101 的电量计是库仑计 + 电压混合，长期存放后要一次完整充放才准；两个
采样点不足以下结论，但**赛前点检应该看电压，不要只看百分比**——调研 §2.3 的判断在这块
电池上得到了支持。

### 10.2 还没做（等采集位置定了再做）

1. GIL 判定：周期临时设 500ms（20 倍频），对比 `[PERF] mainloop ... gil_wait=` 与
   `[PERF] vision ... gil_wait=` 的基线。随读次数线性抬升 = C 层没释放 GIL，那么周期只能
   往长调，不能往短调。以 1.02ms 算，10s 节拍下即使全程持 GIL 也只有 0.010%，所以这条
   只影响"以后想把周期调短"的余地。
2. 帧率对照：开/关电量各跑 60s，比 `[FPS] pipeline` 与 `[PERF] vision ... disp=`。
   **判定阈值：差值 <1%（落在噪声内）；若 >1%，说明读跑错线程了。**

### 10.3 台架快照（2026-11-03，MaixCAM-Pro 常态运行）

```
[PERF] vision 57.7Hz loop=17.3ms body=16.0ms cpu=7.1ms gil_wait=8.9ms
       read=12.4 conv=0.0 task=0.9 disp=1.7
[PERF] hud 9.6Hz avg=6.98ms
[PERF] lcd show 6.6Hz avg=2.4ms
[PERF] mainloop 89Hz body=0.82ms cpu=0.64ms gil_wait=0.18ms
```

两条与电量无关但推翻了我先前假设的事实：

- **`[PERF] hud avg=6.98ms`**，不是 `project_config.yaml` 注释里引用的 0.1ms（那是更早
  一次测量）。所以"再加一行 `draw_string` 无所谓"这个结论仍然成立（7ms 里的一小部分），
  但依据要换成这条新数据。
- **主循环 89Hz**（11.2ms/拍），不是名义的 500Hz/2ms。`sleep(2ms)` 被 Linux 定时器粒度
  拉长，body 只有 0.82ms。于是"每 5000 拍采一次 = 10s"实际是 56s —— 计数器不能当时间
  单位，已改成 `time.monotonic()` 判据。

拔电池验证：Pro 应返回 -1（不画），且日志只有一条。

## 11. 落地状态

已落地（`python -m pytest tests/` 全绿，含 21 例电量用例）：

| 文件 | 内容 |
|---|---|
| `framework/hal/interface/battery.py` | `BatteryReading` / `Battery` / `NullBattery` |
| `framework/hal/interface/__init__.py` | 导出 |
| `framework/hal/platforms/maixcam1/battery.py` | `pmu` 通道，create 阶段构造一次（`~2.4ms`） |
| `framework/hal/platforms/maixcam2/battery.py` | sysfs，`min_poll_s=8` |
| `framework/hal/platforms/{linux,mock}/__init__.py` | `create_battery()` → `NullBattery` |
| `framework/hal/platforms/{maixcam1,maixcam2}/__init__.py` | `create_battery()` |
| `framework/hal/machine.py` | `Machine.battery`（create 阶段），失败不致命 |
| `framework/hal/battery_sampler.py` | 读一次 + 抗跳变 + 发布快照 |
| `modules/zw_battery_module/__init__.py` | 主循环节拍：monotonic 判据 + 每拍一行 + 退避/回落 |
| `modules/zw_opencv_module/display/hud.py` | `battery_fn` 参数 + `_draw_battery` + 第二行避让 |
| `app/main.py` | 建采样器、接线到 HUD、`battery_stale_s=30` |
| `project_config.yaml` | `battery:` 段（`enabled` / `poll_interval_ms` / `low_percent`） |
| `tests/test_battery_hal.py` | 接口契约 + 两个平台的读数语义 |
| `tests/test_battery_sampler.py` | 快照语义 + 跳跃阈值 + 每拍落盘/无读数限流/退避回落 |

## 12. 不做

- 不上报协议。payload 固定 10B，Brain 已对齐；整车主电该在 panzer 侧用 ADC 测
  （调研 §7），不要为这个在视觉板加分压电路。
- 不做低电自动停机/降级。只显示，是否动作由人决定——比赛里因一次误读自动断流，代价远
  大于电量耗尽本身。
- 不改内核 CW2015 的 8s 轮询，也不为了"读得更快"直接碰 I2C。
- 不在 `competition_mode: true` 时停轮询：10s 一次 1ms 出头不值得增加一条耦合，而关掉 HUD
  后日志里的 `[BAT]` 仍是点检信息。
- ~~不做"充电中"指示~~：**已做**（2026-11-03）。原先的判断是"多一次 I2C 事务、
  `is_charging()` 0.28ms 不值"，但实测这 0.28ms 落在 1.30ms/10s 的预算里可忽略，
  而插电时百分比会**短暂倒挂**（见 §10.1：插电后 9%→8%），不做充电指示反而更容易误判。
  颜色: charging 绿 / ok 白 / low 红 / unknown 灰。

