# 日志管道：一条 `log_print` 路径

> 2026-11-03 · 与 `docs/architecture/battery_hal_hud.md` §7.5 配套
> 适用分支 `feat/explosive-dock-trim`（`platform: maixcam1`）
> 本文描述**现状**（读码 + 实测），不是设想。标注"缺口"的部分是刻意保留的，
> 每条都写了"为什么现在不改"。

## 0. 一句话

运行期日志只有一个入口：`log_print(msg)`。调用方**只入队**，写盘与上屏由单独的
`log-writer` 线程完成。`logs/debug.log` 是存档，`sys.__stdout__` 是控制台 ——
同一行同时去这两个地方，不存在"只进文件"或"只上屏"的分支。

## 1. 全景

```
 任意线程 (主循环 / 视觉 / 显示 / UART)                  log-writer 线程                出口
 ─────────────────────────────────────                  ─────────────────              ────
 log_print(msg)
   ├─ _is_perf_log(msg)? ──命中──► return（连队列都不碰）
   ├─ datetime.now().strftime("[%H:%M:%S] ")
   └─ _LOG_QUEUE.put_nowait((line, msg)) ──┐
        └─ 满 → _DROP_COUNT += 1，不阻塞、不抛
                                            ▼
                                    queue(maxsize=2048)
                                            ▼
                                    _writer_loop()
                                      ├─ open("logs/debug.log","a") → write → flush → close
                                      └─ 控制台:
                                           调试台未导入      → sys.__stdout__.write
                                           调试台导入但未跑  → sys.__stdout__.write
                                           调试台在跑        → dc.log(msg)（TUI 面板）
                                      └─ 队列空且丢过包 → 打一条 [WARN] dropped N
```

## 2. 三个阶段

### 2.1 调用方（唯一有成本的地方）

| 步骤 | 成本 |
|---|---|
| `datetime.now().strftime()` | ~0.5µs |
| `*args` 拼接 | ~0.3µs |
| `_is_perf_log()` 判断 | ~0.1µs（名单为空时直接返回 False） |
| `put_nowait` | ~0.2µs |
| **合计（实测，PC）** | **9.1µs / 行** |

**没有 I/O，没有锁**。这是整条管道唯一的设计承诺：任何实时线程都可以放心调用。

### 2.2 闸门（`_is_perf_log`）

两类判据，目的都是"把反复刷的观测行降下来/吞掉，事件类日志留住"：

1. `[PERF] ` 前缀：受 `observability` 三键控制 —— `perf`（总开关）、`perf_detail`
   （细粒度探针）、`perf_interval_s`（汇总窗口，默认 10s；1 = 旧行为每秒一条）。
   `perf_detail: false` 只放 vision / mainloop / lcd show 三条常驻行；
   调试页的 `[PERF] page/tuner/laser/rescue` 属 detail 类，且窗口固定 1s
   （只在页面打开时发，是交互调参反馈）。
2. 名单子串匹配：`perf_log.names`，默认组 7 条 ——
   `[PERF] vision/mainloop/lcd show`、`[UART TX] DATA_STREAM`、`[UART TX] CMD_ACK`、
   `[WDT] `、`[MEM] `。`enabled: false` 时**只**吞 names 里列出的行（当前默认只列
   `[WDT] `，MEM 与 DATA_STREAM 保留）。

**被吞的行既不进控制台也不进 `debug.log`**（在入队前 return）。所以"关掉 perf 日志"
等于"这条历史不存在"，不是"只是不显示"。启动时会打一行
`[PERFLOG] perf_log=custom([WDT] ) observability.perf=ON perf_detail=OFF interval=10.0s`，
以这行为准（`perf_log=` 是三态：`ON` / `custom(名单)` / `MUTED`；`interval=` 是生效值）。

### 2.3 writer 线程

- **每行 `open→write→flush→close`**：不持有长句柄，所以日志文件可以被随时
  `rm` / 拷走而不影响运行；代价是每行一次 open（实测 ~39µs），但这在独立线程里。
- 丢包只在队列**排空后**补一条汇总，避免洪泛时反而更慢。
- 调试台在跑时改投 `dc.log()`，并从 `sys.__stdout__` 撤回 —— 否则 TUI 全屏会被打花。

## 3. 消费者一览

| 消费者 | 来源 | 开关 | 备注 |
|---|---|---|---|
| 控制台 stdout | `sys.__stdout__` | 恒开 | 设备 SSH / IDE 终端看到的就是这个 |
| 存档 `logs/debug.log` | 同上一条路径 | 恒开 | **无轮转**，见 §6.1 |
| `DebugConsole`（rich TUI） | `dc.log()` | `debug_console_enabled`（默认 false） | 关时连 `rich` 都不导入（CAM1 只有 128MB 用户内存） |
| `ConsoleCapture` | 把 `print()` 接进 `log_print` | 随 TUI 开关 | 关时 `sys.stdout` 保持原样 |

## 4. 分层规则：谁该在哪里打日志

| 层 | 该做的 | 不该做的 |
|---|---|---|
| HAL（`platforms/*`） | 返回事实：值 + `reason` | 不打日志。"为什么读不到"放进返回值 |
| 采样器（`battery_sampler`） | 状态归约：去抖、`level` | 不决定"要不要说"；只对无快照的即时事件（抛异常、跳变被拒）就地打一行 |
| 模块（`modules/*`） | 节拍 + 格式 + 节流，**唯一出口** | 不碰硬件、不解释像素 |
| HUD（`display/hud.py`） | 只画 | 不打日志（视觉线程，10Hz，日志会把它拖慢） |
| 协调器 / 协议 | 状态变化事件 | 不按帧打（除 `[PERF]`/`[FPS]` 这类闸门内的观测行） |

**反例（2026-11-03 踩过）**：`percent=-1` 时 HAL 打一句"pmu 不可用"、模块再打一句
"无读数"，同一个事实两句话，而控制台只可能看到其中一句（`logger.warning` 只进
`logs/app.log`）。正确形状是让 `reason` 跟着读数/快照走，出口只打一行。

## 5. 落地案例：电量（唯一按"每拍"打的模块）

```
[10s] zw_battery_module._poll_if_due
        └─ snap = sampler.poll() ──► BatterySnapshot(percent, mv, level, reason, ts)
              ├─ if snap is not None: log_print(...)   → 控制台 + debug.log
              └─ HUD 每帧读同一个快照（10Hz，无 I/O）     → LCD
```

- **判据是"有没有新快照"**，不是"值变没变"：`poll()` 返回新发布的快照或 `None`
  （读异常、跳变被拒），模块只有一句 `if snap is not None`，不存任何日志状态。
- **每 10s 一行**：8640 行/天 ≈ 730 KB/天。存档要能回答"那段时间采没采"，只打变化时
  文件里是断的。`percent=-1` 也照打 —— 没有第二套策略，要安静就 `battery.enabled: false`。
- 三个消费者（控制台 / 存档 / HUD）都从同一个 `frozen` 快照读，不存在"屏幕上是 A、
  日志里是 B"。

## 6. 已知缺口（明确"暂不修"）

### 6.1 `debug.log` 无轮转、无上限

`_writer_loop` 只 append，不检查大小。按当前发射点实测推算：

| 场景 | 行/秒 | 体积 |
|---|---|---|
| 2026-11 之前（PERF/detail/DATA_STREAM/WDT 全开） | 4.8 | 1.8 MB/h · **42 MB/天** |
| 当前默认（PERF 3 行/10s、detail 关、只静音 WDT） | ~1.7 | 0.6 MB/h · **15 MB/天** |

> 旧行按 2026-11-03 的发射点实测；新行沿用同一 104 B/行口径重算，
> 主残项是 `[UART TX] DATA_STREAM`（1 条/秒，推流率验收要它跟 `[FPS]` 对拍）。

发射点：`[PERF] vision/mainloop/lcd show` + `[UART TX] DATA_STREAM` 各 1/s，
`[FPS]` 1/5s，`[WDT]`×3 与 `[MEM]` 6.7~11s 一条，`[BAT]` 1/10s。

**暂不修的理由**：赛前动启动路径不划算。真要动手，最小改法是
`start_log_writer()` 里若 `debug.log > 8MB` 就 `os.replace` 成 `debug.log.1`（留一代），
6 行代码，不碰任何调用点。

### 6.2 没有级别，没有模块字段

只有 `[HH:MM:SS]` 前缀 + 调用方自觉写的 `[TAG]`。日期、级别、logger 名都没有。
后果：跨天无法定位；无法按级别过滤；`grep` 是唯一手段。

**暂不修的理由**：210 个调用点，改签名（`log_print(msg, level=..., module=...)`）本身
向后兼容，但"谁该用哪个级别"要逐个过一遍，属于赛后重构。

### 6.3 两套并行体系

| 体系 | 落点 | 现状 |
|---|---|---|
| `log_print` | `logs/debug.log` + 控制台 | 主力，210 处 |
| `logging`（`logger.*`） | `logs/app.log`，root WARNING，**无 console handler** | 153 处，设备上看不见 |
| `LoggerFactory` / `ConfigurableLogger` | `logs/modules/*.log` | 只剩 `performance.py` 在用，其余是死代码 |

`app/main.py` 的 `logging.basicConfig(filename="logs/app.log")` 是关键事实：**logger 打的
字在设备控制台完全不出现**。所以"某个状态看不见"的第一反应不该是"加个 log_print"，
而是先确认它现在走的是哪套。

### 6.4 为什么不在主循环里用 `logging.FileHandler`

实测：`FileHandler.emit()` = **调用方线程同步 `write+flush`，39.1µs/行**；`log_print`
只入队 9.1µs，写盘在别的线程。主循环里已经有一次可能阻塞的 I2C 读（电量 1.3ms），
再开第二条同步 I/O 路径，卡死面翻倍。**任何"单独落一个文件"的想法都要先过这一关。**

### 6.5 `ConsoleCapture._file` 是死句柄

`utils/console_capture.py:15` 打开了 `logs/debug.log` 却从不写（真正的写入走
`log_print`）。留着无害，但会让人误以为有第二条写入路径。

## 7. 改动这条管道前的三条硬约束

1. **不在热路径打日志**。视觉每帧、UART 每字节都属于热路径；要观测就用 `[PERF]` 汇总。
2. **不在持锁区打日志**。`PresenterService.submit()` 持 `presenter._lock`，里面任何
   可能阻塞的调用都会连带卡住主循环。
3. **不为单个功能开第二条 I/O 路径**。要存档就写进 `debug.log`；要结构化就改
   `log_print` 的签名，别在旁边另起炉灶。

## 8. 排查清单

| 现象 | 先查什么 |
|---|---|
| 控制台没有某行 | 启动行 `[PERFLOG] ...` 是 ON 还是 MUTED；`_is_perf_log` 是否命中 |
| 文件里有、控制台没有 | 一般不可能（同一路径）；TUI 开着时控制台= TUI 面板 |
| 控制台有、文件里没有 | 同上；除非 `logs/` 不可写（writer 静默吞异常） |
| 出现过 `[WARN] log_print queue full, dropped N` | 队列 2048 打满 = 有人在高频打日志，去抓发射点而不是调大队列 |
| 电量读数与屏幕不一致 | 不应发生：HUD 与日志读同一个 `BatterySnapshot`；先看 `read=` 是否异常 |
