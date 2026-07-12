# 线程 / Tick / 状态机 拓扑

> 最后更新: 2026-07-09
> 分支: feat/thread-safe-sm-tick

---

## 1. 线程 & 速率总览

| 线程 | 频率 | 触发方式 | 主循环延迟 | 核心职责 |
|------|------|---------|-----------|---------|
| **Main** | **~300 Hz** | `select.select` 定时挂起 | `0.00333` (3.33ms) | 模块 loop + `coordinator.loop()` + 出队 SM 事件 |
| **Camera Process** | Camera FPS (目标 120~300) | `_process_loop` busy poll | 无帧时 1ms sleep | 采集 → 检测 → 发伺服帧 → publish FrameResult |
| **Heartbeat** | **10 Hz** | `time.sleep(0.1)` | — | 发心跳帧，检测 MCU 超时 |
| **UART Receiver** | 事件驱动 | Serial 数据回调 | — | 帧解析 → EventBus publish |

---

## 2. 主循环流向图

```
MAIN THREAD (~300 Hz, select.select 3.33ms)
┌─────────────────────────────────────────────────────────┐
│  ModuleManager.run_main_loop(coordinator)               │
│                                                         │
│  while self.running:                                    │
│    for module in _loop_methods:                         │
│      module.loop()              ← 例如 opencv display  │
│                                                         │
│    coordinator.loop()                                   │
│    ├─ with _sm_lock:                                    │
│    │   while _sm_queue:                                 │
│    │     _sm_queue.popleft()()   ← 消费事件 lambda     │
│    │                                                   │
│    └─ mission_sm.run_to_completion()                    │
│         ├─ update() → on_execute → 可能 cascade 转场   │
│         └─ 循环直到状态稳定 (max_steps=10)              │
│                                                         │
│    select.select([], [], [], 0.00333)  ← ~300Hz tick   │
└─────────────────────────────────────────────────────────┘

COORDINATOR.LOOP() 内部展开：

  loop()
  ├─ _sm_queue 出队（线程安全，一次性消费所有待处理事件）
  │   ├─ mission_sm.on_action_done(...)     ← 来自 ACTION_DONE 帧
  │   ├─ mission_sm.on_arrived(zone_id)     ← 来自 ARRIVED 帧
  │   ├─ mission_sm.start()                 ← 来自 CMD_START_QR
  │   ├─ mission_sm.on_discovery_done()     ← 来自 CMD_DISCOVERY_DONE
  │   ├─ mission_sm.set_error(...)          ← 来自 EMERGENCY_STOP
  │   ├─ mission_sm.on_qr_result(str)       ← 来自 QR 识别
  │   └─ _apply_visual_status(vis, flags)   ← 来自视觉 latch
  │
  └─ mission_sm.run_to_completion()
       └─ 触发 cascade 转场:
           PLACE_ROUGH → ALIGN_ROUGH → PICK_ROUGH (1s 延时到期)
           CHECK_LOAD  → ALIGN_RAW / NAV_TO_* (立即)
           RING_DISCOVERY → ALIGN_ROUGH / ALIGN_TEMP (discovery_done)
```

---

## 3. 事件入队 & 消费路径

```
UART RECEIVER THREAD                CAMERA PROCESS THREAD
─────────────────────────           ────────────────────────
Serial data callback                _process_loop
  └─ FrameParser.feed(data)           └─ process_all()
       └─ _handle_frame(frame)              └─ publish(FrameResult)
            │                                    │
            │ EventBus.publish                   │
            ▼                                    ▼
      ┌──────────────────────────────────────────────┐
      │            EVENT BUS (同步分发)                │
      │  订阅者被发布者线程直接调用                      │
      └──────────────────────────────────────────────┘
            │                    │
            ▼                    ▼
     _on_arrived(evt)     _on_vision_results(evt)
     _on_action_done(evt)   │
     _on_mcu_cmd(evt)       ├─ "qr_detect"
     _on_emergency(evt)     │   → _on_qr_result_event
     _on_heartbeat(evt)     │     → _enqueue_sm(lambda)
     _on_request_sync(evt)  │
     _on_qr_result_event    ├─ "track_cargo/ring_*"
       (从 camera 线程来)     │   → _handle_track_result(data)
                            │     → visual_sm.update()    ← 直接调！
                            │     → _send(伺服帧)          ← 直接发！
                            │     → 如果 latch 触发
                            │       → _enqueue_sm(lambda)
                            │
                            ▼
                      ┌────────────────┐
                      │   _sm_queue    │  ← 线程安全队列 (_sm_lock)
                      │  Deque[Fn]     │
                      └────────────────┘
                            │
                            │ coordinator.loop() 出队消费
                            ▼
                   MAIN THREAD (~300Hz)
                      mission_sm.run_to_completion()
```

---

## 4. 视觉伺服数据路径（300fps 关键路径）

```
CAMERA PROCESS THREAD  (高频，不经过主循环)
┌─────────────────────────────────────────────────────────┐
│  process_all()                                          │
│    → per camera: get_frame()                            │
│    → run_tasks_serial()                                 │
│    → VisionResult { task_name, success, result_data }   │
│                                                         │
│  publish(FrameResult)                                   │
│    → _on_vision_results()                               │
│      → _handle_track_result(data)                       │
│                                                         │
│        visual_sm.context.percent_error_x = ...  立即    │
│        visual_sm.context.percent_error_y = ...  立即    │
│        visual_sm.update()                        立即    │
│                                                         │
│        build_visual_servo_data_frame(pe_x, pe_y,        │
│          flags, visual_state)                           │
│        _send(frame)  ← 直发 UART，不走队列！           │
│                                                         │
│        if ready_latched:                                │
│          _enqueue_sm(_apply_visual_status)  ← 仅 SM    │
│            入队                                        │
└─────────────────────────────────────────────────────────┘

 伺服帧发送路径:
   Camera Thread → _send() → uart_interface.send_raw()
                              └─ with _write_lock:
                                   serial_controller.send_bytes(frame)

 延迟: ~0.1ms (不经过主循环)
```

---

## 5. 状态机双架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      BaseStateMachine                            │
│  _lock: threading.RLock()  (所有状态操作都持有此锁)               │
│  _states: Dict[str, State]                                        │
│  _event_transitions: { from_state: { event: to_state } }         │
│                                                                   │
│  核心方法:                                                         │
│    trigger(event)  → 查表 → _do_transition                        │
│    update()        → on_execute → 可能 _do_transition             │
│    run_to_completion(max_steps=10) → loop update() until stable  │
│    _do_transition() → on_exit → on_enter → callbacks              │
└─────────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
         │                              │
┌────────────────────┐    ┌──────────────────────────┐
│  MissionStateMachine│   │  VisualStateMachine       │
│  18 states, 16 evts │   │  5 states (IDLE/SEARCH/   │
│                     │   │   TRACKING/RECOVERY/FAIL)  │
│  操作线程: MAIN     │   │  操作线程: CAMERA          │
│  入队方式:           │   │                            │
│    _enqueue_sm(lambda)│  │  update() 由相机线程直接调  │
│    loop() 出队 +     │   │  start()/stop() 由 UART   │
│    run_to_completion │   │  线程 (有 RLock)            │
│                     │   │                            │
│  context:            │   │  context:                  │
│    MissionContext    │   │    VisualContext           │
│    (dataclass)      │   │    (读写均在同一线程)      │
└────────────────────┘    └──────────────────────────┘
```

### MissionStateMachine 状态流

```
WAIT_START ──START──→ READ_QR ──QR_OK──→ NAV_TO_RAW ──ARRIVED_RAW──→ ALIGN_RAW
                                                                        │
                                                                  on_execute
                                                                  ready_to_pick
                                                                        │
                                                                        ▼
                                                                   PICK_RAW
                                                                        │
                                                                  ACTION_DONE
                                                                        ▼
                                                                  CHECK_LOAD
                                                                  on_execute
                                                                  cargo_confirmed (True)
                                                                        │
                                                    ┌───────────────────┤
                                                    │                   │
                                              !batch_done          batch_done
                                                    │                   │
                                                    ▼                   ▼
                                              ALIGN_RAW          NAV_TO_ROUGH → RING_DISCOVERY → ...
                                              (loop next)        (zone transition)
```

### VisualStateMachine 状态流

```
IDLE ──START──→ SEARCH ──TARGET_FOUND(10帧)──→ TRACKING ──TARGET_LOST(5帧)──→ SEARCH
                      │                             │
                  RECOVERY_FAILED              TARGET_RECOVERED
                      │                             │
                      └────── RECOVERY ←────────────┘

  SEARCH: consecutive_detected_frames >= 10  → TRACKING (auto)
  TRACKING: consecutive_lost_frames >= 5  → SEARCH (auto)
  RECOVERY: 由外部事件驱动
```

---

## 6. 同步点 & 锁

```
锁                                    保护对象                    持有者
─────────────────────────────────────────────────────────────────────────
_sm_lock (threading.Lock)             SM 事件队列 _sm_queue    入队者（任意线程）
                                                                出队者（主线程）

mission_sm._lock (RLock)              MissionSM 内部状态       主线程（run_to_completion）
                                                                心跳线程（set_error — 已入队）
                                                                (trigger / do_transition)

visual_sm._lock (RLock)               VisualSM 内部状态       相机线程 (update)
                                                                UART 线程 (start/stop)
                                                                (trigger / do_transition)

uart._write_lock (Lock)               UART 发送串口            _send() 调用者（多线程）
                                                                主线程 / 相机线程 / 心跳线程

uart._state_lock (RLock)               UART 状态字段            UART 接收线程
                                                                外部查询者

event_bus._lock (RLock)               订阅者列表                EventBus 操作者
```

### 锁争用分析

```
_sm_lock:
  · 持有时间: deque.append / popleft（ns 级）
  · 冲突概率: 极低（一次性出队所有元素，释放锁后才跑 lambda）

mission_sm._lock:
  · 持有时间: run_to_completion 全程（可能包含 cascade 转场, ~μs-ms 级）
  · 冲突概率: 极低（只有主线程获取）

visual_sm._lock:
  · 相机线程 update 时持有，UART 线程 start/stop 时持有
  · 冲突概率: 低（update 频率高但时间短；start/stop 极少触发）

uart._write_lock:
  · 相机线程 120~300Hz 发伺服帧，持续持有 lock
  · 心跳线程 10Hz 发心跳，短暂竞争
  · 冲突概率: 中（伺服帧频繁，但持锁时间 ~μs）
```

---

## 7. 计时 & 延迟

| 路径 | 延迟 | 瓶颈 |
|------|------|------|
| **UART 帧 → SM 处理** | **~1.7ms**（平均半 tick） | 主循环 tick 3.33ms |
| **视觉帧 → 伺服帧发送** | **~0.1ms**（直发） | 无（不经过主循环） |
| **视觉 latch → SM 转场** | **~1.7ms + 0ms** | 入队等 tick + 出队后立即转场 |
| **PLACE_ROUGH 1s 定时** | **±1.7ms**（半 tick） | 主循环 tick 3.33ms |
| **心跳超时检测** | **~0.3s + ~1.7ms** | heartbeat_interval + tick |
| **MCU 无响应 → is_linked=false** | **~0.3s** | 超时检测（不触发 SM 转场） |

### 各循环精度

| 循环 | 基准速率 | 精度机制 | 实际精度 |
|------|---------|---------|---------|
| Main loop | ~300 Hz (3.33ms) | `select.select` → `ppoll/nanosleep` | ~1ms（依赖内核 HZ） |
| Camera process | Camera FPS | 总线空转 + 1ms idle sleep | ~帧间隔 |
| Heartbeat | 10 Hz | `time.sleep(0.1)` | ~±10ms（受 GIL 影响） |
| Visual SM update | 每帧 | 相机线程直接调 | 帧间隔 |
| Mission SM run_to_completion | 每 tick | 主线程 coordinator.loop | 3.33ms tick 边界 |

---

## 8. UART TX 多路发送

```
_send(frame) 可被多线程调用，线程安全：

  ┌─ Main Thread ──────────────────────────┐
  │  _send_initial_status()                 │
  │  _deactivate_all_visual() → status 帧   │
  │  _apply_qr_result() → QR + status 帧    │
  └─────────────────────────────────────────┘

  ┌─ Camera Thread ─────────────────────────┐
  │  _handle_track_result() → 伺服帧 (高频) │
  │  _handle_discovery_result() → 伺服帧    │
  └─────────────────────────────────────────┘

  ┌─ Heartbeat Thread ──────────────────────┐
  │  _heartbeat_loop() → 心跳帧 (10Hz)      │
  └─────────────────────────────────────────┘

  所有 _send 内部：
    with uart._write_lock:
        serial.send_bytes(frame)
```

---

## 9. 数据流所有权

```
数据                       写入者              读取者                 保护
──────────────────────────────────────────────────────────────────────────
_sm_queue                  任意线程           主线程 loop()          _sm_lock
MissionContext             主线程 (仅)         主线程 / 心跳线程     mission_sm._lock
VisualContext              相机线程            相机线程 / 主线程     无锁（单写者模式）
_ready_frames/latched      相机线程            相机线程              无锁（单线程）
uart._current_zone         UART 接收线程       任意线程              uart._state_lock
_last_mcu_heartbeat        UART 接收线程       心跳线程              无锁（单调时间戳，良性竞争）
_heartbeat_seq             心跳线程 (仅)        心跳线程              无锁（单写者）
```

---

## 10. 关键观察

1. **伺服帧是唯一真正高频的路由**（相机线程直发，不受主循环限制）
2. **MissionSM 是单线程模型**（所有操作通过 `_enqueue_sm` 串行化到主线程）
3. **VisualSM 是例外**（相机线程直接 `update()`，因为需要逐帧跟踪判定）
4. **锁争用风险点**: `uart._write_lock` 在相机线程高频发帧时被持续持有，心跳线程可能短暂等待
5. **300Hz 主循环精度依赖内核**: 如果内核 `HZ < 1000`，`select.select` 精度会退化。当前平台约 300Hz，所以 `MAIN_LOOP_DELAY = 0.00333`。升级内核后可降到 `0.001`。
