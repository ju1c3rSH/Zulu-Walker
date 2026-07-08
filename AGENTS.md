<!-- DO NOT TRACK THIS FILE IN GIT (add to .gitignore or use --assume-unchanged) -->

# AGENTS.md

Zulu-Walker — modular Python app for Orange Pi 5B: camera vision + STM32 UART + RTMP streaming.

## Entry Points
- `python run.py main` — production (loads modules from `AUTO_START_MODULES`)
- `python run.py debug [-c CAM_IDX] [-W WIDTH] [-H HEIGHT] [--debug-uv] [--debug-cam]` — parameter tuning GUI

## Module System
Every module in `modules/` **must** implement: `init()` → `start()` → `loop()` → `stop()`.
`loop()` is called on the main thread at ~100 Hz. Keep it minimal.

## Critical Gotchas

### OpenCV GUI on Main Thread (Linux/X11)
`cv2.imshow()`, `cv2.waitKey()`, `cv2.namedWindow()` **MUST run on the main thread** on Linux.
The `loop()` method in `zw_opencv_module/__init__.py` calls `display_frame()` — this is intentional.

### Enum Import Consistency
Different import paths create independent class definitions — `isinstance` and `==` will fail.
Always import `DetectMethod` from `modules.zw_opencv_module.detectors.circle_target_detector`.

### ParamPanel Scale Bug
`get_raw_params()` saves raw slider values. `load_params()` loads them directly.
**Never divide by scale** when loading. When adding new params, delete `debug_params.yaml` to regenerate.

### Missing Dependencies
`requirements.txt` lists only `pyserial`. The app also **requires** `opencv-python`, `numpy`, and `pyyaml`.

## Platform: Orange Pi 5B (RK3588)
- Camera capture thread pinned to **little cores [0,1,2,3]** (A55)
- Processing thread pinned to **big cores [4,5,6,7]** (A76)
- Camera: V4L2 + MJPG codec, 120 fps target, 2-frame queue, non-blocking reads
- UART: defaults to `/dev/ttyS4` @ 921600 baud

## Config Files (auto-generated, delete to recreate defaults)
- `modules/zw_opencv_module/config/debug_params.yaml` — detection method + per-method params
- `modules/zw_opencv_module/config/uv_params.yaml` — UV spot color ranges
- `modules/zw_opencv_module/config/camera_params.yaml` — camera hardware params (exposure, WB, etc.)

## Detection Methods
`edge_drawing_quads` (default) > `contour_ellipse` > `edge_contour_ellipse` > `test_line_quad`.
Default params in `param_utils.py:get_default_params()`.

## Architecture

```
context/                        # 项目级中枢层：事件总线 + 状态机 + 协调器
    event_bus.py                # EventBus — typed pub/sub, thread-safe
    events.py                   # 事件 dataclass 定义
    mission_state_machine.py    # MissionStateMachine (18 states, mirrors STM32)
    mission_context.py          # MissionCoordinator — wires UART ↔ Vision ↔ State
    visual_state_machine.py     # VisualStateMachine — 5-state tracking SM

utils/
    state_machine/
        base.py                 # BaseStateMachine (generic framework)
        bridge.py               # StateActionBridge — 状态 enter/exit 触发动作

modules/
    zw_opencv_module/           # 相机采集 + 视觉检测
    zw_uart_module/             # UART 帧收发 + 协议定义
```

## Competition State Machine & Protocol (工创赛物流小车)

### Mission State Machine (`context/mission_state_machine.py`)
- 18 states, IDs 0–17: `WAIT_START(1) → READ_QR(2) → NAV_TO_RAW(3) → ALIGN_RAW(4) → PICK_RAW(5) → CHECK_LOAD(6) → NAV_TO_ROUGH(7) → ALIGN_ROUGH(8) → PLACE_ROUGH(9) → PICK_ROUGH(10) → NAV_TO_TEMP(11) → ALIGN_TEMP(12) → PLACE_TEMP(13) → NAV_TO_RAW_SECOND(14) → RETURN_HOME(15) → FINISHED(16) → ERROR(17)`
- Hybrid event model: MCU events (ARRIVED_*, ACTION_DONE) use `trigger()`; internal decisions (ready_to_pick, cargo_count) use `on_execute()` via `update()`.
- **3-loop batch**: RAW (pick 3) → ROUGH (place 3) → ROUGH (pick 3) → TEMP (place 3). Batch 2 reuses same states via `NAV_TO_RAW_SECOND`.
- `advance_target()` returns `bool` (True = batch cycle done, zone transition needed).
- `is_batch_complete()` for pre-transition assertions (`cargo_count==0 and step==0`).

### Visual Task Auto-Switch
- Managed by `StateActionBridge` in `mission_context.py:_wire_state_actions()`:
  - `READ_QR` → `qr_detect` on `cam_qr`
  - `ALIGN_RAW` → `track_cargo` on `cam_cargo`
  - `RING_DISCOVERY` → `ring_discovery` on `cam_cargo` (activated by MCU CMD, not auto)
  - `ALIGN_ROUGH`/`ALIGN_TEMP` → no visual (camera blocked by cargo, MCU uses mapping + inertial nav)
  - NAV/WAIT/FINISHED/ERROR → all visual tasks disabled
- Camera routing: `cam_id.endswith("_qr")` for QR, `cam_id.endswith("_cargo")` for tracking.

### CargoSet Runtime Tracking
- `CargoSet` is created once in `MissionCoordinator.__init__` and stored on `MissionContext`.
- `cargo_pick_stack` (FIFO) records item indices in pick order; place handlers `popleft()` from it.
- `_CheckLoadState.on_execute` calls `item.pick()` on matching CargoItem. PLACE_ROUGH/TEMP handlers call `item.place()`.
- No CargoSet dependency on state machine decisions — purely for debug tracking.

### UART Protocol (`modules/zw_uart_module/protocol.py`)
- Sub-commands: `CMD_START_QR(0x01)`, `CMD_STOP_VISUAL(0x06)` only.
- `ActionId` IntEnum for `TYPE_ACTION_DONE`: PICK_RAW=1, PLACE_ROUGH=2, PLACE_TEMP=3, PICK_ROUGH=4.
- Frame types `0x10-0x18` for mission sync; legacy zone types `0x01-0x04` still defined.
- Docs: `docs/competition/protocol.md` (MCU team spec), `docs/architecture/state_machine.md` (design doc).

### Misc
- `Color` (IntEnum) from `modules.zw_opencv_module.models.color` — use everywhere, not raw ints.
- `ColorTrackable` Protocol (`modules.zw_opencv_module.processors.base`) with `@runtime_checkable`.
- `context.events` re-exported from `context.__init__`.

## Testing
No test infrastructure, formatters, linters, or CI exist. `_archived_tests/` has been deleted.
