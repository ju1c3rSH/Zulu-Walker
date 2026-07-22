# Architecture Overview

Zulu-Walker uses a **three-layer architecture** that separates reusable infrastructure from application logic, making it easy to adapt for different robots and use cases.

## Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│                    app/  (your code)                      │
│  Application logic, mission coordination, custom states  │
│  Write your own coordinator, main.py, state machines     │
├─────────────────────────────────────────────────────────┤
│                 modules/  (optional, reusable)            │
│  Camera pipeline (zw_opencv_module)                       │
│  UART communication (zw_uart_module)                      │
│  Third-party or custom modules                            │
├─────────────────────────────────────────────────────────┤
│                framework/  (reusable core)                  │
│  EventBus · StateMachine · ModuleManager · HAL            │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

```
                  project_config.yaml
                         │
              Machine.create(config)
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      CameraHub      UART         AIInference
      Display
            │
            ▼
    ModuleManager.load("module_name")
            │
     ┌──────┴──────┐
     ▼             ▼
  init() → start() → loop() → stop()

    Main Loop (~300 Hz):
    ┌──────────────────────────────┐
    │  for module in modules:      │
    │    module.loop()             │
    │  coordinator.loop()          │
    │    → drain events            │
    │    → update state machine    │
    │  display.show(frame)         │
    │  time.sleep(0.00333)         │
    └──────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **HAL interfaces as Protocols** | Structural subtyping — mock objects pass `isinstance` checks without inheritance, easy to test |
| **Platform singleton (CameraHub)** | Single `open`/`get` point for all cameras, platform-agnostic |
| **Display owned by main loop** | Not by VisionManager — `show()` returns `bool` to signal exit |
| **ModuleManager lifecycle** | All modules follow `init(machine, event_bus)` → `start()` → `loop()` → `stop()` |
| **State machine in main thread** | No multi-threaded state access issues; events are queued and consumed synchronously |
| **Mock platform default** | Framework works out of the box with `pip install numpy pyyaml && python run.py` |
| **Zero hard dependencies** | Core needs only `numpy` + `pyyaml`. cv2, serial, maix are optional platform extras |

## When to Use Each Layer

### framework/ — always included
This is your foundation. Import from `framework.event_bus`, `framework.state_machine`, `framework.hal`, `framework.module_manager` directly.

### modules/ — optional, mix and match
Pre-built modules that provide common robot capabilities. Include only what you need. Each module is discoverable by `ModuleManager` by its package name under `modules/`.

### app/ — you write this
Your application code. Reference examples in the [Building Applications](building_apps.md) guide.
