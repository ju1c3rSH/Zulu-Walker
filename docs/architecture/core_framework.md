# Core Framework

## EventBus

Thread-safe publish/subscribe bus with typed events and event history.

```python
from framework.event_bus import EventBus

bus = EventBus(history_size=1000)

# Subscribe
bus.subscribe(MyEventType, my_handler)
bus.unsubscribe(MyEventType, my_handler)

# Publish
bus.publish(MyEventType(...))

# Inspect history
recent = bus.history
bus.clear_history()
```

- Subscribers are called **synchronously** on the publisher's thread
- Subscriber exceptions are caught and logged (one bad subscriber won't crash the bus)

## State Machine

### BaseStateMachine

Generic, extensible state machine. Define states by subclassing `State` with three hooks:

```python
from framework.state_machine import BaseStateMachine, State

class IdleState(State):
    def on_enter(self, context, from_state):
        pass                                  # setup when entering
    def on_execute(self, context):
        return "RUNNING" if ready else None   # return state name to auto-transition
    def on_exit(self, context, to_state):
        pass                                  # cleanup when leaving
```

Wire states and transitions:

```python
sm = BaseStateMachine()
sm.register_state("IDLE", IdleState())
sm.register_state("RUNNING", RunningState())
sm.set_initial_state("IDLE")

sm.register_transition("IDLE", "RUNNING", event="START")
sm.register_transition("IDLE", "RUNNING", condition=lambda ctx: ctx.ready)
```

Drive the machine:

```python
sm.trigger("START")          # event-driven transition
sm.update()                  # execute current state
sm.run_to_completion()       # keep updating until stable (cascade)
```

Additional capabilities:

- **Callbacks**: `add_state_change_callback(fn)`, `on_state_enter(state, fn)`, `on_state_exit(state, fn)`
- **Context**: Any data object you attach via `sm.context = my_ctx`
- **Thread safety**: All state operations hold an `RLock`
- **Introspection**: `current_state`, `previous_state`, `state_duration`, `state_frame_count`

### StateActionBridge

Declarative bindings — run actions when entering/exiting states without modifying state classes:

```python
from framework.state_machine import StateActionBridge

bridge = StateActionBridge(sm)
bridge.when_enter("RUNNING", start_motors)
bridge.when_exit("RUNNING", stop_motors)
bridge.when_enter({"IDLE", "ERROR"}, deactivate_all)
```

### VisualStateMachine

A concrete 5-state tracking FSM built on `BaseStateMachine`, ready to use:

```
IDLE → SEARCH → TRACKING → RECOVERY → FAIL
```

- **IDLE**: Standby, no vision task
- **SEARCH**: Looking for target (→ TRACKING after 10 consecutive detections)
- **TRACKING**: Following target, outputting servo deviations (→ SEARCH after 5 lost frames)
- **RECOVERY**: Expanded search after tracking loss (external trigger)
- **FAIL**: Terminal error (→ IDLE on RESET)

Use via `from framework.visual_state_machine import VisualStateMachine`.

```python
vsm = VisualStateMachine()
vsm.context = VisualContext()

vsm.start()        # IDLE → SEARCH
vsm.stop()         # → IDLE
vsm.update()       # per-frame update; auto-transitions SEARCH↔TRACKING
```

## ModuleManager

Discovers, loads, and manages module lifecycles. Modules are Python packages under `modules/<name>/` with optional lifecycle hooks.

```python
from framework.hal import Machine
from framework.module_manager import ModuleManager

machine = Machine.create("project_config.yaml")
manager = ModuleManager(machine, event_bus=bus)

manager.load("zw_opencv_module")    # loads modules.zw_opencv_module
manager.load("zw_uart_module")      # loads modules.zw_uart_module

# Or bulk register:
manager.register_many(["zw_opencv_module", "zw_uart_module"])
```

### Module Lifecycle

| Hook | When | Signature |
|------|------|-----------|
| `init(machine, event_bus)` | On load | `init(machine: Machine, event_bus=None) -> None` |
| `start()` | After init | `start() -> None` |
| `loop()` | Every main loop tick (~300 Hz) | `loop() -> None` |
| `stop()` | On shutdown | `stop() -> None` |

All hooks are optional — `ModuleManager` checks for their existence with `hasattr`.

### Main Loop

```python
manager.run_main_loop(coordinator=None, tick_callback=None, display_callback=None)
```

- Calls every module's `loop()` in sequence
- If a `coordinator` is provided, calls `coordinator.loop()` (event drain + SM update)
- If `display_callback` is provided, calls it (typically `display.show(frame)`)
- Sleeps `MAIN_LOOP_DELAY` (3.33 ms, ~300 Hz)
- Handles `KeyboardInterrupt` and per-module exceptions
- Calls `stop_all()` on exit

## Logging

```python
from framework.log import fw_log
fw_log("State transition:", old, "→", new)
```

Thin wrapper around `logging.getLogger("framework")`.
