# Threading Model

## Overview

| Thread | Typical Rate | Purpose |
|--------|-------------|---------|
| **Main** | ~300 Hz (configurable) | Module loops, coordinator, state machine, display |
| **Camera Process** | Camera FPS | Frame capture + detection pipeline |
| **UART Receiver** | Event-driven | Serial data reception → EventBus |
| **Heartbeat** (optional) | Custom (e.g. 10 Hz) | Keepalive/watchdog |

## Main Loop

The main loop runs in the main thread and follows a strict tick cycle:

```python
while running:
    for module in modules:
        module.loop()                # module-level per-tick logic
    if coordinator:
        coordinator.loop()            # event drain + state machine update
    if display_callback:
        display_callback()            # display.show(frame) → bool
    time.sleep(MAIN_LOOP_DELAY)       # default 0.00333 s (~300 Hz)
```

- All module `loop()` calls are serial — one blocking module stalls all others
- The delay is a simple `time.sleep`, subject to GIL and OS scheduling (±1-5 ms typical)
- Rate is configurable via `ModuleManager.MAIN_LOOP_DELAY`

## Thread Safety

| Resource | Protection | Accessors |
|----------|-----------|-----------|
| State machine internal state | `RLock` on each `BaseStateMachine` | Main thread (via `update()`) |
| Event queue | `Lock` (`_sm_lock`) | Any thread (enqueue), main thread (dequeue) |
| UART send | `Lock` (`_write_lock`) | Any thread |
| UART state | `RLock` (`_state_lock`) | UART thread + external readers |
| EventBus subscriber list | `RLock` | Any thread |

### Best Practices

- **Don't call state machine methods from camera threads** — use EventBus to queue events, let main thread consume them
- **UART receiver thread should publish events, not call state machine directly**
- **Vision results** should flow through a thread-safe deque or Queue, not shared mutable state

## Cross-Thread Data Flow

```
UART Receiver Thread         Camera Process Thread
        │                            │
  EventBus.publish()          deque.append(results)
        │                            │
        ▼                            ▼
  Main Thread — coordinator.loop()
        │
  queue consumer → SM events
  drain_results → vision results → SM update → servo output
```

The state machine and display update are always on the main thread, avoiding complex multi-threaded state access.

## State Machine Cascade

`run_to_completion(max_steps=10)` calls `update()` repeatedly until the state stops changing. This handles cascading transitions (e.g., arriving at a zone auto-triggers discovery → align → place in one tick).

```python
def run_to_completion(self, max_steps=10):
    steps = 0
    for _ in range(max_steps):
        prev = self.current_state
        self.update()
        if self.current_state == prev:
            break
        steps += 1
    return steps
```

The bound `max_steps` prevents infinite loops from buggy state definitions.
