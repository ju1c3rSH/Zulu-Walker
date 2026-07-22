# Building Applications

This guide walks through creating a new application using the Zulu-Walker framework.

## Workflow

1. **Choose your platform** → set `platform:` in `project_config.yaml` (mock, linux, maixcam2)
2. **Define your state machine** → subclass `State`, wire transitions in a `BaseStateMachine`
3. **Write modules** → optional reusable capabilities under `modules/`
4. **Write your app** → coordinator that bridges events, state machine, and modules
5. **Configure entry point** → `run.py` ties everything together

## Example: Line-Following Robot

### 1. Define States

```python
# app/line_follower/states.py
from framework.state_machine import State

class IdleState(State):
    def on_enter(self, ctx, from_state):
        print("Waiting for start button")

    def on_execute(self, ctx):
        return None  # stay idle until START event

    def on_exit(self, ctx, to_state):
        pass

class FollowingState(State):
    def on_enter(self, ctx, from_state):
        print("Following line")

    def on_execute(self, ctx):
        if ctx.lost_counter >= 30:      # line lost for 30 frames
            return "SEARCHING"
        return None

    def on_exit(self, ctx, to_state):
        pass

class SearchingState(State):
    def on_enter(self, ctx, from_state):
        print("Searching for line")

    def on_execute(self, ctx):
        if ctx.line_detected:
            return "FOLLOWING"
        if ctx.search_timeout:
            return "ERROR"
        return None

    def on_exit(self, ctx, to_state):
        pass
```

### 2. Wire the Machine

```python
# app/line_follower/machine.py
from framework.state_machine import BaseStateMachine
from .states import IdleState, FollowingState, SearchingState

class LineFollowerSM(BaseStateMachine):
    class Events:
        START = "START"
        STOP = "STOP"

    def __init__(self):
        super().__init__()
        self.register_state("IDLE", IdleState())
        self.register_state("FOLLOWING", FollowingState())
        self.register_state("SEARCHING", SearchingState())
        self.set_initial_state("IDLE")

        self.register_transition("IDLE", "FOLLOWING", event=self.Events.START)
        self.register_transition("FOLLOWING", "SEARCHING", event="LINE_LOST")
        self.register_transition("SEARCHING", "FOLLOWING", event="LINE_FOUND")
        self.register_transition("SEARCHING", "ERROR", event="TIMEOUT")
        self.register_transition("*", "IDLE", event=self.Events.STOP)
```

### 3. Write a Coordinator

```python
# app/line_follower/coordinator.py
from framework.event_bus import EventBus
from .machine import LineFollowerSM

@dataclass
class LineFollowerContext:
    line_detected: bool = False
    lost_counter: int = 0
    search_timeout: bool = False

class LineFollowerCoordinator:
    def __init__(self, machine, event_bus: EventBus):
        self.machine = machine
        self.sm = LineFollowerSM()
        self.sm.context = LineFollowerContext()

        event_bus.subscribe(StartEvent, lambda e: self.sm.trigger("START"))
        event_bus.subscribe(StopEvent, lambda e: self.sm.trigger("STOP"))

    def loop(self):
        # Read camera, update context
        cam = self.machine.camera_hub.get("front")
        if cam:
            frame = cam.read()
            if frame is not None:
                line = detect_line(frame)  # your detection logic
                self.sm.context.line_detected = line is not None
                self.sm.context.lost_counter = 0 if line else self.sm.context.lost_counter + 1

        # Update state machine
        self.sm.run_to_completion()

        # Act based on state
        if self.sm.is_in_state("FOLLOWING"):
            steer_motors(compute_steering(frame))  # your control logic
        elif self.sm.is_in_state("SEARCHING"):
            spin_in_place()
        else:
            stop_motors()
```

### 4. Wire in run.py

```python
# run.py
def main():
    from framework.event_bus import EventBus
    from framework.hal import Machine
    from framework.module_manager import ModuleManager
    from app.line_follower.coordinator import LineFollowerCoordinator

    bus = EventBus()
    machine = Machine.create("project_config.yaml")
    manager = ModuleManager(machine, event_bus=bus)

    coordinator = LineFollowerCoordinator(machine, bus)

    manager.run_main_loop(coordinator=coordinator)
```

## Example: RTMP Surveillance Camera

Use the existing `zw_opencv_module` for camera pipeline + streaming:

```yaml
# project_config.yaml
platform: linux
cameras:
  - source: "/dev/video0"
    width: 1920
    height: 1080
    fps: 30
```

```python
# run.py
from framework.event_bus import EventBus
from framework.hal import Machine
from framework.module_manager import ModuleManager

bus = EventBus()
machine = Machine.create("project_config.yaml")
manager = ModuleManager(machine, event_bus=bus)
manager.load("zw_opencv_module")

def display():
    vm = __import__("modules.zw_opencv_module", fromlist=["get_vision_manager"]).get_vision_manager()
    if vm and vm.frame_composer:
        frame = vm.frame_composer.compose()
        if frame is not None:
            return machine.display.show(frame)
    return True

manager.run_main_loop(display_callback=display)
```

## Architecture Patterns

### State → Action Binding

Use `StateActionBridge` to separate state logic from action execution:

```python
from framework.state_machine import StateActionBridge

bridge = StateActionBridge(sm)
bridge.when_enter("FOLLOWING", lambda: camera_module.enable_task("line_detect"))
bridge.when_enter("IDLE", lambda: camera_module.disable_task("line_detect"))
```

### Event → SM Communication

Send events from hardware callbacks to the state machine via EventBus:

```python
def on_button_press(data):
    bus.publish(StartEvent())

# In coordinator.loop():
#   sm.trigger("START") — triggered via EventBus subscription
```

### Module → Module Communication

Use EventBus for cross-module communication:

```python
# uart module publishes SensorEvent
# coordinator subscribes and updates state machine context
bus.subscribe(SensorEvent, self._on_sensor_event)
```

## Framework Capabilities Summary

| Capability | How | Provided by |
|-----------|-----|-------------|
| State machine | `BaseStateMachine` + `State` subclasses | `framework.state_machine` |
| Visual tracking FSM | `VisualStateMachine` (ready-to-use) | `framework.visual_state_machine` |
| Event bus | `EventBus` publish/subscribe | `framework.event_bus` |
| Module lifecycle | `ModuleManager.load/start/loop/stop` | `framework.module_manager` |
| Camera capture | Multi-camera via `CameraHub` | `framework.hal` |
| Display | `Display.show()` returning `bool` | `framework.hal.interface` |
| UART | Serial send/receive/receiver thread | `framework.hal.interface` |
| AI inference | Multi-model NPU inference | `framework.hal.interface` |
| Platform swap | Single YAML line changes backend | `framework.hal.platforms` |
| Pipeline camera | Task-based vision processing pipeline | `modules.zw_opencv_module` |
| Serial protocol skeleton | Generic UART interface with EventBus | `modules.zw_uart_module` |
| Logging | `fw_log()` thin wrapper | `framework.log` |
| CPU affinity | Bind threads to cores | `utils.cpu_affinity` |

## When NOT to use this framework

- You need a full ROS2 ecosystem with distributed nodes
- You're building a pure web/mobile app (no hardware interaction)
- You need real-time guarantees (< 1 ms latency)
