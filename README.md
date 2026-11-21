[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platform-maixcam2%20%7C%20linux%20%7C%20mock-orange.svg)](docs/architecture/hal.md)
[![Architecture](https://img.shields.io/badge/architecture-HAL%20%2B%20state%20machine%20%2B%20event%20bus-lightgrey.svg)](docs/architecture/overview.md)
[![Stars](https://img.shields.io/github/stars/ju1c3rSH/Zulu-Walker.svg?style=social)](https://github.com/ju1c3rSH/Zulu-Walker)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

# Zulu-Walker

> A modular Python framework for vision-driven robotics on single-board computers and Sipeed MaixCAM2 boards - hardware abstraction, a generic state-machine engine, and a pluggable camera pipeline, with only the mission logic left to you.

## Table of Contents

- [Status](#status)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Adding a Vision Solution](#adding-a-vision-solution)
- [Deploying to MaixCAM2](#deploying-to-maixcam2)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Status

**Active - beta.**

Zulu-Walker began as the software stack of a 2026 National Undergraduate Electronics Design Contest entry (vehicle-mounted balancing rolling-ball control) and was then split into a reusable framework plus thin project layers. This branch carries the framework; the complete mission implementation - UART master/slave bridge, ball state estimation with an alpha-beta filter, two-phase rail calibration, WiFi video streaming and record signalling - lives on the `release/2026H` branch.

The core API is stable enough to build on, but expect churn while the contest runs.

## Features

- **Three-layer architecture** - `app/` holds your mission logic, `modules/` holds optional capabilities, `framework/` holds the reusable engine. The framework never imports application code.
- **Hardware abstraction layer** - `Camera`, `Display`, `Uart`, `AIInference`, `Sink`, `Watchdog` and `SysInfo` are structural `Protocol`s, implemented by three backends: `maixcam2` (Axera AX630C NPU, MIPI CSI sensor, LCD, hardware watchdog), `linux` (V4L2 camera, pyserial, OpenCV window) and `mock` (no hardware, for CI).
- **Generic state machine engine** - `State` and `BaseStateMachine` support event-driven and return-value-driven transitions, a `run_to_completion()` cascade guard, per-state and global enter/exit callbacks, and `RLock`-protected state.
- **Event bus** - typed publish/subscribe with thread-safe dispatch and a bounded history buffer.
- **Module manager** - `init -> start -> loop -> stop` lifecycle, per-platform gating, a configurable tick rate, and discovery by package name.
- **Vision pipeline** - capture to `image2cv` to serial task execution to a thread-safe result deque consumed by the main thread, with one NPU inference per frame.
- **Opt-in vision solutions** - cargo, circle target, ring, QR, line following and pendulum rail calibration ship in the repository but stay dormant until you import them.
- **Framed UART protocol** - SOF, length, type, payload and CRC16-CCITT, parsed by a byte-level state machine that re-anchors on noise, plus command and data-stream frames for master/slave links.
- **Debug console** - a Rich TUI with live status, performance gates and a key listener, running on any desktop terminal.
- **Architecture guardrails** - platform-isolation and boot-scope checks run standalone and in pre-commit.
- **No hard dependencies** - `numpy` and `pyyaml` only; `opencv`, `pyserial`, `rich` and `pyzbar` are optional platform extras.

## Installation

Python 3.10 or newer. MaixCAM2 runs MaixPy, which ships its own interpreter and the `maix` module.

```bash
git clone https://github.com/ju1c3rSH/Zulu-Walker.git
cd Zulu-Walker
python -m pip install -r requirements.txt
```

Linux desktop or SBC extras (V4L2 camera, serial port, OpenCV window):

```bash
python -m pip install -r requirements-linux.txt
```

## Quick Start

The checked-in `project_config.yaml` targets the MaixCAM2. On a desktop, set `platform: mock` to run without hardware, or `platform: linux` to drive a real camera and serial port.

```bash
python run.py main                          # run the application
python run.py debug line                    # interactive detector debuggers
python run.py debug cargo -c 0 -W 640 -H 480
```

Available debug runners: `cargo`, `circle`, `ring`, `line`, `pendulum_calib`. Each opens a live preview with a parameter panel; press `q` to quit.

Booting the framework from Python is three objects and one coordinator:

```python
from framework.event_bus import EventBus
from framework.hal import Machine
from framework.module_manager import ModuleManager

bus = EventBus()
machine = Machine.create("project_config.yaml")
manager = ModuleManager(machine, event_bus=bus)
manager.register_many(["zw_opencv_module"])
manager.run_main_loop(coordinator=MyCoordinator(bus))
```

`MyCoordinator` is the only project-specific piece: it drains the event bus, advances the state machines and turns vision results into actuator commands.

## Configuration

All hardware settings live in `project_config.yaml`:

```yaml
platform: maixcam2            # maixcam2 | linux | mock
display_enabled: true

cameras:
  - camera_id: "main"
    source: null              # on-board sensor on MaixCAM2
    width: 1280
    height: 352
    fps: 60
    exposure_us: 3000
    gain: 200
    aec:                      # software auto-exposure (PI controller on ROI brightness)
      enabled: true
      target_mean: 80
      gain_min: 50
      gain_max: 600

uart_defaults:
  port: "/dev/ttyS4"
  baudrate: 921600

ai:
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_1280x352.mud"
      model_type: "auto"      # auto -> maix.nn.YOLO11
  active: "yolo11n"
```

`platform` selects the backend. Optional capabilities - watchdog, memory stats, exit check - are probed by name on each platform, so a backend can add one without the framework knowing about it. `display_enabled: false` is a hard switch: the display is never created and the vision side stops composing frames.

Pipelines live in `modules/zw_opencv_module/config/vision_config.yaml`: each pipeline binds one camera to an ordered list of tasks, and every task can be toggled at runtime.

## Architecture

```
.
├── app/                    # application layer - your mission logic
│   ├── main.py             # boot skeleton: watchdog -> Machine -> ModuleManager -> coordinator
│   └── coordinator.py      # the per-project "dirty code" slot
├── framework/              # reusable core, platform agnostic
│   ├── hal/                # Protocol interfaces + platform backends
│   ├── state_machine/      # State, BaseStateMachine, StateActionBridge
│   ├── event_bus.py        # typed publish/subscribe
│   ├── module_manager.py   # module lifecycle and main loop
│   ├── cmd_queue.py        # thread-safe command queue
│   ├── slot.py             # latest-value slot
│   └── ui_state.py         # UI geometry model
├── modules/                # optional, reusable capabilities
│   ├── zw_opencv_module/   # camera pipeline, processors, detectors
│   ├── zw_uart_module/     # framed UART transport
│   └── zw_wifi_stream/     # JPEG streaming (MaixCAM2 only)
├── utils/                  # logging, TUI debug console, CPU affinity, geometry
├── deploy/maixcam/         # MaixPy package manifest and firmware entry point
├── scripts/                # architecture fitness checks
├── tests/                  # stdlib unittest suite
└── project_config.yaml     # platform, camera, UART and AI configuration
```

How it fits together:

- `framework/` never imports `maix`, `cv2` or `serial` outside `framework/hal/platforms/` - `scripts/check_platform_isolation.py` enforces this.
- The main loop runs in the main thread on a 2 ms tick; capture and detection run in the vision thread; UART reception runs on a platform callback thread.
- State machines are advanced only from the main thread. Other threads publish on the `EventBus` or push into a `Slot` / `CmdQueue`, and the coordinator drains them.
- `app/coordinator.py` is the single project-specific file. Replacing it is the whole integration cost of a new mission.

## Adding a Vision Solution

Detector-backed solutions ship dormant: nothing imports them, they cost nothing at runtime, and they stay out of the device package unless the manifest lists them.

```python
from modules.zw_opencv_module.processors import enable_optional

enable_optional(["TrackCargoProcessor"])   # before pipelines start
```

Then enable the matching task in `vision_config.yaml`. An unknown name raises `KeyError` at startup instead of silently doing nothing later.

## Deploying to MaixCAM2

1. Copy `deploy/maixcam/app.yaml` and `deploy/maixcam/main.py` into the repository root.
2. Adjust the `files` list in `app.yaml`. The default ships the framework core; add `modules/zw_opencv_module/detectors/` and the matching processor only when your project uses them.
3. Build the package with `maixpack` in the repository root, then copy the artifact to the board.

Three rules keep the boot path safe on the device:

- Feed the hardware watchdog first and between every slow step - WiFi bring-up, model load, calibration loops.
- Objects created before `Machine.create()` receive platform capabilities through late-injection setters, never through their constructors.
- No display thread: overlays are composed in the vision thread and the main loop pushes frames to the sinks.

## Development

```bash
python -m pip install -r requirements-dev.txt    # pytest + pre-commit

python -m unittest discover -s tests              # stdlib suite
python -m pytest tests                            # pytest-style contracts
python scripts/check_platform_isolation.py
python scripts/check_boot_scope.py
python -m compileall -q app framework modules utils
```

All of them are wired into `.pre-commit-config.yaml`; `pre-commit install` runs them on every commit.

The tracked suite covers the framework core: `Slot`, `CmdQueue`, `ui_state` and the LCD present cadence. The cadence contract test imports the `maixcam2` HAL, so it is skipped wherever the MaixPy runtime is missing; tests for competition features live on the branch that implements them.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Core framework](docs/architecture/core_framework.md)
- [Hardware abstraction layer](docs/architecture/hal.md)
- [Threading model](docs/architecture/threading.md)
- [State machines](docs/architecture/state_machine.md)
- [Building applications](docs/architecture/building_apps.md)
- [Framework guide](docs/architecture/framework-guide.md) - platform matrix, boot rules, opt-in solutions
- [Competition notes and protocol specs](docs/competition/)

Most documents are written in Chinese; code, comments and configuration keys are in English.

## Contributing

Issues and pull requests are welcome at <https://github.com/ju1c3rSH/Zulu-Walker>.

When touching `framework/`, keep it free of platform imports - `scripts/check_platform_isolation.py` fails the build otherwise. Prefer a capability hook that is probed by name over branching on the platform string, and keep new behaviour in `modules/` or `app/` when it is not generic.

## License

Released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

Share and modify this project for any non-commercial purpose, as long as you give credit and release your contributions under the same license. Commercial use needs a separate agreement with the author. The full legal code is in [LICENSE](LICENSE).
