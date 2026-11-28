[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platform-MaixCAM%20%7C%20Linux%20%7C%20Windows-orange.svg)](#supported-platforms)
[![Engines](https://img.shields.io/badge/engines-maix__nn%20%7C%20onnxruntime%20%7C%20ncnn-blueviolet.svg)](#inference-engines)
[![Stars](https://img.shields.io/github/stars/ju1c3rSH/Zulu-Walker.svg?style=social)](https://github.com/ju1c3rSH/Zulu-Walker)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

# Zulu-Walker

> A modular Python framework for vision-driven robotics - hardware abstraction, a generic state-machine engine, a pluggable inference kernel and a data-only HUD, with only the mission logic left to you.

## Table of Contents

- [Status](#status)
- [Features](#features)
- [Supported Platforms](#supported-platforms)
- [Inference Engines](#inference-engines)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Adding a Vision Solution](#adding-a-vision-solution)
- [Deploying to MaixCAM](#deploying-to-maixcam)
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
- **Hardware abstraction layer** - `Camera`, `Display`, `Uart`, `AIInference`, `FrameSink`, `Watchdog` and `SysInfo` are structural `Protocol`s with four backends: MaixCAM, Linux, Windows and `mock` (no hardware, for CI).
- **Pluggable inference engines** - one `ai.engine` key selects `maix_nn` (board NPU), `onnxruntime` or `ncnn`. Engines are deployment configuration, not code branches: no candidate chain, no silent downgrade.
- **Generic state machine engine** - `State` and `BaseStateMachine` support event-driven and return-value-driven transitions, a `run_to_completion()` cascade guard, per-state and global enter/exit callbacks, and `RLock`-protected state.
- **Event bus** - typed publish/subscribe with thread-safe dispatch and a bounded history buffer.
- **Module manager** - `init -> start -> loop -> stop` lifecycle, per-platform gating, a configurable tick rate, and discovery by package name.
- **Vision pipeline** - capture to `image2cv` to serial task execution, with results delivered through a latest-value slot consumed by the main thread.
- **Data-only HUD** - a layout turns `ViewState` into a `Scene` of primitives plus hit boxes; the same layout renders through OpenCV or straight onto a `maix.Image`. Layouts are registered by name and can be replaced from configuration.
- **Opt-in vision solutions** - cargo, circle target, ring, line-following and EOD-colour detectors ship in the repository but stay dormant until you import them.
- **Framed UART protocol** - SOF, length, type, payload and CRC16-CCITT, parsed by a byte-level state machine that re-anchors on noise, plus command and data-stream frames for master/slave links.
- **Debug console** - a Rich TUI with live status, performance gates and a key listener, running on any desktop terminal.
- **Architecture guardrails** - platform-isolation, boot-scope and device-package checks run standalone and in pre-commit.
- **No hard dependencies** - `numpy` and `pyyaml` only; `opencv`, `pyserial`, `rich`, `onnxruntime` and `ncnn` are optional platform extras.

## Supported Platforms

| Platform | `platform` value | Camera | Display | Notes |
|---|---|---|---|---|
| MaixCAM2 | `maixcam2` | MIPI CSI (`maix.camera`) | 640×480 LCD (`maix.display`) | AX630C NPU, hardware watchdog, I2C gauge |
| MaixCAM / Pro | `maixcam1` | MIPI CSI | 552×368 touchscreen | Same HAL contracts, smaller panel |
| Linux | `linux` | V4L2 (`cv2.VideoCapture`) | OpenCV window | Panel UI with mouse input |
| Windows | `linux` | DirectShow (`cv2.VideoCapture`) | OpenCV window | Reuses the linux backend; no serial required |
| None | `mock` | synthetic frames | none | For CI and headless tests |

Optional capabilities - watchdog, memory stats, exit check, panel - are probed by name on each platform, so a backend can add one without the framework knowing about it.

## Inference Engines

| Engine | `ai.engine` | Runs on | Artifact |
|---|---|---|---|
| MaixPy NPU | `maix_nn` | MaixCAM / MaixCAM2 | `.mud` + `manifest.yaml` |
| ONNX Runtime | `onnxruntime` | Linux / Windows | `.onnx` + `manifest.yaml` |
| ncnn | `ncnn` | Linux / Windows | `.param` / `.bin` + `manifest.yaml` |
| Platform backend | *(omit the key)* | MaixCAM | `.mud` only, no manifest |

`maix_nn`, `onnxruntime` and `ncnn` all run through the same kernel (`framework/inference/`): the manifest pins the artifact, the labels and the frame contract, so swapping an engine does not change calling code. Leaving `ai.engine` empty falls back to the platform's own backend, which reads `.mud` directly and ignores manifests.

An engine named in configuration but unavailable fails at startup - there is no fallback chain. `python scripts/check_accelerators.py` reports which execution providers are actually live.

## Installation

Python 3.10 or newer. MaixCAM runs MaixPy, which ships its own interpreter and the `maix` module.

```bash
git clone https://github.com/ju1c3rSH/Zulu-Walker.git
cd Zulu-Walker
python -m pip install -r requirements.txt
```

Desktop extras (OpenCV window, serial port, inference engines):

```bash
python -m pip install -r requirements-linux.txt
```

## Quick Start

The checked-in `project_config.yaml` targets the MaixCAM2. On a desktop, set `platform: mock` to run without hardware, or `platform: linux` to drive a real camera and serial port.

```bash
python run.py main                          # run the application
python run.py main --config configs/project_config.desktop.yaml
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
platform: maixcam2            # maixcam2 | maixcam1 | linux | mock

cameras:
  - camera_id: "main"
    source: null              # on-board sensor on MaixCAM
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
  engine: "maix_nn"           # maix_nn | onnxruntime | ncnn | omit for the platform backend
  models:
    - nick_name: "yolo11n"
      model: "/root/models/steelball_1280x352/steelball_1280x352.mud"
      manifest: "/root/models/steelball_1280x352/manifest.yaml"
  active: "yolo11n"

display:
  backend: "panel"            # desktop only: an OpenCV window carrying the HUD
  layout: "desktop"           # device | desktop | "module.path:ClassName"
```

`platform` selects the backend. `display_enabled: false` is a hard switch: the display is never created and the vision side stops composing frames.

The HUD layout is resolved by name at panel creation; an unknown name fails there rather than on the first frame. `examples/custom_layout.py` is a runnable example of providing your own - the layout class itself is about forty lines.

Pipelines live in `modules/zw_opencv_module/config/vision_config.yaml`: each pipeline binds one camera to an ordered list of tasks, and every task can be toggled at runtime.

## Adding a Vision Solution

Detector-backed solutions ship dormant: nothing imports them, they cost nothing at runtime, and they stay out of the device package unless the manifest lists them.

```python
from modules.zw_opencv_module.processors import enable_optional

enable_optional(["TrackCargoProcessor"])   # before pipelines start
```

Then enable the matching task in `vision_config.yaml`. An unknown name raises `KeyError` at startup instead of silently doing nothing later.

## Deploying to MaixCAM

1. Copy `deploy/maixcam/app.yaml` and `deploy/maixcam/main.py` into the repository root.
2. Adjust the `files` list in `app.yaml`. It is an exact allow-list: a module that is imported but not listed is not shipped, and the failure appears only after flashing.
3. Verify the list is a closed dependency set, then build with `maixpack`:

```bash
python scripts/check_package.py
maixpack
```

Three rules keep the boot path safe on the device:

- Feed the hardware watchdog first and between every slow step - WiFi bring-up, model load, calibration loops.
- Objects created before `Machine.create()` receive platform capabilities through late-injection setters, never through their constructors.
- No display thread: overlays are composed in the vision thread and the main loop pushes frames to the sinks.

## Development

```bash
python -m pip install -r requirements-dev.txt    # pytest + pre-commit

python -m pytest tests -q                        # framework contracts
python scripts/check_platform_isolation.py
python scripts/check_boot_scope.py
python scripts/check_package.py
python -m compileall -q app framework modules utils
```

All of them are wired into `.pre-commit-config.yaml`; `pre-commit install` runs them on every commit.

The tracked suite covers the framework core: `Slot`, `CmdQueue`, `ui_state`, the LCD present cadence, the frame accessors, the layout arithmetic, the composer and the pointer. Tests for competition features live on the branch that implements them.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Core framework](docs/architecture/core_framework.md)
- [Hardware abstraction layer](docs/architecture/hal.md)
- [Threading model](docs/architecture/threading.md)
- [State machines](docs/architecture/state_machine.md)
- [Building applications](docs/architecture/building_apps.md)
- [Framework guide](docs/architecture/framework-guide.md) - platform matrix, boot rules, opt-in solutions
- [Inference kernel](docs/architecture/inference/) - engine / artifact / platform separation
- [Desktop inference backends](docs/architecture/desktop_inference_backends.md)
- [Competition notes and protocol specs](docs/competition/)

Most documents are written in Chinese; code, comments and configuration keys are in English.

## Contributing

Issues and pull requests are welcome at <https://github.com/ju1c3rSH/Zulu-Walker>.

When touching `framework/`, keep it free of platform imports - `scripts/check_platform_isolation.py` fails the build otherwise. Prefer a capability hook that is probed by name over branching on the platform string, and keep new behaviour in `modules/` or `app/` when it is not generic.

After adding an import anywhere in the boot path, run `scripts/check_package.py`: the device package is an allow-list, and a missing entry is invisible until the board fails to start.

## License

Released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

Share and modify this project for any non-commercial purpose, as long as you give credit and release your contributions under the same license. Commercial use needs a separate agreement with the author. The full legal code is in [LICENSE](LICENSE).
