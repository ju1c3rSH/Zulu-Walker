# Zulu-Walker Framework

A lightweight, modular Python framework for building **robot vision and control applications**. Designed for single-board computers (Orange Pi, Raspberry Pi, MaixCAM) with camera and serial connectivity.

## Architecture at a Glance

```
project/
├── framework/          ← Reusable core (no app-specific code)
├── modules/            ← Optional pre-built modules
├── app/                ← Your application code
├── run.py              ← Entry point
└── project_config.yaml ← Platform & hardware config
```

### Layers

| Layer | Purpose | Example contents |
|-------|---------|-----------------|
| **framework/** | Reusable engine — state machine, HAL, event bus, module loader | `state_machine/`, `hal/`, `event_bus.py`, `module_manager.py` |
| **modules/** | Optional pre-built capabilities — camera pipeline, UART comms | `zw_opencv_module/`, `zw_uart_module/` |
| **app/** (yours) | Application-specific code — mission logic, custom behaviors | Your `coordinator.py`, `main.py`, etc. |

## Features

- **State Machine Framework** — Generic `BaseStateMachine` with event-driven transitions, `run_to_completion` cascade, enter/exit callbacks, and reusable `VisualStateMachine` for tracking
- **Hardware Abstraction Layer** — Swap hardware without touching application code (mock / linux / maixcam2)
- **Module System** — `ModuleManager` auto-loads modules by name, manages lifecycle (`init` → `start` → `loop` → `stop`)
- **Event Bus** — Typed publish/subscribe with thread-safe dispatch and history
- **Platform-ready** — Ships with mock (dev), Linux (V4L2 + UART), and MaixCAM2 (K210 NPU) backends
- **Zero production dependencies** — `numpy` + `pyyaml` only; platform extras optional

## Quick Start

```bash
# Install
pip install numpy pyyaml

# Run with mock platform (no hardware needed)
python run.py

# Customize hardware in project_config.yaml
```

To create a new application, see [Building Applications](architecture/building_apps.md).

## What You Can Build

- **Robotics** — Vision-guided robot arm, line follower, autonomous rover
- **Smart Camera** — Object detection, tracking, QR scanning, RTMP streaming
- **Drone Ground Station** — Telemetry display, mission control, camera feed
- **Automation Controller** — Serial-based PLC replacement with camera feedback
- **Education Platform** — Learn state machines, computer vision, embedded systems

## Project Structure

```
framework/
├── event_bus.py                  # Pub/sub event bus
├── log.py                        # Framework logger
├── module_manager.py             # Module lifecycle manager
├── visual_state_machine.py       # Reusable 5-state tracking FSM
├── state_machine/
│   ├── base.py                   # Generic BaseStateMachine + State
│   └── bridge.py                 # StateActionBridge
└── hal/
    ├── machine.py                # Top-level Machine (create from config)
    ├── camera_hub.py             # CameraHub (singleton, multi-camera)
    ├── interface/                # Protocols (Camera, Display, Uart, AI)
    └── platforms/                # Implementations
        ├── mock/                 # Software-only, no hardware
        ├── linux/                # V4L2, serial, cv2 display
        └── maixcam2/             # K210 NPU, MaixCAM camera

modules/
├── zw_opencv_module/             # Camera processing pipeline
└── zw_uart_module/               # Serial UART communication

utils/                            # Shared utilities (cpu_affinity, logging, etc.)
```

## Next Steps

- [Architecture Overview](architecture/overview.md)
- [Core Framework](architecture/core_framework.md)
- [Hardware Abstraction Layer](architecture/hal.md)
- [Threading Model](architecture/threading.md)
- [Building Applications](architecture/building_apps.md)
