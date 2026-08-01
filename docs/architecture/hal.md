# Hardware Abstraction Layer

The HAL decouples application logic from physical hardware. You write code against **Protocols** (structural type interfaces); the platform factory provides the concrete implementations.

## Architecture

```
framework/hal/
├── __init__.py           # Exports Machine, CameraHub
├── machine.py            # Machine.create(config_path) — builds everything
├── camera_hub.py         # CameraHub — singleton, multi-camera registry
└── interface/            # Abstract Protocols
    ├── camera.py         # Camera (read, set, release)
    ├── display.py        # Display (show → bool, close)
    ├── uart.py           # Uart (send, receive, start_receiver)
    └── ai.py             # AIInference (detect, classify, add, switch)
└── platforms/            # Concrete implementations
    ├── mock/             # No hardware needed — dev & testing
    ├── linux/            # V4L2 cameras, serial UART, cv2 display
    └── maixcam2/         # MaixCAM (K210) camera, NPU, display
```

## How It Works

### 1. Configure

```yaml
# project_config.yaml
platform: linux
cameras:
  - source: "/dev/v4l/by-id/usb-...-video-index0"
    width: 640
    height: 480
    fps: 120
uart_defaults:
  port: "/dev/ttyS4"
  baudrate: 921600
ai:
  models:
    - nick_name: yolo11n
      model: "/path/to/model"
  active: yolo11n
```

### 2. Build

```python
from framework.hal import Machine

machine = Machine.create("project_config.yaml")
# Returns a Machine with:
#   machine.camera_hub   — CameraHub (all cameras opened)
#   machine.display      — Display
#   machine.uart         — Uart
#   machine.ai           — AIInference (optional)
```

### 3. Use

```python
# Camera
cam = machine.camera_hub.get("my_cam")
frame = cam.read()

# Display
machine.display.show(frame)

# UART
machine.uart.send(b"hello")
machine.uart.start_receiver(my_callback)

# AI
dets = machine.ai.detect(frame)  # → list[Detection]
```

## Interfaces (Protocols)

The types use `@runtime_checkable` Protocol, so any object with matching methods passes `isinstance` checks:

```python
from framework.hal.interface import Camera

isinstance(mock_cam, Camera)   # True

@runtime_checkable
class Camera(Protocol):
    @property
    def camera_id(self) -> str: ...
    def read(self) -> Optional[np.ndarray]: ...
    @property
    def fps(self) -> float: ...
    def release(self) -> None: ...
    def set(self, prop_id: int, value) -> bool: ...
```

Why Protocols instead of ABCs? Mock objects don't need to inherit from framework base classes, making testing simpler.

## Platforms

### mock
- **Camera**: Generates chessboard/static frames
- **Display**: No-op (logs `show()` calls)
- **UART**: In-memory loopback
- **AI**: Returns empty detections
- **Use case**: Development, CI testing, documentation examples

### linux
- **Camera**: `cv2.VideoCapture` with V4L2, by-id device detection, MJPEG/CUDA support
- **Display**: `cv2.imshow` + `waitKey(1)`, returns `False` on `q`/`ESC`
- **UART**: `pyserial` with receiver thread, reconnection support
- **AI**: Stub (real models not included)
- **Use case**: Orange Pi 5, Raspberry Pi, any Linux SBC

### maixcam2
- **Camera**: `maix.camera.Camera` driver
- **Display**: `maix.display.Display` (lcd)
- **UART**: `maix.serial.Serial` — note `in_waiting` via `ioctl`
- **AI**: `maix.nn` NPU inference
- **Use case**: Sipeed MaixCAM (K210)

## CameraHub

Singleton for managing multiple cameras:

```python
hub = CameraHub.init_instance("linux")
hub.open("front", source="/dev/video0", width=640, height=480)
hub.open("bottom", source="/dev/video1", width=320, height=240)

cam = hub.get("front")
frame = cam.read()

hub.close("front")
hub.release_all()
```

## Adding a New Platform

Create a new directory under `framework/hal/platforms/<name>/` with:

```python
# __init__.py — factory functions
def create_camera(source, width=640, height=480, **kwargs) -> Camera: ...
def create_display() -> Display: ...
def create_uart(port, baudrate=921600) -> Uart: ...
def create_ai() -> AIInference: ...   # optional
```

Then set `platform: <name>` in `project_config.yaml`. The `Machine.create` method dynamically imports `framework.hal.platforms.<name>` and calls your factories.

## AIInference

Multi-model registry with single-active-model constraint (suits NPU hardware).

```python
machine.ai.add("yolo11n", "/path/to/model")
machine.ai.switch("yolo11n")
dets = machine.ai.detect(frame)              # → list[Detection]
classes = machine.ai.classify(frame)         # → list[(class_id, score)]
mask = machine.ai.get_mask(0)                # → Optional[np.ndarray]
```

`Detection` dataclass includes `x, y, w, h, class_id, score, label, angle, keypoints, mask_index`.
