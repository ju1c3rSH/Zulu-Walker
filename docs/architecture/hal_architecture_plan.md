# HAL 架构重构 — 完整实施计划

## 设计决策总表

| 决策项 | 结论 |
|--------|------|
| Display 归属 | **主循环控制**，VisionManager 不持有 Display；`show()` 返回 `bool`（True=继续，False=退出） |
| VisionManager 使用 HAL Camera | 通过 `CameraHub` 获取 `hal.Camera` 实例，**不直接 import CameraStream** |
| FrameComposer | VisionManager 保留，多相机画面合成正常使用 |
| FFmpegPusher | 保留字段，按配置激活 |
| profiler | 保留 |
| ModuleManager | 新建 `utils/module_manager.py`，接收 `Machine` 实例 |
| CameraHub | **单例**模式，`open()` 返回 `Camera` Protocol，`release_all()` 统一释放 |
| `connect_camera` → `connect_vision` | 统一方法名，与 `CameraManager` 解耦 |
| 设备标识 | `CameraInfo.id` 使用 `/dev/v4l/by-id/` 稳定路径，fallback 到设备名 |
| 平台隔离 | maixcam2/ 所有 `maix import` 延迟到函数内 |
| Module 重命名 | 本轮**不改名**（`zw_opencv_module`），后续 PR 做 |
| 迁移方案 | `camera_config.yaml` 保留不动，VisionManager 读 `vision_config.yaml` |
| 向后兼容 | `_LegacyCameraManagerShim` + `CameraManager` 垫片 |
| `run.py` debug | 暂时保留旧 `CameraManager` 垫片，后续 PR 适配 |

## 目录结构

```
project_config.yaml

hal/
├── __init__.py                     # 根模块，导出 create_hub()
├── interface/
│   ├── __init__.py
│   ├── camera.py                   # Camera Protocol (@runtime_checkable)
│   ├── display.py                  # Display Protocol (show() → bool)
│   ├── uart.py                     # Uart Protocol (含 in_waiting)
│   └── ai.py                       # AIInference Protocol (预留)
└── platforms/
    ├── linux/
    │   ├── __init__.py
    │   ├── camera.py               # LinuxCamera (cv2.VideoCapture + V4L2)
    │   ├── display.py              # LinuxDisplay (cv2.imshow)
    │   └── uart.py                 # LinuxUart (pyserial)
    ├── maixcam2/
    │   ├── __init__.py
    │   ├── camera.py               # MaixCam2Camera (存根，延迟导入)
    │   ├── display.py              # MaixCam2Display (存根)
    │   └── uart.py                 # MaixCam2Uart (存根)
    └── mock/
        ├── __init__.py
        ├── camera.py               # MockCamera (棋盘格帧)
        ├── display.py              # MockDisplay (日志)
        └── uart.py                 # MockUart (环回)

utils/
└── module_manager.py               # ModuleManager 类

modules/
├── zw_opencv_module/
│   ├── __init__.py                 # init(machine) 签名
│   ├── vision_manager.py           # VisionManager + CameraManager 垫片
│   ├── camera_hub.py               # CameraHub (单例)
│   ├── pipeline_camera.py          # PipelineCamera (持有 hal.Camera)
│   ├── task_manager.py             # 保留
│   ├── frame_composer.py           # 保留
│   ├── ffmpeg_pusher.py            # 保留
│   ├── performance.py              # 保留
│   ├── processors/                 # 保留
│   ├── detectors/                  # 保留
│   └── config/
│       ├── camera_config.yaml      # 旧配置，过渡期保留
│       └── vision_config.yaml      # 新配置
└── zw_uart_module/
    ├── __init__.py                 # init(machine) 签名
    ├── uart_driver.py              # STM32UartInterface(uart: Uart)
    └── protocol.py                 # 不动

context/
├── mission_context.py              # connect_vision(VisionManager)

main.py                             # Machine.create() + ModuleManager(machine, bus)

run.py                              # debug 入口暂用 CameraManager 垫片
```

## 文件清单

### 新建（21 文件）

| # | 文件 | 内容 |
|---|------|------|
| 1 | `project_config.yaml` | 平台选择 + 默认参数 |
| 2 | `hal/__init__.py` | 根模块，导出 `create_hub()` |
| 3 | `hal/interface/__init__.py` | 导出所有 Protocol |
| 4 | `hal/interface/camera.py` | `Camera` Protocol (`read`, `release`, `set`, `fps`, `camera_id`) |
| 5 | `hal/interface/display.py` | `Display` Protocol (`show() → bool`, `close`) |
| 6 | `hal/interface/uart.py` | `Uart` Protocol (`in_waiting`, `send`, `receive_all`, `start_receiver`) |
| 7 | `hal/interface/ai.py` | `AIInference` Protocol（预留） |
| 8 | `hal/platforms/linux/__init__.py` | 导出 LinuxCamera, LinuxDisplay, LinuxUart |
| 9 | `hal/platforms/linux/camera.py` | LinuxCamera（V4L2 + 采集线程 + Queue） |
| 10 | `hal/platforms/linux/display.py` | LinuxDisplay（cv2.imshow, `show() → bool`） |
| 11 | `hal/platforms/linux/uart.py` | LinuxUart（pyserial + 后台接收线程） |
| 12 | `hal/platforms/maixcam2/__init__.py` | 存根导出 |
| 13 | `hal/platforms/maixcam2/camera.py` | 存根（延迟导入） |
| 14 | `hal/platforms/maixcam2/display.py` | 存根 |
| 15 | `hal/platforms/maixcam2/uart.py` | 存根 |
| 16 | `hal/platforms/mock/__init__.py` | 导出 MockCamera, MockDisplay, MockUart |
| 17 | `hal/platforms/mock/camera.py` | MockCamera（棋盘格帧） |
| 18 | `hal/platforms/mock/display.py` | MockDisplay（日志） |
| 19 | `hal/platforms/mock/uart.py` | MockUart（环回） |
| 20 | `utils/module_manager.py` | ModuleManager 类 |
| 21 | `modules/zw_opencv_module/config/vision_config.yaml` | 新摄像头配置格式 |

### 修改（9 文件）

| # | 文件 | 改动 |
|---|------|------|
| 22 | `main.py` | 桥接改为 `connect_vision()` |
| 23 | `modules/zw_opencv_module/__init__.py` | `init(machine)` 签名，导出 `get_vision_manager()` |
| 24 | `modules/zw_opencv_module/vision_manager.py` | 重写为 `VisionManager` + `CameraManager` 垫片 |
| 25 | `modules/zw_opencv_module/camera_hub.py` | 新建（单例 CameraHub） |
| 26 | `modules/zw_opencv_module/pipeline_camera.py` | 新建（PipelineCamera 持有 hal.Camera） |
| 27 | `modules/zw_uart_module/__init__.py` | `init(machine)` 签名 |
| 28 | `modules/zw_uart_module/uart_driver.py` | 构造参数改为 `Uart` Protocol |
| 29 | `context/mission_context.py` | `connect_camera()` → `connect_vision()` |
| 30 | `utils/camera_misc_util.py` | `CameraInfo` → `DeviceCameraInfo` |

### 保留不动

| 文件 | 原因 |
|------|------|
| `modules/zw_opencv_module/config/camera_config.yaml` | 过渡期保留，用户手动迁移 |
| `modules/zw_opencv_module/task_manager.py` | 不动 |
| `modules/zw_opencv_module/frame_composer.py` | 不动 |
| `modules/zw_opencv_module/ffmpeg_pusher.py` | 不动 |
| `modules/zw_opencv_module/performance.py` | 不动 |
| `modules/zw_opencv_module/processors/` | 不动 |
| `modules/zw_opencv_module/detectors/` | 不动 |
| `run.py` | debug 入口暂用 CameraManager 垫片 |
| `utils/serial_controller.py` | 暂保留，待确认是否还有引用 |

## 关键接口定义

### `hal/interface/camera.py`

```python
@runtime_checkable
class Camera(Protocol):
    @property
    def camera_id(self) -> str: ...
    def read(self) -> Optional[np.ndarray]: ...  # 非阻塞，返回最新 BGR 帧或 None
    @property
    def fps(self) -> float: ...
    def release(self) -> None: ...
    def set(self, prop_id: int, value) -> bool: ...  # V4L2 属性控制
```

### `hal/interface/display.py`

```python
@runtime_checkable
class Display(Protocol):
    def show(self, frame: np.ndarray) -> bool: ...  # True=继续, False=退出
    def close(self) -> None: ...
```

### `hal/interface/uart.py`

```python
@runtime_checkable
class Uart(Protocol):
    @property
    def in_waiting(self) -> int: ...
    @property
    def is_connected(self) -> bool: ...
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def send(self, data: bytes) -> int: ...
    def receive(self, size: int = 1) -> Optional[bytes]: ...
    def receive_all(self) -> Optional[bytes]: ...
    def start_receiver(self, callback: Callable) -> None: ...
    def stop_receiver(self) -> None: ...
```

### CameraHub（单例）

```python
class CameraHub:
    _instance: Optional["CameraHub"] = None

    @classmethod
    def init_instance(cls, platform: str) -> "CameraHub": ...
    @classmethod
    def instance(cls) -> Optional["CameraHub"]: ...

    def open(self, camera_id, source, width, height, **kwargs) -> Camera: ...
    def get(self, camera_id) -> Optional[Camera]: ...
    def close(self, camera_id): ...
    def release_all(self): ...          # 异常安全：每个 release 独立 try/except
    def list_ids(self) -> list[str]: ...
```

## 平台实现要点

### LinuxCamera

- `cv2.VideoCapture(source, cv2.CAP_V4L2)` 先，fallback 到默认
- 独立采集线程绑定到小核 `[0,1,2,3]`
- `Queue(maxsize=2)`，满时丢弃旧帧
- `read()` 非阻塞，`queue.get_nowait()`
- 亲和性出错时 `logging.error` + `DebugConsole` 输出

### LinuxDisplay

- `cv2.imshow` + `cv2.waitKey(1)`
- 检测 `q` / `ESC` 返回 `False`
- 必须在主线程调用

### LinuxUart

- pyserial + `threading.Lock`
- `start_receiver` 启动后台接收线程
- `receive_all()` 非阻塞，基于 `in_waiting`
- `send()` 加锁

### MockCamera

- 返回带 `MOCK` 文字的棋盘格帧
- `release()` 有 `self` 参数

### MockUart

- bytearray 环回
- `send()` 模拟 MCU 回复 ARRIVED 帧

## 配置迁移对照

### `camera_config.yaml` → `vision_config.yaml`

| 旧字段 | 新字段 | 说明 |
|--------|--------|------|
| `system.output_width` | 移入 FrameComposer 参数 | — |
| `system.layout` | 同上 | — |
| `system.enable_streaming` | VisionManager.config | — |
| `system.rtmp_url` | 同上 | — |
| `system.enable_local_display` → `project_config.yaml` | `display.enabled` | — |
| `cameras[].source` (int) | `cameras[].source` (int/str) | 不强制改字段名 |
| `cameras[].focal_length_*` | 保留 | 标定参数 |
| `cameras[].camera_stream_queue_size` | 保留 | 队列大小 |
| `cameras[].tasks` | 保留不变 | — |
| —— 新增 —— | `cameras[].fps` | 显式帧率 |

### `project_config.yaml` 格式

```yaml
platform: "linux"        # 可选: linux, maixcam2, mock

# camera_defaults 仅供 CameraHub 在没有其他配置时使用
camera_defaults:
  width: 640
  height: 480
  fps: 120

uart_defaults:
  port: "/dev/ttyS4"
  baudrate: 921600
```

## 实施阶段

### Phase 1 Round 1 — HAL 骨架

| Step | 内容 | 文件 | 依赖 |
|------|------|------|------|
| 1 | `hal/interface/` 5 文件 | #3~7 | 无 |
| 2 | `hal/platforms/linux/` 4 文件 | #8~11 | Step 1 |
| 3 | `hal/platforms/maixcam2/` 4 文件 | #12~15 | Step 1 |
| 4 | `hal/platforms/mock/` 4 文件 | #16~19 | Step 1 |
| 5 | `hal/__init__.py` + `project_config.yaml` | #1, #2 | Step 2~4 |

### Phase 1 Round 2 — 模块适配

| Step | 内容 | 文件 | 依赖 |
|------|------|------|------|
| 6 | `camera_hub.py` 单例 | #25 | Step 2~4 |
| 7 | `pipeline_camera.py` | #26 | Step 6 |
| 8 | `vision_manager.py`（VisionManager + 垫片） | #24 | Step 6~7 |
| 9 | `__init__.py` 改 | #23 | Step 8 |
| 10 | `vision_config.yaml` 新建 | #21 | — |
| 11 | `module_manager.py` | #20 | — |
| 12 | `main.py` 改 | #22 | Step 9, 11 |
| 13 | UART 模块改（`__init__.py` + `uart_driver.py`） | #27, #28 | Step 2 |
| 14 | `mission_context.py` 改 | #29 | Step 8 |
| 15 | `camera_misc_util.py` 改名 | #30 | — |

### Phase 2 — 收尾（后续 PR）

| Step | 内容 |
|------|------|
| 16 | 重命名 `zw_opencv_module` → `vision_module`，更新所有 import |
| 17 | `run.py` debug 入口适配 HAL |
| 18 | 更新 `AGENTS.md` 和相关文档 |
| 19 | 确认 `utils/serial_controller.py` 是否可以退役 |
| 20 | 旧 `camera_config.yaml` 删除 |

### Phase 3 — MaixCAM2 适配（后续迭代）

| Step | 内容 |
|------|------|
| 21 | 实现 `maixcam2/camera.py`（用 maix.camera 替代存根） |
| 22 | 实现 `maixcam2/display.py`（maix.display） |
| 23 | 实现 `maixcam2/uart.py`（pyserial + pinmux） |
| 24 | 训练 YOLO11n 模型，添加 AI Backend |

## 关键纠正点（相对初版 Plan 的更新）

| 问题 | 初版 | 修正版 |
|------|------|--------|
| VisionManager 使用 CameraStream | 是 | **改为 hal.Camera via CameraHub** |
| Display.show() 返回值 | 无 | **返回 bool**（True=继续，False=退出） |
| connect_camera / connect_vision | 混乱 | **统一为 connect_vision** |
| CameraHub 生命周期 | 无 release_all | **有 release_all()，异常安全** |
| Uart Protocol 定义 | 散落在各文件 | **统一在 hal/interface/uart.py** |
| run.py 调试入口 | 未提及 | **暂用 CameraManager 垫片** |
| `__init__.py` 导出 | 未更新 | **导出 get_vision_manager() + 垫片** |
| 向后兼容垫片 | 无 | **_LegacyCameraManagerShim + CameraManager** |
| Machine.close() | 简单 close | **每个资源独立 try/except** |
