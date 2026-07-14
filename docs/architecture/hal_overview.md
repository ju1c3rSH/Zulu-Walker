# HAL 架构概览

## 分层架构

```mermaid
graph TB
    subgraph project_config["project_config.yaml"]
        direction LR
        PLATFORM["platform: linux | maixcam2 | mock"]
    end

    subgraph HAL["hal/ — 硬件抽象层"]
        direction TB
        IF["hal/interface/<br/>Camera / Display / Uart / AIInference<br/>Protocol 定义 + @runtime_checkable"]
        PL["hal/platforms/<br/>linux / maixcam2 / mock<br/>平台工厂 + 具体实现"]
    end

    subgraph MODULES["modules/ — 业务模块"]
        direction TB
        VM["vision_module<br/>VisionManager + PipelineCamera<br/>FrameComposer + FFmpegPusher"]
        UM["zw_uart_module<br/>STM32UartInterface<br/>协议帧编解码"]
    end

    subgraph CTX["context/ — 中枢层"]
        direction TB
        EB["EventBus<br/>类型安全发布/订阅"]
        MSM["MissionStateMachine<br/>18 状态 + 3-loop batch"]
        MC["MissionCoordinator<br/>UART ↔ Vision ↔ 状态机"]
    end

    subgraph ENTRY["main.py — 入口"]
        MM["ModuleManager<br/>load → init → start → loop → stop"]
    end

    project_config --> HAL
    HAL --> ENTRY
    ENTRY --> MODULES
    ENTRY --> CTX
    MODULES --> CTX
```

## 数据流

```mermaid
sequenceDiagram
    participant Main as main.py (主循环 ~300Hz)
    participant Vision as vision_module
    participant UART as zw_uart_module
    participant CTX as context (Coordinator)

    loop every tick
        Main->>Vision: module.loop() → display_frame()
        Main->>UART: module.loop() → 串口收发
        Main->>CTX: coordinator.loop() → 状态机

        Vision->>Main: compose_frame() → np.ndarray
        Main->>HAL: machine.display.show(frame) → bool
        HAL-->>Main: True=继续 / False=退出
    end
```

## Machine 与 ModuleManager 分立

```mermaid
classDiagram
    class Machine {
        +CameraHub camera_hub
        +Display display
        +Uart uart
        +AIBackend ai
        +create(config) Machine
        +close()
    }

    class ModuleManager {
        -Machine _machine
        -dict _modules
        -dict _loop_methods
        +load(name) bool
        +get_module(name) module
        +run_main_loop(coordinator)
        +stop_all()
    }

    class CameraHub {
        -dict _cameras
        +open(id, source, w, h) Camera
        +release_all()
        +get(id) Camera
    }

    class VisionManager {
        -dict pipelines
        -FrameComposer composer
        -FFmpegPusher ffmpeg
        +compose_frame() np.ndarray
        +display_frame() bool
        +enable_task(pipe, task)
        +disable_task(pipe, task)
        +set_processor_target(pipe, task, color)
    }

    class PipelineCamera {
        +Camera camera
        +TaskManager task_manager
        +process_frame() tuple
        +enable_task(name)
        +disable_task(name)
        +get_task(name) Task
    }

    Machine --> CameraHub
    Machine --> ModuleManager
    ModuleManager --> VisionManager
    VisionManager --> PipelineCamera
    PipelineCamera --> CameraHub : 通过 hub.get() 获取 Camera
```

## Display 主线程约束

```mermaid
flowchart LR
    subgraph Processing["处理线程 (big cores)"]
        P1["PipelineCamera.process()"]
        P2["compose_frame()"]
    end
    subgraph MainThread["主线程"]
        M1["display_loop()"]
        M2["display.show(frame) → bool"]
        M3["cv2.imshow + waitKey(1)"]
    end

    P1 -->|帧队列| P2
    P2 -->|Queue| M1
    M1 --> M2
    M2 --> M3
    M3 -->|q/ESC| STOP["self._running = False"]
```

## 各平台 Camera 实现

```mermaid
flowchart LR
    subgraph Linux["hal/platforms/linux/"]
        L1["CameraHubLinux<br/>/dev/v4l/by-id/ 探测"]
        L2["LinuxCamera<br/>cv2.VideoCapture + V4L2<br/>采集线程 + Queue"]
    end
    subgraph MaixCam2["hal/platforms/maixcam2/"]
        M1["CameraHubMaixCam2<br/>maix.camera.list_devices()"]
        M2["MaixCam2Camera<br/>maix.camera.Camera<br/>模块级导入"]
    end
    subgraph Mock["hal/platforms/mock/"]
        K1["MockCameraHub<br/>固定 CameraInfo"]
        K2["MockCamera<br/>棋盘格帧"]
    end

    CameraAPI["Camera Protocol<br/>read() → Optional[ndarray]<br/>release()<br/>set(prop_id, value)"] --- Linux
    CameraAPI --- MaixCam2
    CameraAPI --- Mock
```

## 设备标识策略

```mermaid
flowchart TD
    subgraph LinuxUSB["Linux USB Camera"]
        L1["/dev/v4l/by-id/usb-XXXX-*<br/>(稳定标识)"]
        L2["fallback: /dev/videoN<br/>(索引, 重启可能变化)"]
    end
    subgraph LinuxMIPI["Linux MIPI Camera"]
        M1["v4l2-ctl 设备名<br/>'gc4653 0-0030'"]
    end
    subgraph MaixCam2["MaixCAM2"]
        X1["maix.camera.list_devices()<br/>返回值"]
    end

    CI["CameraInfo.id<br/>= 稳定标识"]
    CI --> LinuxUSB
    CI --> LinuxMIPI
    CI --> MaixCam2
```

## 第一阶段实施范围

```mermaid
gantt
    title Phase 1 — HAL 骨架
    dateFormat  YYYY-MM-DD
    section Round 1
    hal/interface/             :a1, 1d
    hal/platforms/linux/       :a2, 1d
    hal/platforms/maixcam2/    :a3, 0.5d
    hal/platforms/mock/        :a4, 0.5d
    hal/__init__.py + project_config :a5, 0.5d
    section Round 2
    ModuleManager + main.py    :b1, 1d
    VisionManager + PipelineCamera :b2, 2d
    CameraHub (单例)           :b3, 0.5d
    UART 模块适配              :b4, 1d
    mission_context 桥接       :b5, 0.5d
    section Round 3
    重命名 zw_opencv_module → vision_module :c1, 1d
    更新所有 15+ import       :c2, 0.5d
```
