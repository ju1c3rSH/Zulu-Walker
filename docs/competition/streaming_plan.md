# WiFi 图传方案 — RTSP 推流

> 适用项目：Zulu-Walker（MaixCAM2 + MSPM0）
> 用途：MaixCAM2 通过 WiFi 向场外 PC/PAD 推送摆杆钢球实时画面，满足赛题第 1 项要求
>
> 赛题要求原文见 [`H题_车载平衡滚球运动控制系统.md`](H题_车载平衡滚球运动控制系统.md)

---

## 1. 方案概述

采用 **纯 RTSP 硬件编码推流**，MaixCAM2 的 AX630C VPU 直接完成 H.264 编码，零 CPU 开销。接收端使用 VLC Media Player 播放并录制视频。

```
 ┌──────────────────┐          WiFi           ┌──────────────┐
 │    MaixCAM2      │      ──────────→        │  PC / PAD    │
 │  ┌────────────┐  │    RTSP H.264 流        │  VLC 3.0.20  │
 │  │  Camera    │──┼─────────────────────────→│  播放 + 录制  │
 │  │  640x640   │  │    rtsp://IP:8554/live   │  .mp4 存档   │
 │  └────────────┘  │                          │              │
 │  │  VPU         │  │  硬件编码，零 CPU      └──────────────┘
 │  │  H.264/H.265 │  │
 │  └────────────┘  │
 └──────────────────┘
```

### 1.1 为什么选 RTSP

| 方案 | CPU 开销 | 延迟 | 接收端要求 | 录制能力 |
|------|---------|------|-----------|---------|
| **RTSP（本方案）** | **零**（硬件编码） | 低 | VLC / ffplay | ✅ 原生支持 |
| HTTP MJPEG | 中（JPEG 编码） | 中 | 浏览器 | 需额外工具 |
| WebRTC | 低 | 低 | 浏览器 | 需 JS 录制 |
| 本地录像 | — | — | — | 需赛后导出 |

### 1.2 性能特征

| 参数 | 值 |
|------|-----|
| 分辨率 | 640×640（与 YOLO 输入一致） |
| 编码格式 | H.264（硬件） |
| 码率 | 自动（CBR ~2-4 Mbps） |
| CPU 占用 | ≈ 0%（硬件 offload） |
| 延迟 | < 200ms（WiFi 环境） |

---

## 2. WiFi 连接

### 2.1 连接方式

MaixCAM2 作为 WiFi Station 连接到场外路由器/热点：

```python
from maix import network

# WiFi Station 模式
wlan = network.WLAN()
wlan.connect("SSID", "password")
print(f"IP: {wlan.ifconfig()[0]}")  # 打印本机 IP
```

### 2.2 配置

```yaml
# project_config.yaml
streaming:
  wifi_ssid: "ZuluWalker_5G"       # 路由器 SSID
  wifi_password: "comp2026"         # 密码
  enabled: true                     # 启动时自动开启推流
```

### 2.3 备选：AP 模式

若赛场无 WiFi 路由器，MaixCAM2 可作为热点直连 PC：

```python
wlan = network.WLAN()
wlan.ap_mode("ZuluWalker", "12345678")  # 自建热点
```

PC 连接此热点后访问 `rtsp://192.168.1.1:8554/live`。

> **不推荐**：AP 模式信号强度、吞吐量和稳定性均弱于 Station 模式。

---

## 3. RTSP 推流实现

### 3.1 核心代码

```python
from maix import camera, image, rtsp

cam = camera.Camera(640, 640, image.Format.FMT_YVU420SP)
# 第二通道用于本地检测处理
cam2 = cam.add_channel(640, 640)

server = rtsp.Rtsp()
server.bind_camera(cam)   # 主通道 → 硬件编码 → RTSP
server.start()

url = server.get_url()
print(f"RTSP: {url}")     # rtsp://192.168.x.x:8554/live

# cam2 继续用于 YOLO 检测 + BallPendulumProcessor
```

### 3.2 注意事项

- `bind_camera(cam)` 后，主 `cam` 对象不能再直接 `read()`，需通过 `cam.add_channel()` 创建第二通道
- RTSP 模块只支持 `image.Format.FMT_YVU420SP`（NV21）格式
- 多路 channel 需在 `bind_camera` 之前创建

### 3.3 生命周期

集成在 `modules/zw_wifi_stream` 模块中：

```python
# zw_wifi_stream/__init__.py
def init(machine, event_bus, **kwargs):
    # 1. 连接 WiFi
    # 2. 从 machine 获取 cam 实例
    # 3. 创建 RtspStreamer, bind_camera, start

def loop():
    pass  # RTSP 在后台线程运行，无需轮询

def stop():
    # 释放 RTSP server
```

---

## 4. 接收端使用（VLC）

### 4.1 播放

1. 打开 VLC Media Player（≥ 3.0.20）
2. `媒体 → 打开网络串流`
3. 输入 `rtsp://192.168.x.x:8554/live`（IP 为 MaixCAM2 实际 IP）
4. 点击播放

### 4.2 录制

VLC 播放时同步录制：

```
视图 → 高级控制 → 录制按钮（红色圆点）
```

或命令行录制：

```bash
vlc rtsp://192.168.x.x:8554/live --sout="#transcode{vcodec=h264}:file{dst=record.mp4}"
```

### 4.3 备选播放器

| 播放器 | 命令 |
|--------|------|
| ffplay | `ffplay rtsp://192.168.x.x:8554/live` |
| ffmpeg 录制 | `ffmpeg -i rtsp://192.168.x.x:8554/live -c copy record.mp4` |

---

## 5. 与检测系统共存

MaixCAM2 的资源分配：

```
Camera Sensor (640x640)
  ├── cam  (主通道) → RTSP 硬件编码 → WiFi 推流
  └── cam2 (第二通道) → YOLO(NPU) → BallPendulumProcessor → UART
                            ↓
                       LCD显示标注帧
```

| 资源 | 占用 | 说明 |
|------|------|------|
| VPU (H.264) | ✅ RTSP | 硬件编码，独立于 CPU |
| NPU (YOLO) | ✅ 检测 | 硬件推理，独立于 CPU |
| CPU | ≈ 5% | 仅 BallPendulumProcessor 浮点计算 + 协议打包 |
| UART | ✅ 数据 | DMA 传输，不占 CPU |
| WiFi | ✅ 推流 | 与检测并行无冲突 |

---

## 6. 代码映射

| 组件 | 文件 | 说明 |
|------|------|------|
| 模块入口 | `modules/zw_wifi_stream/__init__.py` | init/start/stop/loop |
| 流管理器 | `modules/zw_wifi_stream/rtsp_streamer.py` | RtspStreamer 封装 |
| WiFi 连接 | `modules/zw_wifi_stream/wifi_manager.py` | 连接/重连/状态 |
| 主入口 | `app/main.py` | cam→cam2→RTSP 启动流程 |
| 配置 | `project_config.yaml` | streaming 段 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| **v1.0** | 2026-07-29 | 初版方案 |
