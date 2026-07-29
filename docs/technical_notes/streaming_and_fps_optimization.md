# 推流架构 — 最终技术报告

> MaixCAM2 (Axera AX630C) + YOLO11n 钢珠检测 + WiFi 远程图传  
> 2026 电子设计竞赛 H 题 · 车载平衡滚球运动控制系统

---

## 1. 结论：不可调和的三方约束

MaixCAM2 的硬件固件存在一个**死结**，单相机传感器无法同时满足以下三方：

```
                    单传感器 = 单 Camera 对象 = 单格式
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
RTSP bind_camera()          模型 detect(img)              LCD + IDE 预览
  需: FMT_YVU420SP          需: FMT_RGB888               支持: 两者均可
  拒绝: RGB888              拒绝: NV21
  无 push_frame API         无自动格式转换
```

| 路线 | 相机 | 模型 | RTSP | 结果 |
|------|:---:|:---:|:---:|------|
| A: 相机 = RGB888 | ✓ AI _raw 零拷贝 | ✓ | **✗** `support FMT_YVU420SP only!` | RTSP 启动报错 |
| B: 相机 = NV21 | — | **✗** `input_type: RGB888, image format: YVU420SP` | ✓ | 模型检测报错 |
| C: NV21 → cv2 RGB | ✓`to_bytes` 有数据 | **✗** 转出的帧无法注入 RTSP | ✓ | RTSP 无 push_frame API |
| D: RGB888 → cv2 NV21 | ✓ | ✓ | **✗** 同上 | 同上 |
| E: 重训模型 NV21 | ✗ MaixCDK YOLO11 只支持 `rgb`/`bgr` | — | — | 工具链不支持 |

### 关键实验证据

**实验 1**: NV21 相机 + 模型 `_raw=True`（2026-07-29）
```
RuntimeError: input_type: RGB888, image format: YVU420SP
```
→ MaixCDK SDK **不做**自动格式转换，格式不匹配直接抛异常。

**实验 2**: `to_bytes()` 在 NV21 上（2026-07-29）
```python
cam = camera.Camera(640, 640, FMT_YVU420SP)
img = cam.read()
data = img.to_bytes()   # → OK, len=614400 (640×640×1.5)
```
→ `to_bytes()` 返回了正确的 NV21 字节数据。之前的 `[E] convert format failed, not support format 8` **仅是 stderr 噪音**，不影响数据完整性。

**实验 3**: RTSP `bind_camera` 格式检查（2026-07-29）
```
bind camera failed! support FMT_YVU420SP only!
```
→ 固件硬限制，RGB888 相机无法绑定到 RTSP 服务。

**实验 4**: MaixCDK `input_type` 枚举检查
```cpp
// F:\MaixPy-main\components\maix\...\maix_nn_yolo11.hpp
if (input_type == "rgb")
    _input_img_fmt = FMT_RGB888;
else if (input_type == "bgr")
    _input_img_fmt = FMT_BGR888;
else
    log::error("unknown input type: %s");
```
→ 仅支持 `rgb`/`bgr` 两种。`nv21` 不是合法值。

---

## 2. 最优解：RGB888 + HTTP JPEG

三条路口全堵死后，唯一可行路线：

```
RGB888 Camera (FMT_RGB888, 640×640)
  │
  ├── _last_raw (maix Image) ──→ _draw_overlays() ──→ _display_frame
  │                                    │                    │
  │                          ┌────────┤          ┌────────┘
  │                          ▼        │          ▼
  ├── image2cv(ensure_bgr=False)      │   display.show()  ──→ LCD + IDE预览
  │     [:,:,::-1] → BGR numpy       │   (daemon 线程, 帧跳 1/2)
  │                          │        │
  │     AI detect(_raw=True) │        │
  │     零拷贝 RGB888 直通   │        │
  │     NPU 推理 ~12ms       │        │
  │                          │        │
  └──────────────────────────┼────────┘
                             ▼
                     push_frame img
                       deque(maxlen=1)
                             │
                             ▼
                  _send_loop (daemon)
                  JpegStreamer.write()
                  VPU JPEG 硬件编码
                  HTTP MJPEG :8000/stream
                  (帧跳 1/4, 省 CPU)
```

### 为什么是 JPEG

| 推流方式 | 能否接收标注帧 | 相机格式约束 | 适用 |
|------|:---:|------|:---:|
| `rtsp.Rtsp.bind_camera()` | **否** (只绑 Camera) | NV21 only | ✗ |
| `http.JpegStreamer.write()` | **是** (推任意 Image) | 任意格式 | ✓ |
| `uvc` (USB) | 是 | USB 线缆 | 非 WiFi |

### 性能参数

| 指标 | 值 | 说明 |
|------|:---:|------|
| 管线 FPS | **60** | 优化后 |
| AI 推理 | ~12ms (NPU) | `_raw=True` 零拷贝 |
| BGR 转换 | ~1ms | `image2cv` memcpy + `[::-1]` view |
| IDE 预览 | ~2ms (VPU JPEG q=10) + 10ms GIL 持 |
| HTTP 推流 | ~8ms (VPU JPEG) | 帧跳后均摊 ~2ms |
| 线程数 | 4 | 主循环 + 管线 + 显示 daemon + 推流 daemon |
| 推流 URL | `http://<ip>:8000/stream` | 浏览器/VLC 直连 |
| RTSP 备用 | 不可用 | — |

---

## 3. FPS 优化历程

| 阶段 | 改动 | FPS | 关键瓶颈 |
|------|------|:---:|------|
| 基线 | 无推流 | 35 | `display.show()` 在主循环持 GIL 10ms |
| 分离线程 | show() → daemon | 35 | GIL 仍序列化 |
| 帧跳 show | 每 2 帧调一次 show | 40 | `MAIN_LOOP_DELAY=16ms` |
| **降 DELAY** | 16ms→2ms | **60** | 接近理论峰值 |
| 降 IDE 画质 | `set_trans_image_quality(10)` | 60 | 稳定 |
| 帧跳推流 | 每 4 帧推到 HTTP | 60 | 稳定 |
| 队列压缩 | `maxlen=2→1` | 60 | 省 CMM |

### GIL 瓶颈详解

Python GIL 序列化了三个线程的 JPEF 编码。`display.show()` 和 `JpegStreamer.write()` 在 VPU 编码期间**全程持 GIL**。帧跳（display 1/2, stream 1/4）使得大部分帧不触发 GIL 争抢。

```
时间轴 (优化后, 60 FPS):
管线:    [AI:12ms]──[draw:0.5ms]──[AI:12ms]──[draw:0.5ms]──[AI:12ms]
显示:                    [show:10ms GIL]                           [show:10ms]
推流:                                                           [write:8ms]
        ←────────── 27ms, 含 2 帧管线 ──────────→
```

---

## 4. 代码架构

### 线程模型

| 线程 | 线程名 | 职责 | 阻塞操作 |
|------|------|------|:---:|
| **1\. 主循环** | `main_loop` | coordinator.loop() + touch_handler + WDT feed | `time.sleep(0.002)` |
| **2\. 视觉管线** | `vision_processing` | `process_all()` → AI detect → `_update_display_frame()` | NPU (GIL 释放) |
| **3\. 显示 daemon** | `_display_loop` | `get_display_frame()` → `display.show()` | VPU JPEG (GIL 持) |
| **4\. 推流 daemon** | `_run` (JpegStreamer) | `dequeue` → `write()` HTTP MJPEG | VPU JPEG (GIL 持) |

### 关键文件

| 文件 | 角色 |
|------|------|
| `framework/hal/platforms/maixcam2/camera.py` | `FMT_RGB888` 相机, `last_raw` 属性, `image2cv` BGR 转换 |
| `modules/zw_opencv_module/vision_manager.py` | 管线线程, `_update_display_frame()`, `_capture_sink()` 帧跳推流 |
| `modules/zw_opencv_module/processors/ai_inference_processor.py` | `_raw=True` 零拷贝 AI 推理 |
| `modules/zw_wifi_stream/streamer.py` | `JpegStreamer`: async queue + HTTP JPEG MJPEG 推流 |
| `app/main.py` | 入口: `_init_streamer()`, `_build_callbacks()`, touch/display 线程管理 |
| `framework/module_manager.py` | `MAIN_LOOP_DELAY = 0.002` |

### frame-skip 策略

| 组件 | 常量 | 值 | 原因 |
|------|------|:---:|------|
| 显示 LCD + IDE | `_DISPLAY_EVERY_N` | 2 | VPU JPEG 编码持 GIL ~10ms |
| 推流 HTTP MJPEG | `_CAPTURE_EVERY_N` | 4 | VPU JPEG 编码持 GIL ~8ms |
| 推流队列 | `_QUEUE_MAX` | 1 | 仅缓存最新帧 |

---

## 5. 资源使用排查

### VPU

VPU 是独立硬件，JPEG 编码过程 **不消耗 CPU 算力**。但 C++ 绑定在等待 VPU 完成时**持 GIL 不释放**，阻塞其他 Python 线程。

### CPU 消耗来源

| 操作 | CPU | 频率 |
|------|:---:|:---:|
| `image2cv` memcpy 1.2MB | ~1ms/帧 | 每帧 |
| `context switch` 4 线程间 | ~0.1ms | 持续 |
| `coordinator.loop()` Python 逻辑 | ~0.2ms | 500Hz |
| `touch.read()` 硬件读取 | ~0.1ms | 500Hz |

### 内存

| 对象 | 大小 |
|------|:---:|
| RGB888 相机帧 (DMA 缓冲) | 640×640×3 = **1.2MB** ×3 buff = 3.6MB CMM |
| `image2cv` numpy 副本 | 1.2MB (堆) |
| LCD framebuffer | 480×640×4 = 1.2MB |
| HTTP server + JPEG buffer | ~2MB CMM |
| YOLO11n 模型 CMM | ~20MB CMM |
| **总计估算** | **~30MB / 256MB CMM** (12% 占用) |

---

## 6. 已探索但不可行的替代方案

### RTSP + NV21 相机 + 模型 NV21
- 前提：模型 `input_type=nv21`
- 验证：MaixCDK YOLO11 只支持 `rgb`/`bgr`，**不可行**

### NV21 相机 + SDK 自动 NV21→RGB
- 前提：`model.detect(NV21_image)` 正常推理
- 验证：SDK 不自动转换，直接抛 `RuntimeError`，**不可行**

### RGB888 → cv2 NV21 → RTSP
- 前提：RTSP 有 `push_frame` API
- 验证：`rtsp.Rtsp` 无任何手动帧推送接口，**不可行**

### NV21 相机 + `to_jpeg` + `imdecode` → BGR
- 验证：`to_jpeg(95)` VPU 编码 ~4ms + `imdecode` ~2ms = 6ms/帧
- 可行但 FPS 仅 ~10+，**被否决**

### display.show() → send_to_maixvision() 截获
- Maix Vision IDE 走私有 CommProtocol 协议，无 HTTP 端口
- 无法被外部程序拦截，**不可行**

---

## 7. 经验总结

1. **三色约束无法调和**：单 Camera 对象 = 单格式。设计之初应确认所有下游消费者（RTSP、模型、显示）能否接受同一格式。
2. **MaixCDK 的格式支持极有限**：YOLO11 模型仅 `rgb`/`bgr`，无 `nv21`。与 RTSP 的 NV21-only 约束结合形成死局。
3. **GIL 是 Python 多线程的天花板**：`display.show()` 和 `JpegStreamer.write()` 在 C++ 编码期间**全程持 GIL**。帧跳是唯一低成本缓解手段。
4. **`MAIN_LOOP_DELAY` 是隐蔽瓶颈**：从 16ms 降至 2ms 收益 15 FPS。
5. **跨模块私有属性访问 (`_attr`) 脆弱**：公开 property 化 (`last_raw`) 后大幅减少耦合。
6. **JPEG vs H.264 对 CPU 差异不大**：两者都走 VPU 硬件编码，真正的开销在 Python→C++ 调用路径的 GIL 持。#

