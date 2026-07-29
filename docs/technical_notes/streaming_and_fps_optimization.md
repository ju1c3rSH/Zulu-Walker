# 推流与 FPS 优化 — 技术踩坑笔记

> 面向 MaixCAM2 (Axera AX630C) + MaixPy 固件 + YOLO11n 检测

---

## 1. 格式三角难题

### 三个约束

| 组件 | 强制格式 | 来源 |
|------|---------|------|
| RTSP `bind_camera` | **FMT_YVU420SP (NV21) only** | 固件运行时错误 `bind camera failed! support FMT_YVU420SP only!` |
| AI 模型 (YOLO11n) | **FMT_RGB888** | `model.input_format()` 返回值 |
| 显示/LCD `display.show()` | 两者均可 | 官方示例验证 |

### 结论

单相机格式无法同时满足 RTSP (NV21) + 模型 (RGB888)。**必须二选一**，另一端做软件转换。选择 RGB888：(a) 模型零拷贝推理，(b) 推流改用 `http.JpegStreamer`（硬件编码器内部处理 RGB→YUV）。

---

## 2. NV21 格式转换陷阱

### 固件限制

`maix.image.Image.to_format()` **不支持源格式 FMT_YVU420SP (format=8)**。任何 `to_format(FMT_RGB888)` 或 `to_format(FMT_BGR888)` 调用都会触发 C++ 错误：

```
-- [E] convert format failed, not support format 8
```

### 哪些操作会触发

| 操作 | 会触发? | 说明 |
|------|:---:|------|
| `img.to_format(FMT_RGB888)` | **会** | 软件格式转换，NV21→RGB 未实现 |
| `img.to_format(FMT_BGR888)` | **会** | 同上 |
| `img.to_bytes()` | **会** | 内部疑似调用了 to_format |
| `image2cv(img, ensure_bgr=True)` | **会** | 内部调 `to_format(BGR888)` |
| `img.to_jpeg(quality)` | **不会** | 走 VPU 硬件 JPEG 编码器，NV21 为原生输入 |
| `image2cv(img, ensure_bgr=False)` | **不会** | 纯 memcpy，不调用 to_format |
| `display.show(NV21_img)` | **不会** | 走硬件 VO 控制器 |
| `http.JpegStreamer.write(NV21_img)` | **不会** | 走 VPU 硬件编码 |

### 可行绕过

**NV21 → BGR 通过 JPEG 中介**：

```python
jpg_img = nv21_img.to_jpeg(quality=95)           # VPU 硬件编码，NV21 原生输入
jpg_bytes = jpg_img.to_bytes()                    # JPEG 字节
bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)  # OpenCV 解码
```

代价：每帧 ~6ms（编码 ~4ms + 解码 ~2ms），FPS 从 35 降至 10+。

---

## 3. RGB888 相机 — 最终选择

模型 `input_format = FMT_RGB888`，相机切为同格式后所有路径均走硬件：

```python
# camera.py
self._cam = maix.camera.Camera(..., format=maix.image.Format.FMT_RGB888)

# read_raw() — AI 推理用 BGR numpy
np_img = maix.image.image2cv(img, ensure_bgr=False, copy=True)  # 纯 memcpy ~1ms
self._last_frame = np_img[:, :, ::-1]                            # BGR view 零拷贝

# AI 推理 — 零拷贝直通模型
self._ai.detect(raw, _raw=True)   # 相机 Image 直传，格式一致无需转换
```

`_raw=True` 路径只对 RGB888 安全（格式与模型匹配）。**NV21 相机下切勿使用**，否则 `model.detect()` 内部触发 to_format 崩溃。

---

## 4. 推流方案演变

### RTSP → HTTP JPEG

| | RTSP | HTTP JPEG (JpegStreamer) |
|------|------|------|
| 接入方式 | `bind_camera(cam)` | `stream.write(img)` 手动推送 |
| 格式要求 | **NV21 only** | 任意格式（VPU 编码器内转换） |
| 标注帧支持 | **不支持** (只推原始相机帧) | **支持** (推送已绘制的 `_display_frame`) |
| 客户端 | VLC/ffplay `rtsp://` | 浏览器 `http://<ip>:8000/stream` |
| 编码 | H.264 硬件 | JPEG 硬件 (VPU) |

### 最终方案：HTTP JPEG

```python
# streamer.py — JpegStreamer 异步推流
class JpegStreamer:
    def push_frame(self, img):
        with self._lock:
            self._queue.appendleft(img)       # 仅压队列，~0ms

    def _send_loop(self):                     # 后台线程消费
        while self._is_running:
            img = self._queue.pop()
            self._server.write(img)           # VPU JPEG 编码 + HTTP 发送
```

### 数据流

```
RGB888 Camera (640×640)
  │
  ├─ _last_raw (maix Image) ──→ _draw_overlays ──→ _display_frame
  │                                                        │
  │                                     ┌─────────────────┤
  │                                     ↓                  ↓
  ├─ image2cv → BGR numpy ──→ AI (_raw=True)    streamer.push_frame()
  │                                                           │
  └─ display.show() ←── _display_loop daemon               deque
                                                              │
                                                     _send_loop daemon
                                                     http.JpegStreamer.write()
```

---

## 5. GIL 序列化 — FPS 瓶颈的根本原因

### 问题

Python GIL 同一时刻只允许一个线程执行 Python 代码。`display.show()` 和 `JpegStreamer.write()` 在 C++ JPEG 编码期间**全程持 GIL**，阻塞管线线程。

### FPS 演变

| 阶段 | 架构 | FPS | 瓶颈 |
|------|------|:---:|------|
| 初始 | 无推流 | 35 | `display.show()` 在主循环持 GIL ~10ms |
| 分离线程 | display.show() 移入 daemon | 35 | GIL 依旧序列化 |
| 帧跳 show | 每 2 帧调一次 show | 43 | 隔帧释放但不稳定 |
| 帧跳 show+write | 两者都做帧跳 | 35 | `time.sleep(0.016)` 主循环 |
| **MAIN_LOOP_DELAY** | 16ms→2ms + 帧跳 | **60** | 接近瓶颈 |

### 关键发现

**`MAIN_LOOP_DELAY = 0.016`** (16ms) 是隐蔽瓶颈：

```python
# module_manager.py
MAIN_LOOP_DELAY = 0.002   # 优化后
```

`time.sleep(0.016)` 虽然释放 GIL，但主循环恢复后会立即调用 `display_callback`，触发 `display.show()` 再次持 GIL。缩短 sleep 让主循环更快完成迭代、更快释放 GIL。

### 最终架构

```
线程 1 — 管线 (vision_manager._process_loop)
  process_all() → AI 推理 (NPU, 12ms, GIL 释放) → draw overlays → time.sleep(0)

线程 2 — 显示 (_display_loop daemon)
  每 _DISPLAY_EVERY_N 帧: draw exit icon → display.show(frame) → time.sleep(0.001)

线程 3 — 推流 (_send_loop daemon)
  每 _CAPTURE_EVERY_N 帧: dequeue → JpegStreamer.write(img) → time.sleep(0.005)

主线程 — 控制 (run_main_loop)
  coordinator.loop() → touch_handler → time.sleep(0.002)
  不阻塞，不持 GIL 过久
```

**核心优化**：
1. `display.show()` 从主线程剥离到独立 daemon（避免主循环卡住）
2. 显示 + 推流都做帧跳（隔帧释放 GIL）
3. `MAIN_LOOP_DELAY` 从 16ms 降至 2ms
4. AI 推理走 `_raw=True` 零拷贝（BGR→RGB 转换、memcpy 全消除）
5. IDE 预览 JPEG 质量降为 20（硬件编码更快）

---

## 6. 关键代码片段

### camera.py — RGB888 格式 + 公开属性

```python
self._cam = maix.camera.Camera(..., format=maix.image.Format.FMT_RGB888)
self._last_raw: Optional[maix.image.Image] = None

@property
def last_raw(self):
    """The most recent raw maix Image frame (RGB888)."""
    return self._last_raw
```

### ai_inference_processor.py — 零拷贝推理

```python
raw = getattr(camera, "last_raw", None)
if raw is not None:
    detections = self._ai.detect(raw, _raw=True)   # RGB888 零拷贝直通
else:
    detections = self._ai.detect(frame)             # 回退 BGR numpy
```

### vision_manager.py — 帧跳推流

```python
_CAPTURE_EVERY_N = 2

self._capture_seq += 1
if self._capture_sink is not None and self._capture_seq % self._CAPTURE_EVERY_N == 0:
    self._capture_sink(raw_img)
```

### main.py — 统一初始化

```python
def _init_streamer(vm):
    s = get_streamer()
    if s and vm:
        vm.set_capture_sink(s.push_frame)
        s.start_async()
```

---

## 7. `add_channel` 的正确用法

`cam.add_channel(w, h)` 创建一个**同格式**的独立相机读取通道。用于多个消费者同时读帧（如 RTSP 推流 + 管线推理），**不能**用于格式转换。

```python
cam = camera.Camera(..., format=FMT_YVU420SP)
cam2 = cam.add_channel(disp_w, disp_h)  # 第二通道，同格式 NV21
server.bind_camera(cam)                  # RTSP 绑原始 cam
img = cam2.read()                        # 管线读通道
```

当不再使用 `bind_camera` 后（改用 HTTP JPEG 推流），不再需要 channel 分离。

---

## 8. 推流 URL 对照

| 方式 | URL | 客户端 |
|------|-----|--------|
| Maix Vision IDE | 自动 (USB RNDIS) | Maix Vision 软件 |
| RTSP `bind_camera` | `rtsp://<ip>:8554/live` | VLC, ffplay |
| HTTP JPEG `JpegStreamer` | `http://<ip>:8000/stream` | 浏览器, Python requests |
| `display.set_trans_image_quality(20)` | IDE 预览 JPEG 质量 | — (启动时一行) |

---

## 9. 经验总结

1. **相机格式必须匹配模型 input_format()** — 任何不匹配都会引入转换链，且固件对 NV21 的软件转换支持非常有限
2. **JPEG 硬件编码器是 NV21→JPEG 的安全通路** — `to_jpeg()` 不走 `to_format()`，不触发 format 8 错误
3. **GIL 是 Python 多线程的性能天花板** — `display.show()` / `JpegStreamer.write()` 等 C++ 调用持 GIL 时间需精确控制
4. **帧跳是低成本 GIL 优化手段** — 简单且有效，不引入额外同步开销
5. **`MAIN_LOOP_DELAY` 影响全局吞吐** — 看似无害的 sleep 可能是隐蔽的瓶颈
6. **跨模块私有属性访问 (`_attr`) 增加耦合** — 用 property 公开化，减少脆弱性
7. **推流方式应按需选择** — RTSP/H.264 适合高画质低码率，HTTP JPEG 适合标注帧推流；Maix Vision IDE USB 预览足以覆盖调试场景
