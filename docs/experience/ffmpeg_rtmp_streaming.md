# FFmpeg RTMP 推流开发经验

## 问题概述

在 Orange Pi 5B 上实现 OpenCV 视频流 RTMP 推流功能时，遇到多个关键问题，涉及 asyncio 事件循环管理和 FFmpeg 命令参数顺序。

---

## 问题 1：asyncio 事件循环冲突

### 现象
```
RuntimeError: Event loop is closed
Error in push_frame_sync: Event loop is closed
```

### 根本原因
- `asyncio.run()` 每次调用都会创建新的事件循环并在结束时关闭它
- 高频调用（60Hz）导致事件循环反复创建/销毁
- `asyncio.subprocess.Process` 与创建它的事件循环绑定，不能跨循环使用

### 错误代码模式
```python
def push_frame_sync(self, frame):
    loop = asyncio.get_event_loop()  # 创建未运行的循环
    if loop.is_running():
        ...
    else:
        return asyncio.run(self.push_frame(frame))  # 报错！已有循环
```

### 解决方案：后台常驻事件循环

```python
import threading

class FFmpegPusher:
    def __init__(self, ...):
        self._loop = None
        self._loop_thread = None
        self._loop_lock = threading.Lock()

    def _get_or_create_loop(self):
        """获取或创建后台事件循环（线程安全）"""
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._run_loop,
                    args=(self._loop,),
                    daemon=True
                )
                self._loop_thread.start()
            return self._loop

    def _run_loop(self, loop):
        """在后台线程中运行事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def push_frame_sync(self, frame):
        """同步推送帧（使用后台事件循环）"""
        loop = self._get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.push_frame(frame), loop
        )
        return future.result(timeout=5.0)
```

### 关键要点
1. **所有异步操作必须在同一个事件循环中执行** - 包括 `start()`、`push_frame()`、`close()`
2. **使用 `asyncio.run_coroutine_threadsafe()`** - 从同步代码提交任务到运行中的循环
3. **后台线程运行 `loop.run_forever()`** - 保持循环持续运行
4. **使用线程锁保护循环创建** - 避免并发创建多个循环

---

## 问题 2：FFmpeg 命令参数顺序错误

### 现象
FFmpeg 推流命令执行成功，但使用 `flv1` 编码器而非预期的 `libx264`：
```
Stream #0:0 -> #0:0 (rawvideo (native) -> flv1 (flv))
```

### 根本原因
FFmpeg 命令参数顺序错误，编码器参数被添加到输出 URL 之后，被视为"尾随选项"而被忽略。

### 错误代码
```python
command = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    ...
    '-i', '-',
    '-f', 'flv',
    '-flvflags', 'no_duration_filesize',
    self.rtmp_url  # URL 在这里
]
# 编码器参数在 URL 之后添加！
command.extend(self._get_software_encoder_params())
```

### 正确顺序
FFmpeg 命令必须遵循：**输入参数 → 编码器参数 → 输出格式 → URL**

```python
command = ['ffmpeg', '-y']

# 1. 输入参数
command.extend([
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{self.width}x{self.height}',
    '-r', str(self.fps),
    '-i', '-',
])

# 2. 编码器参数（必须在输出格式之前）
command.extend(self._get_software_encoder_params())

# 3. 输出格式和 URL
command.extend([
    '-f', 'flv',
    '-flvflags', 'no_duration_filesize',
    self.rtmp_url
])
```

### FFmpeg 参数顺序规则
```
ffmpeg [全局选项] [输入选项] -i 输入 [输出选项] 输出
```

| 顺序 | 参数类型 | 示例 |
|------|----------|------|
| 1 | 输入参数 | `-f rawvideo -pix_fmt bgr24 -s 1280x720 -r 30 -i -` |
| 2 | 编码器参数 | `-c:v libx264 -preset ultrafast -tune zerolatency` |
| 3 | 输出格式 | `-f flv -flvflags no_duration_filesize` |
| 4 | 输出 URL | `rtmp://localhost/live/stream` |

---

## 问题 3：FFmpeg 输出读取阻塞

### 现象
FFmpeg 输出读取使用 `read()` 方法，阻塞到流关闭，无法实时获取错误信息。

### 解决方案
使用 `readline()` 逐行读取：

```python
async def read_stream(stream, prefix):
    while not self._stopped:
        try:
            line = await stream.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='ignore').strip()
            if text:
                print(f"[{prefix}] {text}")
        except asyncio.CancelledError:
            break
```

---

## 问题 4：硬件编码器回退机制

### 经验
Orange Pi 5B 的 `h264_rkmpp` 硬件编码器可能不稳定，需要实现自动回退到软件编码（`libx264`）。

### 实现要点
1. **检测编码错误** - 监控 FFmpeg stderr 输出
2. **标记回退状态** - 避免无限重试
3. **重启 FFmpeg 进程** - 使用软件编码参数重新启动

---

## 调试技巧

### 1. 实时查看 FFmpeg 输出
```python
# 在 stderr 输出中添加前缀区分
print(f"[FFmpeg-err] {text}")
```

### 2. 验证编码器选择
检查 FFmpeg 输出中的 Stream mapping：
```
# 正确
Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (libx264))

# 错误（使用了默认 flv1 编码器）
Stream #0:0 -> #0:0 (rawvideo (native) -> flv1 (flv))
```

### 3. 检查进程状态
```python
if self.process.returncode is not None:
    print(f"FFmpeg exited with code {self.process.returncode}")
```

---

## 关键代码检查清单

- [ ] 所有异步操作使用同一个事件循环
- [ ] 高频调用场景使用后台常驻事件循环
- [ ] FFmpeg 命令参数顺序：输入 → 编码器 → 输出格式 → URL
- [ ] 使用 `readline()` 而非 `read()` 读取子进程输出
- [ ] 添加超时机制避免无限等待
- [ ] 实现硬件编码失败时的软件编码回退

---

## 相关文件

- `modules/zw_opencv_module/ffmpeg_pusher.py` - FFmpeg 推流核心实现
- `modules/zw_opencv_module/camera_manager.py` - 相机管理和推流调用

---

*记录时间：2026-04-06*
*平台：Orange Pi 5B (ARM64)*
*FFmpeg 版本：6.1*
