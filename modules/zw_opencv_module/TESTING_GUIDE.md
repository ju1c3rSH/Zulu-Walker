# QR码识别与RTMP推流融合测试指南

## 概述

本项目提供了QR码识别与RTMP推流的融合测试脚本，针对香橙派5B等嵌入式设备优化了延迟性能。

## 文件说明

### 主要脚本
1. **`qr_streaming_demo.py`** - 主测试脚本，集成QR识别、推流和性能监控
2. **`check_hardware.py`** - 硬件诊断工具，检查摄像头和编码器支持
3. **`camera_tasks.py`** - 新增了`process_frame_for_qr()`方法，支持外部帧处理

### 优化特性
- **延迟优化**: 摄像头缓冲区设置、FFmpeg零延迟参数、动态帧率控制
- **QR识别**: 可配置检测频率，避免每帧检测造成的CPU负担
- **调试信息**: 实时显示FPS、延迟、QR识别结果等信息
- **硬件加速**: 支持香橙派的h264_v4l2m2m硬件编码器

## 安装依赖

```bash
pip install opencv-python opencv-contrib-python numpy
```

确保FFmpeg已安装：
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# 检查安装
ffmpeg -version
```

## 使用步骤

### 1. 硬件诊断
运行硬件检查脚本，确认系统支持：

```bash
python check_hardware.py
```

检查输出中的关键信息：
- 可用的H.264编码器（特别是硬件编码器）
- 摄像头缓冲区是否可设置为1
- 摄像头读取延迟

### 2. 启动RTMP服务器
需要本地RTMP服务器。使用nginx-rtmp或类似工具：

```bash
# 使用Docker运行nginx-rtmp
docker run -d -p 1935:1935 --name nginx-rtmp tiangolo/nginx-rtmp

# 推流地址: rtmp://localhost/live/stream
# 播放地址: rtmp://localhost/live/stream
```

### 3. 运行融合测试

#### 基本使用（软件编码）：
```bash
python qr_streaming_demo.py --rtmp rtmp://localhost/live/stream
```

#### 启用硬件加速（香橙派）：
```bash
python qr_streaming_demo.py --rtmp rtmp://localhost/live/stream --hardware-accel
```

#### 调整参数：
```bash
python qr_streaming_demo.py \
  --rtmp rtmp://localhost/live/stream \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --qr-freq 2 \
  --hardware-accel
```

参数说明：
- `--rtmp`: RTMP服务器地址（默认: rtmp://localhost/live/stream）
- `--width`: 视频宽度（默认: 640）
- `--height`: 视频高度（默认: 480）
- `--fps`: 目标帧率（默认: 30）
- `--camera`: 摄像头索引（默认: 0）
- `--hardware-accel`: 启用硬件加速编码
- `--qr-freq`: QR检测频率，每N帧检测一次（默认: 1）
- `--check-only`: 仅检查摄像头和FFmpeg，不实际运行

### 4. 测试QR识别
在摄像头前展示QR码，观察：
1. 画面中的绿色QR码边框
2. 左上角的QR码内容显示
3. 调试信息中的QR检测状态

### 5. 查看性能统计
停止推流（Ctrl+C）后，脚本会显示性能统计：
- 总帧数和平均帧率
- 各环节处理时间（QR检测、推流等）
- 估计的总延迟

## 延迟优化要点

### 摄像头设置
- 缓冲区大小设置为1帧（`CAP_PROP_BUFFERSIZE=1`）
- 使用MJPEG编码格式（如果摄像头支持）
- 队列大小设置为2，避免帧堆积

### FFmpeg参数
- `-preset ultrafast -tune zerolatency`: 最快编码速度，零延迟调优
- `-g $fps`: GOP大小等于帧率，减少关键帧间隔
- `-bufsize 1000k -maxrate 2000k`: 控制缓冲区大小和最大码率
- 硬件编码器: `h264_v4l2m2m`（香橙派）

### 软件优化
- 动态帧率控制：根据实际处理时间调整等待
- 可配置的QR检测频率：避免每帧检测
- 异步I/O：不阻塞主循环

## 故障排除

### 1. 摄像头无法打开
```bash
# 检查摄像头索引
python check_hardware.py

# 尝试不同索引
python qr_streaming_demo.py --camera 1
```

### 2. FFmpeg错误
```bash
# 检查FFmpeg安装
ffmpeg -version

# 检查RTMP服务器
curl -I http://localhost:1935/
```

### 3. 高延迟
- 降低分辨率：`--width 640 --height 480`
- 降低帧率：`--fps 15`
- 增加QR检测间隔：`--qr-freq 3`
- 启用硬件加速：`--hardware-accel`

### 4. QR识别失败
- 确保安装opencv-contrib-python：`pip install opencv-contrib-python`
- 调整摄像头对焦和光照
- 使用清晰的QR码图片

## 性能预期

| 配置 | 预期延迟 | 备注 |
|------|----------|------|
| 软件编码 640x480@30fps | 100-200ms | 常规配置 |
| 硬件编码 640x480@30fps | 50-100ms | 香橙派优化 |
| 软件编码 1280x720@15fps | 150-250ms | 高清但低帧率 |
| 硬件编码 1280x720@30fps | 80-150ms | 最佳平衡 |

## 下一步改进

1. **多线程处理**: 分离QR识别和推流到不同线程
2. **网络自适应**: 根据网络状况调整码率和分辨率
3. **硬件解码**: 使用硬件解码器进一步降低延迟
4. **Web界面**: 提供网页控制界面和状态监控

## 参考链接
- [OpenCV QR码检测文档](https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html)
- [FFmpeg编码指南](https://trac.ffmpeg.org/wiki/Encode/H.264)
- [nginx-rtmp配置](https://github.com/arut/nginx-rtmp-module)