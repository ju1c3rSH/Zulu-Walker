#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查香橙派硬件编码器和摄像头设置的诊断脚本
"""

import subprocess
import sys
import cv2

def check_ffmpeg_encoders():
    """检查可用的FFmpeg编码器"""
    print("=" * 60)
    print("检查FFmpeg编码器")
    print("=" * 60)

    try:
        # 检查h264编码器
        result = subprocess.run(['ffmpeg', '-encoders'],
                              capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            print("错误：无法获取编码器列表")
            return

        # 查找h264编码器
        lines = result.stdout.split('\n')
        h264_encoders = []

        for line in lines:
            if 'h264' in line.lower():
                h264_encoders.append(line.strip())

        print(f"找到 {len(h264_encoders)} 个H.264编码器:")
        for encoder in h264_encoders:
            print(f"  {encoder}")

        # 检查硬件编码器
        hardware_encoders = []
        for encoder in h264_encoders:
            if 'v4l2m2m' in encoder or 'omx' in encoder or 'mmal' in encoder or 'vaapi' in encoder:
                hardware_encoders.append(encoder)

        if hardware_encoders:
            print(f"\n找到 {len(hardware_encoders)} 个硬件编码器:")
            for encoder in hardware_encoders:
                print(f"  {encoder}")
        else:
            print("\n未找到硬件编码器，将使用软件编码(libx264)")

    except FileNotFoundError:
        print("错误：未找到ffmpeg命令")
    except Exception as e:
        print(f"检查编码器时出错: {e}")

def check_camera_properties(camera_index=0):
    """检查摄像头属性"""
    print("\n" + "=" * 60)
    print(f"检查摄像头 {camera_index} 属性")
    print("=" * 60)

    try:
        # 尝试使用V4L2打开摄像头
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            backend = "V4L2"
        except:
            cap = cv2.VideoCapture(camera_index)
            backend = "默认"

        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return

        print(f"摄像头后端: {backend}")

        # 检查支持的属性
        properties = [
            (cv2.CAP_PROP_BUFFERSIZE, "CAP_PROP_BUFFERSIZE", "缓冲区大小"),
            (cv2.CAP_PROP_FPS, "CAP_PROP_FPS", "帧率"),
            (cv2.CAP_PROP_FRAME_WIDTH, "CAP_PROP_FRAME_WIDTH", "宽度"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "CAP_PROP_FRAME_HEIGHT", "高度"),
            (cv2.CAP_PROP_FOURCC, "CAP_PROP_FOURCC", "编码格式"),
            (cv2.CAP_PROP_AUTOFOCUS, "CAP_PROP_AUTOFOCUS", "自动对焦"),
            (cv2.CAP_PROP_BRIGHTNESS, "CAP_PROP_BRIGHTNESS", "亮度"),
            (cv2.CAP_PROP_CONTRAST, "CAP_PROP_CONTRAST", "对比度"),
        ]

        for prop_id, prop_name, prop_desc in properties:
            try:
                value = cap.get(prop_id)
                print(f"{prop_desc} ({prop_name}): {value}")
            except:
                print(f"{prop_desc} ({prop_name}): 不支持")

        # 测试设置缓冲区大小
        print("\n测试设置缓冲区大小:")
        try:
            # 先获取当前值
            current = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            print(f"  当前缓冲区大小: {current}")

            # 尝试设置为1
            success = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  设置为1: {'成功' if success else '失败'}")

            if success:
                new_value = cap.get(cv2.CAP_PROP_BUFFERSIZE)
                print(f"  新缓冲区大小: {new_value}")
        except Exception as e:
            print(f"  测试缓冲区设置时出错: {e}")

        # 测试读取几帧以检查延迟
        print("\n测试帧读取延迟:")
        import time

        frame_count = 10
        delays = []

        for i in range(frame_count):
            start = time.time()
            ret, frame = cap.read()
            end = time.time()

            if ret:
                delay = (end - start) * 1000  # 转换为毫秒
                delays.append(delay)
                print(f"  第{i+1}帧: {delay:.1f}ms, 尺寸: {frame.shape}")
            else:
                print(f"  第{i+1}帧: 读取失败")
                break

        if delays:
            avg_delay = sum(delays) / len(delays)
            print(f"\n平均帧读取延迟: {avg_delay:.1f}ms")
            print(f"理论最大FPS: {1000/avg_delay:.1f} (假设无处理延迟)")

        cap.release()

    except Exception as e:
        print(f"检查摄像头属性时出错: {e}")

def check_system_info():
    """检查系统信息"""
    print("\n" + "=" * 60)
    print("检查系统信息")
    print("=" * 60)

    try:
        # 检查操作系统
        import platform
        print(f"系统: {platform.system()} {platform.release()}")
        print(f"机器: {platform.machine()}")
        print(f"处理器: {platform.processor()}")

        # 检查Python和OpenCV版本
        print(f"Python版本: {sys.version}")
        print(f"OpenCV版本: {cv2.__version__}")

        # 检查可用摄像头数量
        print("\n检查可用摄像头:")
        for i in range(5):  # 检查前5个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  摄像头 {i}: 可用")
                cap.release()
            else:
                print(f"  摄像头 {i}: 不可用")

    except Exception as e:
        print(f"检查系统信息时出错: {e}")

def main():
    """主函数"""
    print("香橙派硬件诊断工具")
    print("=" * 60)

    # 检查系统信息
    check_system_info()

    # 检查FFmpeg编码器
    check_ffmpeg_encoders()

    # 检查摄像头属性
    check_camera_properties(0)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

    print("\n建议:")
    print("1. 如果找到 h264_v4l2m2m 编码器，可以在推流时使用 --hardware-accel")
    print("2. 如果摄像头缓冲区大小可设置为1，将有助于降低延迟")
    print("3. 如果平均帧读取延迟 > 33ms (30fps)，考虑降低分辨率")

if __name__ == '__main__':
    main()