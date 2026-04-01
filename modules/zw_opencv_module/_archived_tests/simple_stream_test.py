#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的RTMP推流测试，用于诊断硬件编码器问题
"""

import subprocess
import sys
import time

def test_ffmpeg_command(cmd, description):
    """测试FFmpeg命令"""
    print(f"\n测试: {description}")
    print(f"命令: {' '.join(cmd)}")

    try:
        # 运行命令5秒
        import signal

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 等待一会儿让FFmpeg启动
        time.sleep(1)

        # 检查进程是否还在运行
        if process.poll() is not None:
            # 进程已经结束，读取错误输出
            stdout, stderr = process.communicate()
            print(f"✗ 进程立即退出，返回码: {process.returncode}")
            print(f"错误输出:\n{stderr.decode('utf-8', errors='ignore')[:500]}")
            return False

        # 进程还在运行，尝试发送一些数据
        print("✓ 进程启动成功，测试通过")

        # 终止进程
        process.terminate()
        process.wait(timeout=2)

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_hardware_vs_software():
    """对比硬件和软件编码器"""
    print("RTMP推流编码器对比测试")
    print("=" * 60)

    # 测试不同编码器
    tests = [
        {
            "name": "硬件编码 (h264_v4l2m2m)",
            "cmd": [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', 'testsrc=size=640x480:rate=30',
                '-t', '3',  # 运行3秒
                '-c:v', 'h264_v4l2m2m',
                '-pix_fmt', 'yuv420p',
                '-b:v', '1000k',
                '-f', 'flv',
                'rtmp://localhost/live/stream'
            ]
        },
        {
            "name": "软件编码 (libx264)",
            "cmd": [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', 'testsrc=size=640x480:rate=30',
                '-t', '3',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-f', 'flv',
                'rtmp://localhost/live/stream'
            ]
        },
        {
            "name": "硬件编码 (低分辨率)",
            "cmd": [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', 'testsrc=size=320x240:rate=15',
                '-t', '3',
                '-c:v', 'h264_v4l2m2m',
                '-pix_fmt', 'yuv420p',
                '-b:v', '500k',
                '-f', 'flv',
                'rtmp://localhost/live/stream'
            ]
        },
        {
            "name": "硬件编码 (无音频)",
            "cmd": [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', 'testsrc=size=640x480:rate=30',
                '-t', '3',
                '-c:v', 'h264_v4l2m2m',
                '-pix_fmt', 'yuv420p',
                '-b:v', '1000k',
                '-an',  # 无音频
                '-f', 'flv',
                'rtmp://localhost/live/stream'
            ]
        }
    ]

    results = {}

    for test in tests:
        success = test_ffmpeg_command(test["cmd"], test["name"])
        results[test["name"]] = success

        # 等待一下避免端口冲突
        time.sleep(1)

    print("\n" + "=" * 60)
    print("测试结果总结:")
    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")

    return results

def check_rtmp_server():
    """检查RTMP服务器"""
    print("\n检查RTMP服务器...")

    # 检查1935端口
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 1935))

        if result == 0:
            print("✓ RTMP服务器端口 1935 可访问")
            return True
        else:
            print("✗ RTMP服务器端口 1935 不可访问")
            print("  请确保RTMP服务器正在运行:")
            print("  docker run -d -p 1935:1935 --name nginx-rtmp tiangolo/nginx-rtmp")
            return False
    except Exception as e:
        print(f"✗ 检查RTMP服务器时出错: {e}")
        return False

def main():
    """主函数"""
    print("简单RTMP推流测试")
    print("=" * 60)

    # 检查FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg可用")
            # 提取版本信息
            lines = result.stdout.split('\n')
            if lines:
                print(f"  版本: {lines[0]}")
        else:
            print("✗ FFmpeg不可用")
            return 1
    except:
        print("✗ FFmpeg不可用")
        return 1

    # 检查RTMP服务器
    if not check_rtmp_server():
        print("\n警告: RTMP服务器可能未运行，但继续测试...")

    # 运行编码器测试
    results = test_hardware_vs_software()

    print("\n" + "=" * 60)
    print("建议:")

    hardware_worked = any("硬件编码" in name and success
                         for name, success in results.items())

    if hardware_worked:
        print("1. 硬件编码器可用，但可能需要特定参数")
        print("2. 尝试在qr_streaming_demo.py中使用: --hardware-accel")
    else:
        print("1. 硬件编码器可能有问题，建议使用软件编码")
        print("2. 在qr_streaming_demo.py中不要使用--hardware-accel参数")

    print("3. 如果软件编码也失败，检查RTMP服务器")

    return 0

if __name__ == '__main__':
    sys.exit(main())