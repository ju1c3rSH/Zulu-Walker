#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试香橙派硬件编码器支持的参数
"""

import subprocess
import sys
import time

def test_hardware_encoder(encoder_name="h264_rkmpp", width=640, height=480, fps=30):
    """测试硬件编码器是否支持特定参数"""
    print(f"测试硬件编码器: {encoder_name}")
    print(f"分辨率: {width}x{height}, 帧率: {fps}fps")
    print("=" * 60)

    # 测试1: 检查编码器是否存在
    print("\n1. 检查编码器支持...")
    try:
        result = subprocess.run(['ffmpeg', '-encoders'],
                              capture_output=True, text=True, encoding='utf-8')
        if encoder_name in result.stdout:
            print(f"✓ 找到编码器: {encoder_name}")
        else:
            print(f"✗ 未找到编码器: {encoder_name}")
            print("可用的H.264编码器:")
            for line in result.stdout.split('\n'):
                if 'h264' in line.lower():
                    print(f"  {line.strip()}")
            return False
    except Exception as e:
        print(f"✗ 检查编码器时出错: {e}")
        return False

    # 测试2: 测试简单编码
    print("\n2. 测试简单编码（生成测试图案）...")
    test_cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'testsrc=size={width}x{height}:rate={fps}',
        '-t', '2',  # 只运行2秒
        '-c:v', encoder_name,
        '-pix_fmt', 'nv12',
        '-f', 'null',
        '-'
    ]

    print(f"命令: {' '.join(test_cmd)}")

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✓ 简单编码测试通过")
        else:
            print("✗ 简单编码测试失败")
            print(f"错误输出:\n{result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"✗ 运行测试时出错: {e}")
        return False

    # 测试3: 测试不同分辨率
    print("\n3. 测试不同分辨率支持...")
    resolutions = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080)
    ]

    for w, h in resolutions:
        test_cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'testsrc=size={w}x{h}:rate=15',  # 降低帧率测试
            '-t', '1',  # 只运行1秒
            '-c:v', encoder_name,
            '-pix_fmt', 'nv12',
            '-f', 'null',
            '-'
        ]

        print(f"  测试 {w}x{h}: ", end='')
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                print("✓ 支持")
            else:
                print("✗ 不支持")
        except:
            print("✗ 错误")

    # 测试4: 测试不同比特率
    print("\n4. 测试不同比特率参数...")
    bitrates = ['500k', '1000k', '2000k', '4000k']

    for bitrate in bitrates:
        test_cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'testsrc=size=640x480:rate=15',
            '-t', '1',
            '-c:v', encoder_name,
            '-pix_fmt', 'nv12',
            '-b:v', bitrate,
            '-f', 'null',
            '-'
        ]

        print(f"  比特率 {bitrate}: ", end='')
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                print("✓ 支持")
            else:
                print("✗ 不支持")
        except:
            print("✗ 错误")

    # 测试5: 测试RTMP输出（无实际连接）
    print("\n5. 测试RTMP格式输出（模拟）...")
    test_cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'testsrc=size=640x480:rate=15',
        '-t', '1',
        '-c:v', encoder_name,
        '-pix_fmt', 'nv12',
        '-f', 'flv',
        '/dev/null'  # 输出到空设备
    ]

    print(f"命令: {' '.join(test_cmd[:10])}...")
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✓ RTMP格式输出测试通过")
        else:
            print("✗ RTMP格式输出测试失败")
            print(f"错误输出:\n{result.stderr[:500]}")
    except Exception as e:
        print(f"✗ 运行测试时出错: {e}")

    print("\n" + "=" * 60)
    print("硬件编码器测试完成")
    return True


def test_v4l2_codec_details():
    """测试V4L2编码器详细信息"""
    print("\n检查V4L2编码器详细信息...")

    # 检查V4L2设备
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("V4L2设备列表:")
            print(result.stdout[:1000])
        else:
            print("v4l2-ctl命令不可用")
    except:
        print("无法运行v4l2-ctl命令")

    # 检查编码器能力
    print("\n检查编码器能力（可能需要sudo）...")
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'encoder=h264_rkmpp'],
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("编码器h264_rkmpp支持:")
            lines = result.stdout.split('\n')
            for line in lines[:50]:  # 显示前50行
                if 'Supported' in line or 'pixel formats' in line or line.strip():
                    print(f"  {line}")
        else:
            print("无法获取编码器信息")
    except:
        print("无法获取编码器信息")


def main():
    """主函数"""
    print("香橙派硬件编码器兼容性测试")
    print("=" * 60)

    # 测试默认编码器
    encoder = "h264_rkmpp"

    # 检查是否有其他编码器
    try:
        result = subprocess.run(['ffmpeg', '-encoders'],
                              capture_output=True, text=True, encoding='utf-8')
        available_encoders = []
        for line in result.stdout.split('\n'):
            if 'h264' in line.lower():
                available_encoders.append(line.strip())

        if available_encoders:
            print(f"找到 {len(available_encoders)} 个H.264编码器:")
            for enc in available_encoders:
                print(f"  {enc}")

            # 优先选择硬件编码器
            hw_encoders = [enc for enc in available_encoders
                          if 'h264_rkmpp' in enc or 'omx' in enc or 'mmal' in enc]
            if hw_encoders:
                encoder = hw_encoders[0].split()[1]  # 获取编码器名称
                print(f"\n选择硬件编码器: {encoder}")
        else:
            print("未找到H.264编码器")
            return 1
    except:
        print("无法获取编码器列表")

    # 测试硬件编码器
    test_hardware_encoder(encoder, width=640, height=480, fps=30)

    # 测试V4L2详细信息
    if 'h264_rkmpp' in encoder:
        test_v4l2_codec_details()
    test_v4l2_codec_details()
    print("\n" + "=" * 60)
    print("建议:")
    print("1. 如果硬件编码器测试失败，使用软件编码 (--no-hardware-accel)")
    print("2. 分辨率建议: 640x480 或 800x600")
    print("3. 比特率建议: 1000k-2000k")
    print("4. 帧率建议: 15-30fps")

    return 0


if __name__ == '__main__':
    sys.exit(main())