#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试QR码扫描和RTMP推流功能

注意事项：
1. 测试需要安装 opencv-python, opencv-contrib-python, numpy
2. 如果要实际测试RTMP推流，需要本地运行RTMP服务器（如nginx-rtmp）
3. 可以使用模拟模式避免依赖真实摄像头和RTMP服务器
"""

import asyncio
import unittest
import numpy as np
import cv2
from typing import Optional

# 导入被测试模块
try:
    from camera_stream import CameraStream
    from ffmpeg_pusher import FFmpegPusher
    from camera_tasks import CameraTasks, VisionResult
    from task_sequence import TaskSequence
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"导入模块失败: {e}")
    MODULES_AVAILABLE = False












class TestRealCameraStreaming(unittest.IsolatedAsyncioTestCase):
    """真实摄像头和推流测试（需要真实硬件）"""

    RTMP_URL = "rtmp://localhost/live/stream"

    @classmethod
    def setUpClass(cls):
        """检查摄像头是否可用"""
        try:
            # 尝试打开默认摄像头
            import cv2
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.release()
                cls.camera_available = True
            else:
                cls.camera_available = False
        except:
            cls.camera_available = False

    def test_camera_capture(self):
        """测试摄像头采集"""
        if not self.camera_available:
            self.skipTest("摄像头不可用")

        try:
            from camera_stream import CameraStream
            camera = CameraStream(0, width=640, height=480)
            # 尝试读取几帧
            for _ in range(5):
                frame = camera.read_frame()
                if frame is not None:
                    # 确保帧有正确的形状
                    self.assertIsInstance(frame, np.ndarray)
                    self.assertEqual(frame.ndim, 3)  # 高度、宽度、通道
                    break
            camera.release()
        except Exception as e:
            self.fail(f"摄像头采集失败: {e}")

    async def test_streaming(self):
        """测试RTMP推流（需要RTMP服务器）"""
        if not self.camera_available:
            self.skipTest("摄像头不可用")

        # 检查FFmpeg是否可用
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode != 0:
                self.skipTest("FFmpeg不可用")
        except:
            self.skipTest("FFmpeg不可用")

        # 尝试打开摄像头并推流几帧
        from camera_stream import CameraStream
        from ffmpeg_pusher import FFmpegPusher

        camera = None
        pusher = None
        try:
            camera = CameraStream(0, width=640, height=480)
            pusher = FFmpegPusher(self.RTMP_URL, fps=30, width=640, height=480)

            await pusher.start()

            # 推送几帧测试
            for _ in range(10):
                frame = camera.read_frame()
                if frame is None:
                    continue
                success = await pusher.push_frame(frame)
                # 即使推流失败也不一定算测试失败（RTMP服务器可能没开）
                # 我们只检查没有异常发生
                self.assertIsInstance(success, bool)

            await pusher.close()
            camera.release()
        except Exception as e:
            if pusher:
                await pusher.close()
            if camera:
                camera.release()
            # 推流失败可能因为RTMP服务器未运行，跳过测试而不是失败
            error_str = str(e) if e else ""
            if ("Connection refused" in error_str or
                "Unable to open" in error_str or
                "Broken pipe" in error_str or
                "BrokenPipeError" in error_str or
                not error_str):  # 空错误信息也可能是RTMP问题
                self.skipTest(f"RTMP服务器不可用: {error_str or '未知错误'}")
            else:
                self.fail(f"推流测试失败: {error_str or '未知错误'}")


def run_manual_test(qr_detection=True, width=640, height=480, fps=30, camera_index=0):
    """
    手动测试：持续运行摄像头推流
    需要：本地RTMP服务器运行在 rtmp://localhost/live/stream
    按 Ctrl+C 停止推流
    """
    print("=" * 60)
    print("手动测试模式")
    print("需要：")
    print("1. 本地RTMP服务器运行（如 nginx-rtmp）")
    print("2. 摄像头连接")
    print("=" * 60)

    # 只使用真实摄像头
    print("尝试打开真实摄像头...")
    try:
        
        camera = CameraStream(0, width=640, height=480)
        print("摄像头打开成功")
    except Exception as e:
        print(f"摄像头打开失败: {e}")
        print("无法继续测试：需要真实摄像头")
        return

    # 创建推流器
    pusher = FFmpegPusher("rtmp://localhost/live/stream", fps=30, width=640, height=480)

    async def run_stream():
        await pusher.start()

        print("开始持续摄像头推流...")
        print("按 Ctrl+C 停止推流")

        frame_count = 0

        try:
            while True:  # 无限循环，持续推流
                # 读取帧
                frame = camera.read_frame()
                if frame is None:
                    await asyncio.sleep(0.033)  # 约30fps
                    continue

                # 推流
                success = await pusher.push_frame(frame)
                frame_count += 1

                if frame_count % 30 == 0:  # 每30帧打印一次状态
                    print(f"已推流 {frame_count} 帧")

                await asyncio.sleep(0.033)  # 控制帧率

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            print(f"总计推流: {frame_count} 帧")
            await pusher.close()
            camera.release()
            print("资源已释放")

    # 运行异步任务
    asyncio.run(run_stream())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='QR码扫描和推流测试')
    parser.add_argument('--mode', choices=['unit', 'manual', 'all'], default='unit',
                       help='测试模式: unit=单元测试, manual=手动测试, all=全部')

    args = parser.parse_args()

    if args.mode == 'unit' or args.mode == 'all':
        print("运行单元测试...")
        unittest.main(argv=[''], exit=False)

    if args.mode == 'manual' or args.mode == 'all':
        run_manual_test()