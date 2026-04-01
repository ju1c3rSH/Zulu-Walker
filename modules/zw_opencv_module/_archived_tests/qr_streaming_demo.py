#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR码识别与RTMP推流融合演示（优化延迟版）

功能：
1. 从摄像头捕获视频
2. 实时进行QR码识别
3. 将识别结果和调试信息叠加到画面上
4. 使用优化的低延迟参数推流到RTMP服务器
5. 显示实时性能统计信息

延迟优化措施：
- 摄像头缓冲区设置为1帧
- FFmpeg使用超快编码和零延迟调优
- 动态帧率控制，避免固定睡眠
- 分离QR识别和推流线程（可选）

使用方式：
python qr_streaming_demo.py --rtmp rtmp://localhost/live/stream --width 640 --height 480 --fps 30
"""

import asyncio
import argparse
import time
import threading
import queue
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
import os

# 导入项目模块
try:
    from camera_stream import CameraStream
    from ffmpeg_pusher import FFmpegPusher
    from camera_tasks import CameraTasks, VisionResult
    from task_sequence import TaskSequence
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在当前目录运行或设置PYTHONPATH")
    MODULES_AVAILABLE = False


class PerformanceMonitor:
    """性能监控器，统计帧率、延迟等信息"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.processing_times = []
        self.qr_detection_times = []
        self.streaming_times = []
        self.start_time = time.time()
        self.frame_count = 0

    def add_frame_time(self):
        """记录帧时间"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1

        # 保持窗口大小
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def add_processing_time(self, processing_time: float):
        """记录处理时间"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)

    def add_qr_detection_time(self, qr_time: float):
        """记录QR检测时间"""
        self.qr_detection_times.append(qr_time)
        if len(self.qr_detection_times) > self.window_size:
            self.qr_detection_times.pop(0)

    def add_streaming_time(self, streaming_time: float):
        """记录推流时间"""
        self.streaming_times.append(streaming_time)
        if len(self.streaming_times) > self.window_size:
            self.streaming_times.pop(0)

    def get_fps(self) -> float:
        """计算帧率"""
        if len(self.frame_times) < 2:
            return 0.0
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / time_span

    def get_avg_processing_time(self) -> float:
        """平均处理时间"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_avg_qr_detection_time(self) -> float:
        """平均QR检测时间"""
        if not self.qr_detection_times:
            return 0.0
        return sum(self.qr_detection_times) / len(self.qr_detection_times)

    def get_avg_streaming_time(self) -> float:
        """平均推流时间"""
        if not self.streaming_times:
            return 0.0
        return sum(self.streaming_times) / len(self.streaming_times)

    def get_total_runtime(self) -> float:
        """总运行时间"""
        return time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """获取所有统计数据"""
        return {
            'fps': self.get_fps(),
            'total_frames': self.frame_count,
            'avg_processing_ms': self.get_avg_processing_time() * 1000,
            'avg_qr_detection_ms': self.get_avg_qr_detection_time() * 1000,
            'avg_streaming_ms': self.get_avg_streaming_time() * 1000,
            'total_runtime_s': self.get_total_runtime(),
            'estimated_latency_ms': (self.get_avg_processing_time() +
                                    self.get_avg_streaming_time()) * 1000
        }


class OptimizedFFmpegPusher(FFmpegPusher):
    """优化的FFmpeg推流器，使用更低的延迟参数"""

    def __init__(self, rtmp_url, fps=30, width=1280, height=720,
                 use_hardware_accel=False):
        super().__init__(rtmp_url, fps, width, height)
        self.use_hardware_accel = use_hardware_accel
        self.fallback_to_software = False  # 标记是否已回退到软件编码

    async def start(self):
        """启动FFmpeg进程（优化延迟版本）"""
        if self.process is not None:
            return

        # 重置停止标志，以防重新启动
        self._stopped = False

        # 基础命令
        command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # 从标准输入读取
        ]

        """根据是否使用硬件加速选择编码参数"""
        if self.use_hardware_accel and not self.fallback_to_software:
            command.extend(self._get_hardware_encoder_params())
        else:
            
            command.extend(self._get_software_encoder_params())

        """设置输出格式和URL"""
        command.extend([
            '-f', 'flv',
            self.rtmp_url
        ])

        print(f"启动FFmpeg命令: {' '.join(command)}")

        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 等待片刻检查进程是否立即退出
        await asyncio.sleep(0.1)
        if self.process.returncode is not None:
            # 进程已退出，可能是参数不兼容
            stderr_output = await self.process.stderr.read()
            error_msg = stderr_output.decode('utf-8', errors='ignore')[:500]
            print(f"FFmpeg进程立即退出 (返回码: {self.process.returncode})")
            print(f"错误输出: {error_msg}")

            if self.use_hardware_accel and not self.fallback_to_software:
                print("硬件编码器可能不兼容，尝试回退到软件编码...")
                self.fallback_to_software = True
                self.process = None
                # 重新启动使用软件编码
                return await self.start()
            else:
                raise RuntimeError(f"FFmpeg进程启动失败: {error_msg}")

        # 启动输出读取任务
        self._reader_task = asyncio.create_task(self._read_output())

    def _get_hardware_encoder_params(self):
        """获取硬件编码器参数，针对V4L2编码器优化（简化版本）"""
        # V4L2编码器限制较多，使用最简参数确保兼容性
        # 尝试最简单的参数集，如果失败则回退到软件编码
        params = [
            '-c:v', 'h264_rkmpp',  # h264_rkmpp硬件编码
            '-pix_fmt', 'nv12',
        ]

        # 根据分辨率使用保守的比特率
        total_pixels = self.width * self.height
        if total_pixels <= 640 * 480:  # 640x480或更小
            bitrate = '1000k'
        elif total_pixels <= 1280 * 720:  # 720p
            bitrate = '2000k'
        else:  # 1080p或更高
            bitrate = '4000k'

        # 使用最小必要参数
        params.extend([
            '-b:v', bitrate,
            '-g', str(min(self.fps, 15)),  # GOP大小，限制最大30
            '-profile:v', 'baseline',  # 基线profile，兼容性最好
                # 编码速度优化
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            # 禁用B帧，降低延迟
            '-bf', '0',
            # 使用固定QP，减少编码决策时间
            '-qp', '28',
            # 强制使用帧内 刷新
            
        ])

        # 尝试添加简单的缓冲控制（如果支持）
        # 但保持简单，避免不兼容参数
        if total_pixels <= 1280 * 720:
            params.extend(['-bufsize', '10k'])

        return params

    def _get_software_encoder_params(self):
        """获取软件编码器参数"""
        return [
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', '23',  # 质量参数，23是默认值
            '-g', str(self.fps),  # GOP大小等于帧率
            '-b:v', '2000k',
            '-bufsize', '1000k',
            '-maxrate', '2000k',
            '-flvflags', 'no_duration_filesize',
        ]

    async def push_frame(self, frame):
        """推送一帧到FFmpeg，如果硬件编码失败则回退到软件编码"""
        result = await super().push_frame(frame)

        # 如果推送失败且是硬件编码第一次失败，尝试回退到软件编码
        if not result and self.use_hardware_accel and not self.fallback_to_software:
            print("硬件编码失败，尝试回退到软件编码...")
            self.fallback_to_software = True

            # 关闭当前进程
            await self.close()

            # 重新启动使用软件编码
            await self.start()

            # 重试推送当前帧
            return await super().push_frame(frame)

        return result


def draw_debug_info(frame: np.ndarray, stats: Dict[str, Any],
                    qr_result: Optional[VisionResult] = None,
                    frame_index: int = 0,
                    encoder_type: str = "软件编码") -> np.ndarray:
    """在帧上绘制调试信息"""

    # 创建半透明背景区域
    overlay = frame.copy()

    # 顶部状态栏
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 绘制性能统计信息
    y_offset = 25
    line_height = 25

    # 帧率信息
    fps_text = f"FPS: {stats['fps']:.1f}"
    cv2.putText(frame, fps_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 延迟信息
    latency_text = f"延迟: {stats['estimated_latency_ms']:.1f}ms"
    cv2.putText(frame, latency_text, (200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 帧计数
    frame_text = f"帧: {stats['total_frames']}"
    cv2.putText(frame, frame_text, (400, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    y_offset += line_height

    # 处理时间
    proc_text = f"处理: {stats['avg_processing_ms']:.1f}ms"
    cv2.putText(frame, proc_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    # QR检测时间
    qr_text = f"QR检测: {stats['avg_qr_detection_ms']:.1f}ms"
    cv2.putText(frame, qr_text, (200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    # 推流时间
    stream_text = f"推流: {stats['avg_streaming_ms']:.1f}ms"
    cv2.putText(frame, stream_text, (400, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    y_offset += line_height

    # QR识别结果
    if qr_result:
        if qr_result.success:
            qr_status = f"QR: {qr_result.result_data[:40]}"
            color = (0, 255, 0)
        else:
            qr_status = f"QR: {qr_result.error_message}"
            color = (0, 0, 255)
    else:
        qr_status = "QR: 未检测"
        color = (200, 200, 200)

    cv2.putText(frame, qr_status, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 时间戳
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, timestamp, (frame.shape[1] - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 底部状态栏（编码信息）
    cv2.rectangle(frame, (0, frame.shape[0] - 30),
                  (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    status_text = f"RTMP推流中 | {encoder_type} | 分辨率: {frame.shape[1]}x{frame.shape[0]} | 按Ctrl+C停止"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


async def qr_streaming_demo(
    rtmp_url: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    camera_index: int = 0,
    use_hardware_accel: bool = False,
    qr_detection_freq: int = 1  # 每N帧进行一次QR检测
):
    """
    QR码识别与推流融合演示主函数

    Args:
        rtmp_url: RTMP服务器地址
        width: 视频宽度
        height: 视频高度
        fps: 目标帧率
        camera_index: 摄像头索引
        use_hardware_accel: 是否使用硬件加速编码
        qr_detection_freq: QR检测频率（每N帧检测一次）
    """

    print("=" * 60)
    print("QR码识别与RTMP推流融合演示（优化延迟版）")
    print("=" * 60)
    print(f"摄像头索引: {camera_index}")
    print(f"分辨率: {width}x{height}")
    print(f"目标帧率: {fps} FPS")
    print(f"RTMP地址: {rtmp_url}")
    print(f"硬件加速: {'启用' if use_hardware_accel else '禁用'}")
    print(f"QR检测频率: 每{qr_detection_freq}帧检测一次")
    print("=" * 60)

    # 检查模块是否可用
    if not MODULES_AVAILABLE:
        print("错误：项目模块导入失败")
        return

    # 初始化摄像头
    print("初始化摄像头...")
    try:
        camera = CameraStream(camera_index, width=width, height=height)
        print("摄像头初始化成功")
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return

    # 初始化CameraTasks（用于QR识别）
    print("初始化QR码识别模块...")
    try:
        # 创建虚拟的TaskSequence（实际不需要批次控制）
        task_seq = TaskSequence(batch1=["READ_QR"], batch2=[])
        camera_tasks = CameraTasks(task_seq, camera)
        print("QR码识别模块初始化成功")
    except Exception as e:
        print(f"QR码识别模块初始化失败: {e}")
        camera.release()
        return

    # 初始化推流器
    print("初始化FFmpeg推流器...")
    try:
        pusher = OptimizedFFmpegPusher(
            rtmp_url,
            fps=fps,
            width=width,
            height=height,
            use_hardware_accel=use_hardware_accel
        )
        await pusher.start()
        print("FFmpeg推流器初始化成功")
    except Exception as e:
        print(f"FFmpeg推流器初始化失败: {e}")
        print("请确保：")
        print("1. FFmpeg已安装并可在PATH中访问")
        print("2. RTMP服务器正在运行")
        camera.release()
        return

    # 初始化性能监控
    monitor = PerformanceMonitor()

    # 主循环
    print("\n开始推流...")
    print("按 Ctrl+C 停止推流")
    print("-" * 40)

    frame_counter = 0
    qr_result = VisionResult(task_name="READ_QR", success=False, error_message="等待首次检测")

    try:
        while True:
            loop_start = time.time()

            # 读取摄像头帧
            frame = camera_tasks._get_frame()
            if frame is None:
                await asyncio.sleep(0.001)  # 短暂等待避免CPU占用过高
                continue

            # 每qr_detection_freq帧进行一次QR检测
            if frame_counter % qr_detection_freq == 0:
                qr_start = time.time()
                # 使用新的process_frame_for_qr方法处理当前帧
                qr_result = camera_tasks.process_frame_for_qr(frame.copy())
                qr_time = time.time() - qr_start
                monitor.add_qr_detection_time(qr_time)

                # 使用CameraTasks处理后的帧（QR绘制已在process_frame_for_qr中完成）
                processed_frame = camera_tasks.current_frame
            else:
                # 如果不是检测帧，保留上一次的检测结果用于显示
                # qr_result保持不变
                processed_frame = frame

            # 绘制调试信息到帧上
            stats = monitor.get_stats()

            # 确定编码器类型
            if pusher.fallback_to_software:
                encoder_type = "软件编码 (硬件回退)"
            elif pusher.use_hardware_accel:
                encoder_type = "硬件编码"
            else:
                encoder_type = "软件编码"

            frame_with_info = draw_debug_info(processed_frame.copy(), stats, qr_result, frame_counter, encoder_type)

            # 推流
            stream_start = time.time()
            success = await pusher.push_frame(frame_with_info)
            stream_time = time.time() - stream_start

            # 更新性能统计
            monitor.add_frame_time()
            monitor.add_processing_time(time.time() - loop_start)
            monitor.add_streaming_time(stream_time)
            frame_counter += 1

            # 控制帧率（动态调整等待时间）
            target_frame_time = 1.0 / fps
            actual_time = time.time() - loop_start
            sleep_time = max(0, target_frame_time - actual_time)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # 每30帧打印一次状态
            if frame_counter % 30 == 0:
                stats = monitor.get_stats()
                print(f"状态: FPS={stats['fps']:.1f}, "
                      f"延迟={stats['estimated_latency_ms']:.1f}ms, "
                      f"总帧数={stats['total_frames']}")

                if qr_result and qr_result.success:
                    print(f"最新QR: {qr_result.result_data[:50]}")

    except KeyboardInterrupt:
        print("\n用户中断推流")
    except Exception as e:
        print(f"推流过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")
        await pusher.close()
        camera.release()

        # 打印最终统计信息
        final_stats = monitor.get_stats()
        print("=" * 60)
        print("推流结束 - 性能统计:")
        print(f"总运行时间: {final_stats['total_runtime_s']:.1f} 秒")
        print(f"总帧数: {final_stats['total_frames']}")
        print(f"平均帧率: {final_stats['fps']:.1f} FPS")
        print(f"平均处理延迟: {final_stats['avg_processing_ms']:.1f} ms")
        print(f"平均QR检测时间: {final_stats['avg_qr_detection_ms']:.1f} ms")
        print(f"平均推流时间: {final_stats['avg_streaming_ms']:.1f} ms")
        print(f"估计总延迟: {final_stats['estimated_latency_ms']:.1f} ms")

        # 显示编码器类型
        if pusher.fallback_to_software:
            print(f"编码器: 硬件编码 -> 软件编码 (回退)")
        elif pusher.use_hardware_accel:
            print(f"编码器: 硬件编码")
        else:
            print(f"编码器: 软件编码")
        print("=" * 60)


def check_camera_available(index: int = 0) -> bool:
    """检查摄像头是否可用"""
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except:
        return False


def check_ffmpeg_available() -> bool:
    """检查FFmpeg是否可用"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'],
                               capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description='QR码识别与RTMP推流融合演示（优化延迟版）'
    )

    parser.add_argument('--rtmp', type=str,
                       default='rtmp://localhost/live/stream',
                       help='RTMP服务器地址 (默认: rtmp://localhost/live/stream)')

    parser.add_argument('--width', type=int, default=640,
                       help='视频宽度 (默认: 640)')

    parser.add_argument('--height', type=int, default=480,
                       help='视频高度 (默认: 480)')

    parser.add_argument('--fps', type=int, default=30,
                       help='目标帧率 (默认: 30)')

    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头索引 (默认: 0)')

    parser.add_argument('--hardware-accel', action='store_true',
                       help='启用硬件加速编码（适用于香橙派等设备）')

    parser.add_argument('--qr-freq', type=int, default=1,
                       help='QR检测频率，每N帧检测一次 (默认: 1)')

    parser.add_argument('--check-only', action='store_true',
                       help='仅检查摄像头和FFmpeg，不实际运行')

    args = parser.parse_args()

    # 检查摄像头
    print("检查摄像头...")
    if not check_camera_available(args.camera):
        print(f"错误：摄像头索引 {args.camera} 不可用")
        print("请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 摄像头是否被其他程序占用")
        print("3. 尝试不同的摄像头索引 (--camera 1, 2, ...)")
        return 1

    print(f"摄像头 {args.camera} 可用")

    # 检查FFmpeg
    print("检查FFmpeg...")
    if not check_ffmpeg_available():
        print("错误：FFmpeg不可用")
        print("请安装FFmpeg：")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        print("Windows: 从 https://ffmpeg.org/download.html 下载")
        return 1

    print("FFmpeg可用")

    # 如果只检查
    if args.check_only:
        print("\n所有检查通过，可以运行推流演示")
        return 0

    # 运行演示
    try:
        asyncio.run(qr_streaming_demo(
            rtmp_url=args.rtmp,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_index=args.camera,
            use_hardware_accel=args.hardware_accel,
            qr_detection_freq=args.qr_freq
        ))
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())