#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级视觉任务测试平台

功能特性：
1. 模块化设计：QR检测、货物检测、推流、显示等模块可插拔
2. 高扩展性：易于添加新的视觉处理器
3. 配置驱动：通过命令行参数控制所有功能
4. 实时监控：性能统计和调试信息
5. 多模式支持：纯检测、检测+推流、检测+显示、检测+推流+显示

使用方式：
python test_visual.py --help
"""

import argparse
import asyncio
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Callable
import sys
import threading
import queue

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
        self.vision_processing_times = {}
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
        """记录总处理时间"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)

    def add_vision_processing_time(self, processor_name: str, processing_time: float):
        """记录特定视觉处理器的处理时间"""
        if processor_name not in self.vision_processing_times:
            self.vision_processing_times[processor_name] = []

        times = self.vision_processing_times[processor_name]
        times.append(processing_time)
        if len(times) > self.window_size:
            times.pop(0)

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

    def get_avg_vision_processing_time(self, processor_name: str) -> float:
        """特定视觉处理器的平均处理时间"""
        if processor_name not in self.vision_processing_times:
            return 0.0
        times = self.vision_processing_times[processor_name]
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_total_runtime(self) -> float:
        """总运行时间"""
        return time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """获取所有统计数据"""
        stats = {
            'fps': self.get_fps(),
            'total_frames': self.frame_count,
            'avg_processing_ms': self.get_avg_processing_time() * 1000,
            'total_runtime_s': self.get_total_runtime(),
        }

        # 添加各个视觉处理器的统计
        for processor_name in self.vision_processing_times:
            avg_time = self.get_avg_vision_processing_time(processor_name)
            stats[f'avg_{processor_name}_ms'] = avg_time * 1000

        return stats


class VisualProcessor:
    """视觉处理器基类"""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.last_result = None
        self.processing_time = 0.0

    async def initialize(self):
        """初始化处理器"""
        pass

    async def process_frame(self, frame: np.ndarray) -> Any:
        """处理帧，返回结果"""
        if not self.enabled:
            return None

        start_time = time.time()
        try:
            result = await self._process_frame_impl(frame)
            self.last_result = result
            return result
        finally:
            self.processing_time = time.time() - start_time

    async def _process_frame_impl(self, frame: np.ndarray) -> Any:
        """子类实现具体的帧处理逻辑"""
        raise NotImplementedError

    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'processing_time_ms': self.processing_time * 1000,
            'last_result': str(self.last_result)[:100] if self.last_result else None
        }

    def draw_on_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """在帧上绘制处理结果"""
        return frame


class QRCodeProcessor(VisualProcessor):
    """QR码检测处理器"""

    def __init__(self, enabled: bool = True, detection_freq: int = 1):
        super().__init__("qr_detector", enabled)
        self.detection_freq = detection_freq
        self.frame_counter = 0
        self.camera_tasks = None

    async def initialize(self):
        """初始化QR检测器"""
        if not self.enabled or not MODULES_AVAILABLE:
            return

        try:
            # 创建虚拟的TaskSequence和CameraStream用于QR检测
            task_seq = TaskSequence(batch1=["READ_QR"], batch2=[])
            # CameraTasks需要CameraStream，但我们使用传入的帧，所以创建一个虚拟的
            from camera_stream import CameraStream
            dummy_camera = CameraStream(0, width=640, height=480)
            self.camera_tasks = CameraTasks(task_seq, dummy_camera)
            print(f"QRCode处理器初始化成功 (检测频率: 每{self.detection_freq}帧检测一次)")
        except Exception as e:
            print(f"QRCode处理器初始化失败: {e}")
            self.enabled = False

    async def _process_frame_impl(self, frame: np.ndarray) -> VisionResult:
        """处理帧进行QR检测"""
        if not self.camera_tasks:
            return VisionResult(task_name="READ_QR", success=False, error_message="处理器未初始化")

        self.frame_counter += 1
        if self.frame_counter % self.detection_freq != 0:
            # 跳过检测，返回上次结果
            return self.last_result or VisionResult(
                task_name="READ_QR",
                success=False,
                error_message="等待检测"
            )

        try:
            # 使用CameraTasks处理帧
            result = self.camera_tasks.process_frame_for_qr(frame.copy())
            return result
        except Exception as e:
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message=f"检测错误: {str(e)}"
            )

    def draw_on_frame(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """绘制QR检测结果"""
        if result and hasattr(result, 'success'):
            # 如果CameraTasks已经在帧上绘制了结果，直接返回
            if hasattr(self.camera_tasks, 'current_frame') and self.camera_tasks.current_frame is not None:
                return self.camera_tasks.current_frame
        return frame

    def get_debug_info(self) -> Dict[str, Any]:
        """获取QR检测的调试信息"""
        info = super().get_debug_info()
        if self.last_result and hasattr(self.last_result, 'success'):
            info['qr_detected'] = self.last_result.success
            if self.last_result.success:
                info['qr_data'] = self.last_result.result_data[:50]
            else:
                info['qr_error'] = self.last_result.error_message
        return info


class CargoDetectorProcessor(VisualProcessor):
    """货物检测处理器（待实现）"""

    def __init__(self, enabled: bool = True):
        super().__init__("cargo_detector", enabled)

    async def _process_frame_impl(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测货物并返回坐标信息"""
        # TODO: 实现货物检测逻辑
        return {
            'cargo_count': 0,
            'coordinates': [],
            'status': '未实现'
        }

    def draw_on_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """绘制货物检测结果"""
        # TODO: 实现绘制逻辑
        return frame


class StreamingModule:
    """推流模块"""

    def __init__(self, rtmp_url: str, width: int, height: int, fps: int,
                 use_hardware_accel: bool = False, enabled: bool = True):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hardware_accel = use_hardware_accel
        self.enabled = enabled
        self.pusher = None
        self.streaming_time = 0.0
        self.fallback_to_software = False

    async def initialize(self):
        """初始化推流器"""
        if not self.enabled:
            print("推流模块已禁用")
            return

        if not MODULES_AVAILABLE:
            print("错误：项目模块不可用，无法初始化推流")
            self.enabled = False
            return

        print(f"初始化推流模块...")
        print(f"  RTMP地址: {self.rtmp_url}")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  帧率: {self.fps}")
        print(f"  硬件加速: {'启用' if self.use_hardware_accel else '禁用'}")

        try:
            self.pusher = FFmpegPusher(
                self.rtmp_url,
                fps=self.fps,
                width=self.width,
                height=self.height,
                use_hardware_accel=self.use_hardware_accel
            )
            await self.pusher.start()
            print("推流模块初始化成功")
        except Exception as e:
            print(f"推流模块初始化失败: {e}")
            print("请确保：")
            print("1. FFmpeg已安装并可在PATH中访问")
            print("2. RTMP服务器正在运行")
            self.enabled = False

    async def push_frame(self, frame: np.ndarray) -> bool:
        """推送一帧"""
        if not self.enabled or not self.pusher:
            return False

        start_time = time.time()
        try:
            success = await self.pusher.push_frame(frame)
            self.streaming_time = time.time() - start_time
            return success
        except Exception as e:
            print(f"推流错误: {e}")
            self.streaming_time = time.time() - start_time
            return False

    async def close(self):
        """关闭推流模块"""
        if self.pusher:
            await self.pusher.close()
            self.pusher = None

    def get_debug_info(self) -> Dict[str, Any]:
        """获取推流调试信息"""
        encoder_type = "软件编码"
        if self.pusher and hasattr(self.pusher, 'use_hardware_accel'):
            if self.pusher.use_hardware_accel:
                if hasattr(self.pusher, 'fallback_to_software') and self.pusher.fallback_to_software:
                    encoder_type = "软件编码 (硬件回退)"
                else:
                    encoder_type = "硬件编码"

        return {
            'enabled': self.enabled,
            'streaming_time_ms': self.streaming_time * 1000,
            'encoder_type': encoder_type,
            'rtmp_url': self.rtmp_url
        }


class DisplayModule:
    """显示模块"""

    def __init__(self, enabled: bool = True, window_name: str = "视觉任务测试"):
        self.enabled = enabled
        self.window_name = window_name
        self.display_time = 0.0

    async def initialize(self):
        """初始化显示模块"""
        if not self.enabled:
            return

        print(f"显示模块初始化 (窗口名: '{self.window_name}')")
        print("提示: 按 'q' 键退出，按 's' 键保存当前帧")

    def show_frame(self, frame: np.ndarray):
        """显示帧"""
        if not self.enabled:
            return

        start_time = time.time()
        try:
            cv2.imshow(self.window_name, frame)
            self.display_time = time.time() - start_time

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return 'quit'
            elif key == ord('s'):
                self.save_frame(frame)
                return 'save'
        except Exception as e:
            print(f"显示错误: {e}")

        return None

    def save_frame(self, frame: np.ndarray):
        """保存当前帧"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"帧已保存: {filename}")

    async def close(self):
        """关闭显示模块"""
        if self.enabled:
            cv2.destroyAllWindows()

    def get_debug_info(self) -> Dict[str, Any]:
        """获取显示调试信息"""
        return {
            'enabled': self.enabled,
            'display_time_ms': self.display_time * 1000,
            'window_name': self.window_name
        }


def draw_debug_info(frame: np.ndarray, stats: Dict[str, Any],
                   processors_info: List[Dict[str, Any]],
                   streaming_info: Dict[str, Any],
                   display_info: Dict[str, Any]) -> np.ndarray:
    """在帧上绘制调试信息"""

    height, width = frame.shape[:2]

    # 创建半透明背景区域
    overlay = frame.copy()

    # 顶部状态栏
    cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 绘制性能统计信息
    y_offset = 25
    line_height = 25

    # 帧率信息
    fps_text = f"FPS: {stats['fps']:.1f}"
    cv2.putText(frame, fps_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 总处理延迟
    latency_text = f"处理延迟: {stats['avg_processing_ms']:.1f}ms"
    cv2.putText(frame, latency_text, (200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 帧计数
    frame_text = f"帧: {stats['total_frames']}"
    cv2.putText(frame, frame_text, (400, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    y_offset += line_height

    # 处理器信息（第一行）
    processor_texts = []
    for info in processors_info:
        if info['enabled']:
            name = info['name'].replace('_', ' ').title()
            time_ms = info.get('processing_time_ms', 0)
            processor_texts.append(f"{name}: {time_ms:.1f}ms")

    if processor_texts:
        proc_line = " | ".join(processor_texts[:3])  # 最多显示3个
        cv2.putText(frame, proc_line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    y_offset += line_height

    # 处理器信息（第二行 - 结果）
    for info in processors_info:
        if info['enabled']:
            name = info['name']
            if name == 'qr_detector' and 'qr_detected' in info:
                if info['qr_detected']:
                    qr_data = info.get('qr_data', '')
                    qr_text = f"QR: {qr_data}" if qr_data else "QR: 检测到"
                    color = (0, 255, 0)
                else:
                    qr_text = f"QR: 未检测到"
                    color = (0, 0, 255)
                cv2.putText(frame, qr_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                break  # 目前只显示QR结果

    # 推流信息
    if streaming_info.get('enabled'):
        stream_x = width - 300
        stream_text = f"推流: {streaming_info.get('streaming_time_ms', 0):.1f}ms"
        cv2.putText(frame, stream_text, (stream_x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 1)

        encoder_text = f"编码: {streaming_info.get('encoder_type', '未知')}"
        cv2.putText(frame, encoder_text, (stream_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 1)

    # 时间戳
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, timestamp, (width - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 底部状态栏
    cv2.rectangle(frame, (0, height - 30),
                  (width, height), (0, 0, 0), -1)

    # 构建状态文本
    status_parts = []
    if any(p['enabled'] for p in processors_info):
        active_processors = [p['name'].replace('_', ' ') for p in processors_info if p['enabled']]
        status_parts.append(f"处理: {', '.join(active_processors)}")

    if streaming_info.get('enabled'):
        status_parts.append("推流中")

    if display_info.get('enabled'):
        status_parts.append("显示中")

    status_text = " | ".join(status_parts) + " | 按 'q' 退出 | 按 's' 保存帧"
    cv2.putText(frame, status_text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


async def run_visual_test_pipeline(
    camera_source: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    # 处理器配置
    enable_qr: bool = True,
    qr_freq: int = 1,
    enable_cargo: bool = False,
    # 推流配置
    enable_streaming: bool = False,
    rtmp_url: str = "rtmp://localhost/live/stream",
    hardware_accel: bool = False,
    # 显示配置
    enable_display: bool = True,
    window_name: str = "视觉任务测试"
):
    """
    运行视觉测试流水线

    Args:
        camera_source: 摄像头索引
        width: 视频宽度
        height: 视频高度
        fps: 目标帧率
        enable_qr: 启用QR检测
        qr_freq: QR检测频率（每N帧检测一次）
        enable_cargo: 启用货物检测
        enable_streaming: 启用推流
        rtmp_url: RTMP服务器地址
        hardware_accel: 启用硬件加速编码
        enable_display: 启用GUI显示
        window_name: 显示窗口名称
    """

    print("=" * 60)
    print("高级视觉任务测试平台")
    print("=" * 60)
    print(f"摄像头索引: {camera_source}")
    print(f"分辨率: {width}x{height}")
    print(f"目标帧率: {fps} FPS")
    print(f"QR检测: {'启用' if enable_qr else '禁用'}")
    if enable_qr:
        print(f"QR检测频率: 每{qr_freq}帧检测一次")
    print(f"货物检测: {'启用' if enable_cargo else '禁用'}")
    print(f"推流: {'启用' if enable_streaming else '禁用'}")
    if enable_streaming:
        print(f"RTMP地址: {rtmp_url}")
        print(f"硬件加速: {'启用' if hardware_accel else '禁用'}")
    print(f"显示: {'启用' if enable_display else '禁用'}")
    print("=" * 60)

    # 检查模块可用性
    if not MODULES_AVAILABLE:
        print("错误：项目模块导入失败")
        print("请确保在当前目录运行或设置PYTHONPATH")
        return

    # 初始化摄像头
    print("初始化摄像头...")
    try:
        camera = CameraStream(camera_source, width=width, height=height)
        print("摄像头初始化成功")
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return

    # 初始化视觉处理器
    processors = []

    if enable_qr:
        qr_processor = QRCodeProcessor(enabled=True, detection_freq=qr_freq)
        await qr_processor.initialize()
        processors.append(qr_processor)

    if enable_cargo:
        cargo_processor = CargoDetectorProcessor(enabled=True)
        await cargo_processor.initialize()
        processors.append(cargo_processor)

    # 如果没有启用任何处理器，至少创建一个虚拟处理器用于统计
    if not processors:
        dummy_processor = VisualProcessor("dummy", enabled=False)
        processors.append(dummy_processor)

    # 初始化推流模块
    streaming_module = StreamingModule(
        rtmp_url=rtmp_url,
        width=width,
        height=height,
        fps=fps,
        use_hardware_accel=hardware_accel,
        enabled=enable_streaming
    )
    await streaming_module.initialize()

    # 初始化显示模块
    display_module = DisplayModule(
        enabled=enable_display,
        window_name=window_name
    )
    await display_module.initialize()

    # 初始化性能监控
    monitor = PerformanceMonitor()

    print("\n开始测试...")
    if enable_display:
        print("提示: 按 'q' 键退出，按 's' 键保存当前帧")
    print("-" * 40)

    frame_counter = 0

    try:
        while True:
            loop_start = time.time()

            # 读取摄像头帧
            frame = camera.read_frame()
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            processed_frame = frame.copy()
            all_results = {}

            # 运行所有视觉处理器
            for processor in processors:
                if not processor.enabled:
                    continue

                process_start = time.time()
                result = await processor.process_frame(processed_frame.copy())
                process_time = time.time() - process_start

                # 更新性能统计
                monitor.add_vision_processing_time(processor.name, process_time)

                # 在帧上绘制处理器结果
                processed_frame = processor.draw_on_frame(processed_frame, result)

                # 保存结果
                all_results[processor.name] = result

            # 推流
            if streaming_module.enabled:
                stream_success = await streaming_module.push_frame(processed_frame)
                if not stream_success and frame_counter % 30 == 0:
                    print(f"推流失败 (帧 {frame_counter})")

            # 显示
            if display_module.enabled:
                display_result = display_module.show_frame(processed_frame)
                if display_result == 'quit':
                    print("用户请求退出")
                    break

            # 更新性能统计
            monitor.add_frame_time()
            monitor.add_processing_time(time.time() - loop_start)
            frame_counter += 1

            # 控制帧率
            target_frame_time = 1.0 / fps
            actual_time = time.time() - loop_start
            sleep_time = max(0, target_frame_time - actual_time)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # 每30帧打印一次状态
            if frame_counter % 30 == 0:
                stats = monitor.get_stats()

                # 收集调试信息
                processors_info = [p.get_debug_info() for p in processors]
                streaming_info = streaming_module.get_debug_info()
                display_info = display_module.get_debug_info()

                # 打印状态
                status_parts = [
                    f"FPS={stats['fps']:.1f}",
                    f"延迟={stats['avg_processing_ms']:.1f}ms",
                    f"帧数={stats['total_frames']}"
                ]

                for processor in processors:
                    if processor.enabled:
                        proc_time = monitor.get_avg_vision_processing_time(processor.name) * 1000
                        status_parts.append(f"{processor.name[:5]}={proc_time:.1f}ms")

                print(f"状态: {' | '.join(status_parts)}")

                # 打印QR检测结果（如果有）
                if enable_qr:
                    for processor in processors:
                        if processor.name == 'qr_detector' and processor.last_result:
                            result = processor.last_result
                            if result.success:
                                print(f"最新QR: {result.result_data[:50]}")

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")

        # 关闭摄像头
        camera.release()

        # 关闭推流模块
        await streaming_module.close()

        # 关闭显示模块
        await display_module.close()

        # 打印最终统计信息
        final_stats = monitor.get_stats()
        print("=" * 60)
        print("测试结束 - 性能统计:")
        print(f"总运行时间: {final_stats['total_runtime_s']:.1f} 秒")
        print(f"总帧数: {final_stats['total_frames']}")
        print(f"平均帧率: {final_stats['fps']:.1f} FPS")
        print(f"平均处理延迟: {final_stats['avg_processing_ms']:.1f} ms")

        # 各个处理器的统计
        for processor in processors:
            if processor.enabled:
                avg_time = monitor.get_avg_vision_processing_time(processor.name)
                print(f"平均{processor.name}时间: {avg_time * 1000:.1f} ms")

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
        description='高级视觉任务测试平台 - 模块化、可插拔的视觉任务测试'
    )

    # 摄像头配置
    parser.add_argument('--camera-source', type=int, default=0,
                       help='摄像头源索引 (默认: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='视频宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='视频高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='目标帧率 (默认: 30)')

    # 视觉处理器配置
    parser.add_argument('--no-qr', action='store_true',
                       help='禁用QR码检测')
    parser.add_argument('--qr-freq', type=int, default=1,
                       help='QR检测频率，每N帧检测一次 (默认: 1)')
    parser.add_argument('--enable-cargo', action='store_true',
                       help='启用货物检测（待实现）')

    # 推流配置
    parser.add_argument('--enable-streaming', action='store_true',
                       help='启用RTMP推流')
    parser.add_argument('--rtmp', type=str,
                       default='rtmp://localhost/live/stream',
                       help='RTMP服务器地址 (默认: rtmp://localhost/live/stream)')
    parser.add_argument('--hardware-accel', action='store_true',
                       help='启用硬件加速编码（适用于香橙派等设备）')

    # 显示配置
    parser.add_argument('--no-display', action='store_true',
                       help='禁用GUI显示（适用于无显示器的服务器环境）')
    parser.add_argument('--window-name', type=str, default='视觉任务测试',
                       help='显示窗口名称 (默认: "视觉任务测试")')

    # 检查模式
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查摄像头和FFmpeg，不实际运行')

    args = parser.parse_args()

    print("检查摄像头...")
    if not check_camera_available(args.camera_source):
        print(f"错误：摄像头索引 {args.camera_source} 不可用")
        print("请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 摄像头是否被其他程序占用")
        print("3. 尝试不同的摄像头索引 (--camera-source 1, 2, ...)")
        return 1

    print(f"摄像头 {args.camera_source} 可用")

    if args.enable_streaming:
        print("检查FFmpeg...")
        if not check_ffmpeg_available():
            print("错误：FFmpeg不可用")
            print("请安装FFmpeg：")
            print("Ubuntu/Debian: sudo apt install ffmpeg")
            print("Windows: 从 https://ffmpeg.org/download.html 下载")
            return 1
        print("FFmpeg可用")

    if args.check_only:
        print("\n所有检查通过，可以运行测试")
        return 0

    # 运行测试
    try:
        asyncio.run(run_visual_test_pipeline(
            camera_source=args.camera_source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            enable_qr=not args.no_qr,
            qr_freq=args.qr_freq,
            enable_cargo=args.enable_cargo,
            enable_streaming=args.enable_streaming,
            rtmp_url=args.rtmp,
            hardware_accel=args.hardware_accel,
            enable_display=not args.no_display,
            window_name=args.window_name
        ))
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())