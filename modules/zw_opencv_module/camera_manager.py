# -*- coding: utf-8 -*-
import os
import sys
import time

# 添加项目根目录到 sys.path 以支持跨模块导入
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Callable, Tuple
from threading import Thread
import numpy as np

from .camera_stream import CameraStream
from .task_manager import TaskManager, Task
from .task_sequence import TaskSequence
from .frame_composer import FrameComposer
from .ffmpeg_pusher import FFmpegPusher
from .processors.base import VisionResult
from .processors.qr_processor import QRProcessor
from .processors.cargo_processor import CargoProcessor
from .performance import profiler
from utils.state import EventType, RobotState, get_state_machine
from modules.zw_uart_module.protocol import ERROR_TYPE_X, ERROR_TYPE_Y


@dataclass
class CameraConfig:
    """相机配置"""

    source: Union[int, str]
    width: int = 640
    height: int = 480
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    enabled: bool = True
    tasks: List[dict] = field(default_factory=list)
    gaussian_blur_enabled: bool = False
    gaussian_blur_kernel_size: int = 5
    gaussian_blur_sigma: float = 1.5


@dataclass
class CameraSystemConfig:
    """相机系统配置"""

    output_width: int = 1280
    output_height: int = 720
    layout: str = "grid"
    enable_streaming: bool = False  # 是否启用RTMP推流
    rtmp_url: str = ""
    enable_local_display: bool = False  # 本地显示窗口
    cameras: List[CameraConfig] = field(default_factory=list)


class Camera:
    """相机类，管理单个相机的流和任务"""

    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.enabled = config.enabled
        self.stream: Optional[CameraStream] = None
        self.task_manager = TaskManager()
        self._last_frame: Optional[np.ndarray] = None  # 缓存最后一帧
        # 高斯模糊参数
        self.gaussian_blur_enabled = config.gaussian_blur_enabled
        self.gaussian_blur_kernel_size = config.gaussian_blur_kernel_size
        self.gaussian_blur_sigma = config.gaussian_blur_sigma
        # 焦距计算器
        self.focal_calculator = None
        self._setup_stream(config)
        self._setup_tasks(config.tasks)
        self._init_focal_calculator(config)

    def _setup_stream(self, config: CameraConfig):
        """设置相机流"""
        try:
            self.stream = CameraStream(config.source, config.width, config.height)
        except Exception as e:
            print(f"Failed to setup camera stream for {self.camera_id}: {e}")
            self.enabled = False

    def _setup_tasks(self, task_configs: List[dict]):
        """设置任务"""
        for task_config in task_configs:
            task_name = task_config.get("name", "")
            task_type = task_config.get("type", "")
            task_enabled = task_config.get("enabled", True)

            processor = self._create_processor(task_type, task_name)
            if processor:
                task = Task(task_name, processor, task_enabled)
                self.task_manager.register_task(task)

    def _create_processor(self, task_type: str, name: str):
        if task_type == "QRProcessor":
            return QRProcessor(name)
        elif task_type == "CargoProcessor":
            return CargoProcessor(name)
        return None

    def _init_focal_calculator(self, config: CameraConfig):
        """初始化焦距距离计算器"""
        if (config.focal_length_mm and config.sensor_width_mm
            and config.sensor_height_mm):
            from .utils.focal_distance_util import CameraIntrinsics, FocalDistanceCalculator
            intrinsics = CameraIntrinsics(
                focal_length_mm=config.focal_length_mm,
                sensor_width_mm=config.sensor_width_mm,
                sensor_height_mm=config.sensor_height_mm,
                image_width=config.width,
                image_height=config.height,
            )
            self.focal_calculator = FocalDistanceCalculator(intrinsics=intrinsics)
        else:
            self.focal_calculator = None

    def get_distance_to_target(self, real_size_mm: float, pixel_size: float):
        """
        计算到目标的距离

        Args:
            real_size_mm: 目标实际尺寸 (mm)
            pixel_size: 目标在图像中的像素尺寸

        Returns:
            目标到相机的距离 (mm)，未配置焦距参数时返回 None
        """
        if self.focal_calculator:
            return self.focal_calculator.calculate_distance(real_size_mm, pixel_size)
        return None

    def get_camera_coords(self, pixel_x: float, pixel_y: float, distance_mm: float):
        """
        获取像素点在相机坐标系下的坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            distance_mm: 目标距离 (mm)

        Returns:
            (X, Y, Z) 相机坐标系下的坐标 (mm)，未配置焦距参数时返回 None
        """
        if self.focal_calculator:
            return self.focal_calculator.pixel_to_camera_coords(
                pixel_x, pixel_y, distance_mm
            )
        return None

    def enable(self):
        """启用相机"""
        self.enabled = True

    def disable(self):
        """禁用相机"""
        self.enabled = False

    def enable_task(self, task_name: str) -> bool:
        """启用指定任务"""
        return self.task_manager.enable_task(task_name)

    def disable_task(self, task_name: str) -> bool:
        """禁用指定任务"""
        return self.task_manager.disable_task(task_name)

    def get_task(self, task_name: str) -> Optional[Task]:
        """获取任务"""
        return self.task_manager.get_task(task_name)

    def get_frame(self) -> Optional[np.ndarray]:
        """获取帧"""
        if not self.enabled or self.stream is None:
            return None
        return self.stream.read_frame()

    def process_frame(self, fps: float = 0.0) -> Tuple[Optional[np.ndarray], Dict[str, VisionResult]]:
        """获取帧并执行所有任务，无新帧时返回缓存帧"""
        if not self.enabled:
            return None, {}

        frame = self.get_frame()
        if frame is not None:
            self._last_frame = frame.copy()  # 保存干净副本，避免累积绘制问题
        elif self._last_frame is not None:
            frame = self._last_frame
        else:
            return None, {}

        # 应用高斯模糊降噪
        if self.gaussian_blur_enabled:
            frame = cv2.GaussianBlur(
                frame,
                (self.gaussian_blur_kernel_size, self.gaussian_blur_kernel_size),
                self.gaussian_blur_sigma
            )

        # 计时：处理阶段
        profiler.start("processing")
        context = {
            "fps": fps,
            "focal_calculator": self.focal_calculator,
        }
        result = self.task_manager.run_tasks_serial(frame, context=context)
        profiler.stop("processing")

        return result

    def release(self):
        """释放资源"""
        if self.stream:
            self.stream.release()
            self.stream = None


class CameraManager:
    """相机管理器（单例），管理多个相机"""

    _instance: Optional["CameraManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.cameras: Dict[str, Camera] = {}
        self.frame_composer: Optional[FrameComposer] = None
        self.ffmpeg_pusher: Optional[FFmpegPusher] = None
        self.config: Optional[CameraSystemConfig] = None
        self.task_sequence: Optional[TaskSequence] = None

        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []

        # FPS 计算
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        self._fps = 0.0

    @classmethod
    def from_config(cls, config_path: str) -> "CameraManager":
        """从配置文件创建CameraManager"""
        instance = cls()
        instance.load_config(config_path)
        return instance

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # 解析系统配置
        system = config_data.get("system", {})
        self.config = CameraSystemConfig(
            output_width=system.get("output_width", 1280),
            output_height=system.get("output_height", 720),
            layout=system.get("layout", "grid"),
            enable_streaming=system.get("enable_streaming", False),
            rtmp_url=system.get("rtmp_url", ""),
            enable_local_display=system.get("enable_local_display", False),
        )

        self.frame_composer = FrameComposer(
            layout=self.config.layout,
            output_size=(self.config.output_width, self.config.output_height),
        )

        # 解析相机配置
        for cam_data in config_data.get("cameras", []):
            cam_id = cam_data.get("id", "")
            if not cam_id:
                continue

            preprocess = cam_data.get("preprocess", {})
            gaussian_blur = preprocess.get("gaussian_blur", {})
            cam_config = CameraConfig(
                source=cam_data.get("source", 0),
                focal_length_mm=cam_data.get("focal_length_mm", None),
                sensor_width_mm=cam_data.get("sensor_width_mm", None),
                sensor_height_mm=cam_data.get("sensor_height_mm", None),
                width=cam_data.get("width", 640),
                height=cam_data.get("height", 480),
                enabled=cam_data.get("enabled", True),
                tasks=cam_data.get("tasks", []),
                gaussian_blur_enabled=gaussian_blur.get("enabled", False),
                gaussian_blur_kernel_size=gaussian_blur.get("kernel_size", 5),
                gaussian_blur_sigma=gaussian_blur.get("sigma", 1.5),
            )
            self.add_camera(cam_id, cam_config)

    def add_camera(self, camera_id: str, config: CameraConfig):
        """添加相机"""
        if camera_id in self.cameras:
            print(f"Camera {camera_id} already exists, replacing...")
            self.remove_camera(camera_id)

        camera = Camera(camera_id, config)
        self.cameras[camera_id] = camera

    def remove_camera(self, camera_id: str):
        """移除相机"""
        camera = self.cameras.pop(camera_id, None)
        if camera:
            camera.release()

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """获取相机"""
        return self.cameras.get(camera_id)

    def get_all_enabled_cameras(self) -> List[Camera]:
        """获取所有启用的相机"""
        return [cam for cam in self.cameras.values() if cam.enabled]

    # === 生命周期控制 ===

    def enable_camera(self, camera_id: str) -> bool:
        """启用指定相机"""
        cam = self.cameras.get(camera_id)
        if cam:
            cam.enable()
            return True
        return False

    def disable_camera(self, camera_id: str) -> bool:
        """禁用指定相机"""
        cam = self.cameras.get(camera_id)
        if cam:
            cam.disable()
            return True
        return False

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        """启用指定相机的指定任务"""
        cam = self.cameras.get(camera_id)
        if cam:
            return cam.enable_task(task_name)
        return False

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        """禁用指定相机的指定任务"""
        cam = self.cameras.get(camera_id)
        if cam:
            return cam.disable_task(task_name)
        return False

    def _set_thread_affinity(self, cores):
        """设置当前线程的 CPU 亲和性（大核心）"""
        try:
            os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, PermissionError):
            pass  # Windows 或权限不足时忽略

    def start(self):
        """启动处理循环（持续执行直到调用stop）"""
        if self._running:
            return

        # 如果启用推流，初始化FFmpegPusher
        if self.config and self.config.enable_streaming and self.config.rtmp_url:
            print(f"[CameraManager] Initializing RTMP streaming to {self.config.rtmp_url}")
            self.ffmpeg_pusher = FFmpegPusher(
                self.config.rtmp_url,
                fps=30,
                width=self.config.output_width,
                height=self.config.output_height,
                use_hardware_accel=True,
            )

        # 注册 UART PICK 事件回调
        from modules.zw_uart_module import get_interface
        uart_interface = get_interface()
        if uart_interface:
            uart_interface.add_pick_callback(self._on_uart_pick_event)

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self):
        """停止处理循环"""
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def _process_loop(self):
        """内部处理循环，持续执行"""
        # 绑定到大核心 (RK3588: 4-7 是大核心 A76)
        self._set_thread_affinity([4, 5, 6, 7])

        # 如果启用推流，先启动FFmpegPusher
        if self.config and self.config.enable_streaming and self.ffmpeg_pusher:
            try:
                self.ffmpeg_pusher.start_sync()
            except Exception as e:
                print(f"[CameraManager] Failed to start FFmpegPusher: {e}")

        while self._running:
            profiler.start("total")

            # FPS 计算
            self._fps_frame_count += 1
            elapsed_fps = time.time() - self._fps_start_time
            if elapsed_fps >= 1.0:
                self._fps = self._fps_frame_count / elapsed_fps
                self._fps_frame_count = 0
                self._fps_start_time = time.time()

            sm = get_state_machine()
            current_state = sm.state

            self._update_tasks_by_state(current_state)

            profiler.start("process_all")
            composed_frame, all_results = self.process_all()
            profiler.stop("process_all")

            # 状态相关的特殊处理
            if current_state == RobotState.READ_QR:
                self._handle_qr_state(all_results, sm)

            # 货物检测处理
            self._handle_cargo_detect(all_results, sm)

            if self.config and self.config.enable_streaming:
                if self.ffmpeg_pusher and composed_frame is not None:
                    profiler.start("ffmpeg_push")
                    self.ffmpeg_pusher.push_frame_sync(composed_frame)
                    profiler.stop("ffmpeg_push")

            # 本地显示窗口（降采样到 640x480 以提高性能）
            if self.config and self.config.enable_local_display and composed_frame is not None:
                profiler.start("local_display")
                display_frame = cv2.resize(composed_frame, (640, 480))
                cv2.imshow("Zulu-Walker Camera Preview", display_frame)
                key = cv2.waitKey(1) & 0xFF
                profiler.stop("local_display")
                if key == ord('q') or key == 27:  # q 或 ESC 退出
                    self._running = False
                    break

            # 定期日志输出
            if self._fps_frame_count == 0 and self._fps > 0:
                print(f"[CameraManager] FPS: {self._fps:.1f} | State: {current_state.value}")

            # 触发全局回调
            profiler.start("callbacks")
            for cb in self._result_callbacks:
                try:
                    cb(all_results)
                except Exception as e:
                    print(f"Error in result callback: {e}")
            profiler.stop("callbacks")

            profiler.stop("total")
            profiler.end_frame()

    def _update_tasks_by_state(self, state: RobotState):
        """根据状态更新任务开关"""
        for camera_id, camera in self.cameras.items():
            if state == RobotState.READ_QR:
                # QR读取状态：启用QR检测任务
                camera.enable_task("qr_detect")
            elif state in (RobotState.IDLE, RobotState.FINISHED, RobotState.ERROR):
                # 空闲/完成/错误状态：可以禁用所有任务以节省资源
                # 但保持相机运行以便监控
                pass
            # 其他状态根据需要扩展

    def _handle_qr_state(self, all_results: Dict, sm):
        """处理QR检测状态"""
        for camera_id, results in all_results.items():
            qr_result = results.get("qr_detect")
            if qr_result and qr_result.success:
                qr_data = qr_result.result_data
                if isinstance(qr_data, dict):
                    data = qr_data.get("data", "")
                else:
                    data = str(qr_data)

                if data:
                    print(f"[CameraManager] QR detected: {data}")

                    # 检查是否为任务序列格式
                    if self._is_task_sequence(data):
                        self._start_task_sequence(data, sm)
                        return

                    # 普通QR码处理
                    sm.set_context("qr_data", data)
                    sm.set_context("qr_camera", camera_id)
                    sm.trigger(EventType.QR_DECODED, data)
                    return

    def _is_task_sequence(self, data: str) -> bool:
        """检查QR数据是否为任务序列格式"""
        if not data:
            return False
        parts = data.split('+')
        for part in parts:
            if not part:
                continue
            if not all(c in '123' for c in part):
                return False
        return True

    def _start_task_sequence(self, qr_data: str, sm):
        """启动任务序列"""
        print(f"[CameraManager] Starting task sequence: {qr_data}")

        self.task_sequence = TaskSequence.from_qr_data(qr_data)

        # 关闭 cam_0 的 qr_detect 任务
        self.disable_task("cam_0", "qr_detect")

        # 开启 cam_1 的 cargo_detect 任务
        self.enable_task("cam_1", "cargo_detect")

        # 设置目标颜色
        target_color = self.task_sequence.get_next_target()
        self._set_cargo_target_color(target_color)

        # 更新状态机
        sm.set_context("task_sequence", qr_data)
        sm.trigger(EventType.QR_DECODED, qr_data)

    def _set_cargo_target_color(self, color: Optional[str]):
        """设置货物检测的目标颜色"""
        cam_1 = self.get_camera("cam_1")
        if cam_1:
            cargo_task = cam_1.get_task("cargo_detect")
            if cargo_task:
                cargo_task.processor.set_target_color(color)

    def _handle_cargo_detect(self, all_results: Dict, sm):
        """处理货物检测"""
        if not self.task_sequence:
            return

        cargo_result = all_results.get("cam_1", {}).get("cargo_detect")
        if cargo_result and cargo_result.success:
            error_data = cargo_result.result_data
            if error_data and error_data.get("target_found"):
                percent_error_x = error_data.get("percent_error_x", 0)
                percent_error_y = error_data.get("percent_error_y", 0)

                # 通过 UART 发送坐标偏差
                from modules.zw_uart_module import send_error
                send_error(ERROR_TYPE_X, percent_error_x)
                send_error(ERROR_TYPE_Y, percent_error_y)

    def _on_uart_pick_event(self, zone_id: int):
        """处理 UART PICK 事件，推进到下一个货物"""
        print(f"[CameraManager] PICK event received, zone={zone_id}")

        if not self.task_sequence:
            return

        self.task_sequence.advance()
        target_color = self.task_sequence.get_next_target()

        if target_color:
            print(f"[CameraManager] Next target color: {target_color}")
            self._set_cargo_target_color(target_color)
        else:
            print("[CameraManager] All batches completed")
            self._finish_task_sequence()

    def _finish_task_sequence(self):
        """完成任务序列，恢复初始状态"""
        self.disable_task("cam_1", "cargo_detect")
        self.enable_task("cam_0", "qr_detect")
        self.task_sequence = None

        sm = get_state_machine()
        sm.trigger(EventType.ALL_BATCHES_DONE)

    def process_all(
        self,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Dict[str, VisionResult]]]:
        """处理所有相机，返回融合画面和所有结果"""
        frames = []
        all_results: Dict[str, Dict[str, VisionResult]] = {}
        camera_ids = []

        # 遍历所有相机（不只是启用的），保持布局稳定
        for cam in self.cameras.values():
            if not cam.enabled:
                # 禁用的相机用黑屏占位
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
                frames.append(frame)
                camera_ids.append(cam.camera_id)
                all_results[cam.camera_id] = {}
                continue

            frame, results = cam.process_frame(fps=self._fps)
            # 即使 frame 为 None，也保持相机位置和顺序
            if frame is None:
                # 创建黑屏帧占位，保持布局稳定
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
            frames.append(frame)
            camera_ids.append(cam.camera_id)
            all_results[cam.camera_id] = results

        if not frames:
            return None, all_results

        # 如果 frame_composer 未初始化，创建默认的
        if self.frame_composer is None:
            self.frame_composer = FrameComposer(
                layout="grid",
                output_size=(1280, 720),
            )

        composed = self.frame_composer.compose(frames, camera_ids)
        return composed, all_results

    # === 回调管理 ===

    def add_result_callback(self, callback: Callable[[Dict], None]):
        """添加结果回调"""
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]):
        """移除结果回调"""
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)


    # === 推流管理 ===

    def start_streaming(self, rtmp_url: str = None):
        """启动推流"""
        url = rtmp_url or (self.config.rtmp_url if self.config else "")
        if not url:
            print("No RTMP URL configured")
            return

        self.ffmpeg_pusher = FFmpegPusher(
            url,
            fps=30,
            width=self.config.output_width if self.config else 1280,
            height=self.config.output_height if self.config else 720,
            use_hardware_accel=False,
        )

        import asyncio

        asyncio.run(self.ffmpeg_pusher.start())

    def stop_streaming(self):
        """停止推流"""
        if self.ffmpeg_pusher:
            self.ffmpeg_pusher.close_sync()
            self.ffmpeg_pusher = None

    def release(self):
        """释放所有资源"""
        self.stop()
        self.stop_streaming()

        for camera in self.cameras.values():
            camera.release()

        self.cameras.clear()

    def __del__(self):
        """析构函数"""
        self.release()
