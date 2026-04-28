# -*- coding: utf-8 -*-
import os
import sys
import time

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
from .frame_composer import FrameComposer
from .ffmpeg_pusher import FFmpegPusher
from .processors.base import VisionResult
from .processors.circle_target_processor import CircleTargetProcessor
from .performance import profiler
from .param_utils import load_detect_params, apply_params_to_detector, get_config_path
from utils.state_machine import VisualStateMachine
from modules.zw_uart_module import send_orange_frame
from modules.zw_uart_module.protocol import (
    ORANGE_STATE_IDLE, ORANGE_STATE_SEARCH, ORANGE_STATE_TRACKING,
    ORANGE_STATE_RECOVERY, ORANGE_STATE_FAIL
)

@dataclass
class CameraConfig:
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
    output_width: int = 640
    output_height: int = 480
    layout: str = "grid"
    enable_streaming: bool = False
    rtmp_url: str = ""
    enable_local_display: bool = False  # 本地显示窗口
    cameras: List[CameraConfig] = field(default_factory=list)


class Camera:
    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.enabled = config.enabled
        self.stream: Optional[CameraStream] = None
        self.task_manager = TaskManager()
        self._last_frame: Optional[np.ndarray] = None
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
        try:
            self.stream = CameraStream(config.source, config.width, config.height)
        except Exception as e:
            print(f"Failed to setup camera stream for {self.camera_id}: {e}")
            self.enabled = False

    def _setup_tasks(self, task_configs: List[dict]):
        for task_config in task_configs:
            task_name = task_config.get("name", "")
            task_type = task_config.get("type", "")
            task_enabled = task_config.get("enabled", True)

            processor = self._create_processor(task_type, task_name)
            if processor:
                task = Task(task_name, processor, task_enabled)
                self.task_manager.register_task(task)

    def _create_processor(self, task_type: str, name: str):
        if task_type == "CircleTargetProcessor":
            return CircleTargetProcessor(name)
        return None

    def _init_focal_calculator(self, config: CameraConfig):
        """初始化焦距距离计算器"""
        if (config.focal_length_mm and config.sensor_width_mm
            and config.sensor_height_mm):
            from utils.focal_distance_util import CameraIntrinsics, FocalDistanceCalculator
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
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable_task(self, task_name: str) -> bool:
        return self.task_manager.enable_task(task_name)

    def disable_task(self, task_name: str) -> bool:
        return self.task_manager.disable_task(task_name)

    def get_task(self, task_name: str) -> Optional[Task]:
        return self.task_manager.get_task(task_name)

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.enabled or self.stream is None:
            return None
        return self.stream.read_frame()

    def process_frame(self, fps: float = 0.0) -> Tuple[Optional[np.ndarray], Dict[str, VisionResult]]:
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
        if self.stream:
            self.stream.release()
            self.stream = None


class CameraManager:
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
        self.state_machine = VisualStateMachine()

        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []

        # FPS 计算
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        self._fps = 0.0

        self._setup_state_callbacks()

    def _setup_state_callbacks(self):
        def on_search(context, from_state):
            print("[CameraManager] Enter SEARCH state")
            for camera in self.cameras.values():
                camera.enable_task("circle_detect")

        def on_tracking(context, from_state):
            print("[CameraManager] Enter TRACKING state")

        def on_idle(context, from_state):
            print("[CameraManager] Enter IDLE state")
            for camera in self.cameras.values():
                camera.disable_task("circle_detect")

        self.state_machine.on_state_enter(VisualStateMachine.States.SEARCH, on_search)
        self.state_machine.on_state_enter(VisualStateMachine.States.TRACKING, on_tracking)
        self.state_machine.on_state_enter(VisualStateMachine.States.IDLE, on_idle)

    @classmethod
    def from_config(cls, config_path: str) -> "CameraManager":
        instance = cls()
        instance.load_config(config_path)
        return instance

    def load_config(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

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
        if camera_id in self.cameras:
            print(f"Camera {camera_id} already exists, replacing...")
            self.remove_camera(camera_id)

        camera = Camera(camera_id, config)
        self.cameras[camera_id] = camera

    def remove_camera(self, camera_id: str):
        camera = self.cameras.pop(camera_id, None)
        if camera:
            camera.release()

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return self.cameras.get(camera_id)

    def get_all_enabled_cameras(self) -> List[Camera]:
        return [cam for cam in self.cameras.values() if cam.enabled]

    def enable_camera(self, camera_id: str) -> bool:
        cam = self.cameras.get(camera_id)
        if cam:
            cam.enable()
            return True
        return False

    def disable_camera(self, camera_id: str) -> bool:
        cam = self.cameras.get(camera_id)
        if cam:
            cam.disable()
            return True
        return False

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        cam = self.cameras.get(camera_id)
        if cam:
            return cam.enable_task(task_name)
        return False

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        cam = self.cameras.get(camera_id)
        if cam:
            return cam.disable_task(task_name)
        return False

    def set_target_color(self, color: Optional[str]):
        self.state_machine.context.target_color = color
        for camera in self.cameras.values():
            task = camera.get_task("circle_detect")
            if task:
                task.processor.set_target_color(color)

    def start(self):
        if self._running:
            return

        if self.config and self.config.enable_streaming and self.config.rtmp_url:
            print(f"[CameraManager] Initializing RTMP streaming to {self.config.rtmp_url}")
            self.ffmpeg_pusher = FFmpegPusher(
                self.config.rtmp_url,
                fps=30,
                width=self.config.output_width,
                height=self.config.output_height,
                use_hardware_accel=True,
            )

        # 从 YAML 加载检测参数并应用到所有检测器
        self._apply_detect_params()

        self.state_machine.start()

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _apply_detect_params(self):
        """从 YAML 加载检测参数并应用到所有检测器"""
        config_path = get_config_path()
        current_method, methods_params = load_detect_params(config_path)
        params = methods_params.get(current_method.value, {})

        for camera in self.cameras.values():
            task = camera.get_task("circle_detect")
            if task and hasattr(task.processor, 'detector'):
                apply_params_to_detector(task.processor.detector, current_method, params)

        print(f"[CameraManager] Loaded detect params: method={current_method.value}")

    def stop(self):
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def _set_thread_affinity(self, cores):
        """设置当前线程的 CPU 亲和性（大核心）"""
        try:
            os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, PermissionError):
            pass  # Windows 或权限不足时忽略

    def _process_loop(self):
        # 绑定到大核心 (RK3588: 4-7 是大核心 A76)
        self._set_thread_affinity([4, 5, 6, 7])

        if self.config and self.config.enable_streaming and self.ffmpeg_pusher:
            try:
                self.ffmpeg_pusher.start_sync()
            except Exception as e:
                print(f"[CameraManager] Failed to start FFmpegPusher: {e}")

        while self._running:
            profiler.start("total")
            start_time = time.time()

            self._fps_frame_count += 1
            elapsed_fps = time.time() - self._fps_start_time
            if elapsed_fps >= 1.0:
                self._fps = self._fps_frame_count / elapsed_fps
                self._fps_frame_count = 0
                self._fps_start_time = time.time()

            profiler.start("process_all")
            composed_frame, all_results = self.process_all()
            profiler.stop("process_all")

            profiler.start("handle_detection")
            self._handle_detection(all_results)
            profiler.stop("handle_detection")

            profiler.start("state_machine")
            self.state_machine.update()
            profiler.stop("state_machine")

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
                ctx = self.state_machine.context
                status = "TRACKING" if ctx.target_found else "SEARCHING"
                target_info = ""
                if ctx.target_found and ctx.target_center:
                    target_info = f" | Target: {ctx.target_center} | Error: X={ctx.percent_error_x:+d}, Y={ctx.percent_error_y:+d}"
                print(f"[CameraManager] FPS: {self._fps:.1f} | State: {status}{target_info}")

            profiler.start("callbacks")
            for cb in self._result_callbacks:
                try:
                    cb(all_results)
                except Exception as e:
                    print(f"Error in result callback: {e}")
            profiler.stop("callbacks")

            profiler.stop("total")
            profiler.end_frame()

    def _handle_detection(self, all_results: Dict):
        ctx = self.state_machine.context

        for camera_id, results in all_results.items():
            circle_result = results.get("circle_detect")
            if circle_result and circle_result.success:
                data = circle_result.result_data
                if data:
                    ctx.target_found = data.get("target_found", False)
                    ctx.target_center = data.get("target", {}).center_coordinates if data.get("target") else None
                    ctx.percent_error_x = data.get("percent_error_x", 0)
                    ctx.percent_error_y = data.get("percent_error_y", 0)
                    ctx.is_quad_detected = data.get("is_quad_detected", False)
                    ctx.target_distance_mm = data.get("target_distance_mm", None)
                    ctx.is_uv_spot_detected = data.get("is_uv_spot_detected", False)
                    ctx.confidence = 1.0 if ctx.target_found else 0.0

                    if ctx.target_found:
                        ctx.consecutive_detected_frames += 1
                        ctx.consecutive_lost_frames = 0
                    else:
                        ctx.consecutive_lost_frames += 1
                        ctx.consecutive_detected_frames = 0
                        #ctx.percent_error_x = 0
                        #ctx.percent_error_y = 0

                    self._send_error_frame(ctx)

                    if self.state_machine.is_searching() and ctx.target_found:
                        self.state_machine.trigger(VisualStateMachine.Events.TARGET_FOUND)
                    elif self.state_machine.is_tracking() and ctx.consecutive_lost_frames >= 3:
                        self.state_machine.trigger(VisualStateMachine.Events.TARGET_LOST)
                break

    def _send_error_frame(self, ctx):
        """
        Send error frame to STM32 using orange_send protocol.

        Frame format: AA BB + state(int32) + deta_x(int32) + deta_y(int32) + distance(float32) + EE
        - state: 根据当前状态机状态映射到枚举值
        - deta_x: percent_error_x
        - deta_y: percent_error_y
        - distance: target_distance_mm (mm)
        """
        # 状态映射
        if self.state_machine.is_idle():
            state = ORANGE_STATE_IDLE
        elif self.state_machine.is_searching():
            state = ORANGE_STATE_SEARCH
        elif self.state_machine.is_tracking():
            state = ORANGE_STATE_TRACKING
        elif self.state_machine.is_recovering():
            state = ORANGE_STATE_RECOVERY
        else:  # FAIL
            state = ORANGE_STATE_FAIL

        # 获取距离，如果未检测到则为 0.0
        distance_mm = ctx.target_distance_mm if ctx.target_distance_mm is not None else 0.0

        send_orange_frame(state, ctx.percent_error_x, ctx.percent_error_y, distance_mm)

    def process_all(self) -> Tuple[Optional[np.ndarray], Dict[str, Dict[str, VisionResult]]]:
        frames = []
        all_results: Dict[str, Dict[str, VisionResult]] = {}
        camera_ids = []

        for cam in self.cameras.values():
            if not cam.enabled:
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
                frames.append(frame)
                camera_ids.append(cam.camera_id)
                all_results[cam.camera_id] = {}
                continue

            frame, results = cam.process_frame(fps=self._fps)
            if frame is None:
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
            frames.append(frame)
            camera_ids.append(cam.camera_id)
            all_results[cam.camera_id] = results

        if not frames:
            return None, all_results

        if self.frame_composer is None:
            self.frame_composer = FrameComposer(
                layout="grid",
                output_size=(1280, 720),
            )

        composed = self.frame_composer.compose(frames, camera_ids)
        return composed, all_results

    def add_result_callback(self, callback: Callable[[Dict], None]):
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]):
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def start_streaming(self, rtmp_url: str = None):
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
        if self.ffmpeg_pusher:
            self.ffmpeg_pusher.close_sync()
            self.ffmpeg_pusher = None

    def release(self):
        self.stop()
        self.stop_streaming()

        for camera in self.cameras.values():
            camera.release()

        self.cameras.clear()

    def __del__(self):
        self.release()
