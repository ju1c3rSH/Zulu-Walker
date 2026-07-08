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
from queue import Queue
import numpy as np

from .camera_stream import CameraStream
from .task_manager import TaskManager, Task
from .frame_composer import FrameComposer
from .ffmpeg_pusher import FFmpegPusher
from .processors.base import VisionResult
from .processors.circle_target_processor import CircleTargetProcessor
from .processors.qr_processor import QRCodeProcessor
from .processors.cargo_processor import TrackCargoProcessor
from .processors.ring_track_processor import RingTrackProcessor
from .processors.ring_discovery_processor import RingDiscoveryProcessor
from .performance import profiler
from .param_utils import (
    load_camera_params, apply_camera_params_to_capture
)
from utils.camera_misc_util import CameraMiscUtil

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
    gaussian_blur_kernel_size: int = 5
    gaussian_blur_sigma: float = 1.5
    camera_stream_queue_size: int = 2


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
        self._last_results: Dict[str, VisionResult] = {}
        self._is_current_frame_gunmu : bool = False #当前是否滚木帧
        # 处理侧诊断计数器
        self._total_frames_polled = 0
        self._empty_queue_count = 0
        self._fresh_frame_count = 0
        self._diag_start_time = time.time()
        # 焦距计算器
        self.focal_calculator = None
        self._setup_stream(config)
        self._setup_tasks(config.tasks)
        self._init_focal_calculator(config)

    def _setup_stream(self, config: CameraConfig):
        try:
            self.stream = CameraStream(config.source, config.width, config.height,
                                       queue_size=config.camera_stream_queue_size)
        except Exception as e:
            if isinstance(config.source, int):
                print(f"[Camera {self.camera_id}] Source {config.source} failed: {e}, searching fallback...")
                cameras = CameraMiscUtil.find_working_cameras()
                if cameras:
                    fallback_idx = cameras[0].index
                    try:
                        self.stream = CameraStream(fallback_idx, config.width, config.height,
                                                   queue_size=config.camera_stream_queue_size)
                        print(f"[Camera {self.camera_id}] Fallback to camera index {fallback_idx} succeeded")
                        self.config.source = fallback_idx
                        return
                    except Exception as e2:
                        print(f"[Camera {self.camera_id}] Fallback index {fallback_idx} also failed: {e2}")
                else:
                    print(f"[Camera {self.camera_id}] No working cameras found")
            else:
                print(f"[Camera {self.camera_id}] String source failed: {e}")
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
        registry = {
            "CircleTargetProcessor": CircleTargetProcessor,
            "QRCodeProcessor": QRCodeProcessor,
            "TrackCargoProcessor": TrackCargoProcessor,
            "RingTrackProcessor": RingTrackProcessor,
            "RingDiscoveryProcessor": RingDiscoveryProcessor,
        }
        cls = registry.get(task_type)
        return cls(name) if cls else None

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

    def process_frame(self, fps: float = 0.0) -> Tuple[Optional[np.ndarray], Dict[str, VisionResult], bool]:
        if not self.enabled:
            return None, {}, False

        self._total_frames_polled += 1
        frame = self.get_frame()
        if frame is not None:
            self._last_frame = frame.copy()  # 保存干净副本，避免累积绘制问题
            self._fresh_frame_count += 1
        elif self._last_frame is not None:
            self._is_current_frame_gunmu = False
            self._empty_queue_count += 1
            # 每 5 秒打印一次空队列率
            elapsed = time.time() - self._diag_start_time
            if elapsed >= 3.0:
                total = self._total_frames_polled
                empty_pct = (self._empty_queue_count / total * 100) if total > 0 else 0
                fresh_fps = self._fresh_frame_count / elapsed
                print(f"[Camera {self.camera_id}] empty: {empty_pct:.0f}% "
                      f"({self._empty_queue_count}/{total}), fresh_fps: {fresh_fps:.1f}")
                self._total_frames_polled = 0
                self._empty_queue_count = 0
                self._fresh_frame_count = 0
                self._diag_start_time = time.time()
                #返回空帧和空context，保持滚木状态，等待下一个新帧到来时重置状态
                return self._last_frame, {}, False
        else:
            self._is_current_frame_gunmu = True
            return None, {}, False


        # 计时：处理阶段
        profiler.start(self.get_task.__name__ + "_processing")
        context = {
            "fps": fps,
            "focal_calculator": self.focal_calculator,
        }
        result = self.task_manager.run_tasks_serial(frame, all_results=context)
        profiler.stop(self.get_task.__name__ + "_processing")

        self._last_frame = result[0]  # 缓存带绘制的帧
        self._last_results = result[1]
        # 0是processed_frame，1是all_results
        return result[0], result[1], True

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

        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []

        # FPS 计算
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        self._fps = 0.0

        # 显示队列（处理线程推帧，主线程消费显示）
        self._display_queue: Queue = Queue(maxsize=1)
        self._last_display_time = 0.0
        self._display_interval = 1.0 / 15  # 15fps display refresh

        self._event_bus = None

    def set_event_bus(self, bus):
        self._event_bus = bus

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
            output_width=system.get("output_width", 320),
            output_height=system.get("output_height", 240),
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
                camera_stream_queue_size=cam_data.get("camera_stream_queue_size", 2),
            )
            self.add_camera(cam_id, cam_config)

        # 绘制仅在需要显示或推流时启用
        draw_enabled = self.config.enable_local_display or self.config.enable_streaming
        for cam in self.cameras.values():
            cam.task_manager.draw_enabled = draw_enabled

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

        self._apply_camera_params()

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _apply_camera_params(self):
        """从 YAML 加载摄像头硬件参数并应用到所有摄像头"""
        yaml_params, user_keys = load_camera_params()
        for camera in self.cameras.values():
            if not (camera.stream and camera.stream.cap):
                continue
            cap = camera.stream.cap
            if user_keys:
                apply_camera_params_to_capture(cap, yaml_params, user_keys)
                print(f"[CameraManager] Applied YAML overrides: {user_keys}")
            else:
                print(f"[CameraManager] No camera_params.yaml, keeping hardware defaults")

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

            profiler.start("process_all")
            composed_frame, all_results, any_fresh = self.process_all()
            profiler.stop("process_all")

            if any_fresh:
                self._fps_frame_count += 1
                elapsed_fps = time.time() - self._fps_start_time
                if elapsed_fps >= 1.0:
                    self._fps = self._fps_frame_count / elapsed_fps
                    self._fps_frame_count = 0
                    self._fps_start_time = time.time()

            if self.config and self.config.enable_streaming:
                if self.ffmpeg_pusher and composed_frame is not None:
                    profiler.start("ffmpeg_push")
                    self.ffmpeg_pusher.push_frame_sync(composed_frame)
                    profiler.stop("ffmpeg_push")

            # 推入显示队列（主线程 loop() 消费）
            if self.config and self.config.enable_local_display and composed_frame is not None:
                now = time.monotonic()
                if now - self._last_display_time >= self._display_interval:
                    self._last_display_time = now
                    if self._display_queue.full():
                        try:
                            self._display_queue.get_nowait()
                        except:
                            pass
                    self._display_queue.put_nowait(composed_frame)

            profiler.start("callbacks")
            for cb in self._result_callbacks:
                try:
                    cb(all_results)
                except Exception as e:
                    print(f"Error in result callback: {e}")
            profiler.stop("callbacks")

            if self._event_bus:
                try:
                    from context.events import FrameResult
                    self._event_bus.publish(FrameResult(all_results))
                except ImportError:
                    pass

            profiler.stop("total")
            if any_fresh:
                profiler.end_frame()
            else:
                profiler.start("idle_wait")
                time.sleep(0.001)
                profiler.stop("idle_wait")

    def process_all(self) -> Tuple[Optional[np.ndarray], Dict[str, Dict[str, VisionResult]], bool]:
        frames = []
        all_results: Dict[str, Dict[str, VisionResult]] = {}
        camera_ids = []
        any_fresh = False

        for cam in self.cameras.values():
            if not cam.enabled:
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
                frames.append(frame)
                camera_ids.append(cam.camera_id)
                all_results[cam.camera_id] = {}
                continue

            frame, cam_all_results, is_fresh = cam.process_frame(fps=self._fps)
            if is_fresh:
                any_fresh = True
            if frame is None:
                frame = np.zeros((cam.config.height, cam.config.width, 3), dtype=np.uint8)
            frames.append(frame)
            camera_ids.append(cam.camera_id)
            all_results[cam.camera_id] = cam_all_results

        if not frames:
            return None, all_results, False

        if self.frame_composer is None:
            self.frame_composer = FrameComposer(
                layout="grid",
                output_size=(640, 480),
            )

        composed = self.frame_composer.compose(frames, camera_ids)
        return composed, all_results, any_fresh

    def add_result_callback(self, callback: Callable[[Dict], None]):
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]):
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def display_frame(self):
        """主线程调用：从显示队列取出帧并显示。返回显示的帧，无帧返回 None。"""
        try:
            frame = self._display_queue.get_nowait()
        except:
            return None
        if self.config and self.config.enable_local_display:
            cv2.imshow("Zulu-Walker Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self._running = False
        return frame

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
