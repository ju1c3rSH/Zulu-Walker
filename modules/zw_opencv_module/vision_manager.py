from __future__ import annotations

import logging
import os
import time
import traceback
from collections import deque
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference

from .frame_composer import FrameComposer
from .ffmpeg_pusher import FFmpegPusher
from .pipeline_camera import PipelineCamera
from .performance import profiler

profiler._enabled = False

_module_dir = os.path.dirname(__file__)


class VisionManager:
    def __init__(self, camera_hub: CameraHub, config_path: str = None, ai: Optional[AIInference] = None) -> None:
        self._hub = camera_hub
        self._config_path = config_path or os.path.join(
            _module_dir, "config", "vision_config.yaml"
        )
        self._pipelines: Dict[str, PipelineCamera] = {}
        self._ai = ai
        self.frame_composer: Optional[FrameComposer] = None
        self.ffmpeg_pusher: Optional[FFmpegPusher] = None
        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []
        self._event_bus = None

        self._fps_data: Dict[str, dict] = {}

        self._composed_frame = None
        self._any_fresh = False
        self._pending_results: deque = deque(maxlen=5)

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def start(self) -> None:
        if self._running:
            return

        if not os.path.exists(self._config_path):
            return

        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)

        pipelines = cfg.get("pipelines", [])
        for pipe_cfg in pipelines:
            pipeline_id = pipe_cfg.get("pipeline_id", "")
            camera_id = pipe_cfg.get("camera_id", "")
            if not pipeline_id or not camera_id:
                continue

            cam = self._hub.get(camera_id)
            if cam is None:
                continue

            focal_length_mm = getattr(cam, 'focal_length_mm', None)
            sensor_width_mm = getattr(cam, 'sensor_width_mm', None)
            sensor_height_mm = getattr(cam, 'sensor_height_mm', None)
            if focal_length_mm is None:
                logger.debug(
                    "Camera '%s' has no intrinsics; distance calculation disabled",
                    camera_id,
                )

            pipe = PipelineCamera(
                pipeline_id=pipeline_id,
                camera=cam,
                task_configs=pipe_cfg.get("tasks", []),
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                image_width=pipe_cfg.get("width", 640),
                image_height=pipe_cfg.get("height", 480),
                ai=self._ai,
            )
            self._pipelines[pipeline_id] = pipe

        self.frame_composer = FrameComposer(
            layout="grid",
            output_size=(640, 480),
        )

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _process_loop(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("vision_processing")

        while self._running:
            try:
                profiler.start("total")
                composed, all_results, any_fresh = self.process_all()
                self._composed_frame = composed
                self._any_fresh = any_fresh

                if self.ffmpeg_pusher and composed is not None:
                    self.ffmpeg_pusher.push_frame_sync(composed)

                for cb in self._result_callbacks:
                    try:
                        cb(all_results)
                    except Exception as e:
                        pass

                if self._event_bus:
                    self._pending_results.append(all_results)

                profiler.stop("total")
                if any_fresh:
                    profiler.end_frame()
                else:
                    time.sleep(0.001)
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

    def process_all(
        self,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Dict], bool]:
        frames = []
        all_results: Dict[str, Dict] = {}
        pipeline_ids = []
        fps_values = []
        any_fresh = False

        for pid, pipe in list(self._pipelines.items()):
            cur_fps = self._fps_data.get(pid, {}).get("fps", 0.0)
            frame, results = pipe.process_frame(fps=cur_fps)

            if frame is not None:
                any_fresh = True
                d = self._fps_data.setdefault(
                    pid, {"count": 0, "start": time.time(), "fps": 0.0}
                )
                d["count"] += 1
                elapsed = time.time() - d["start"]
                if elapsed >= 1.0:
                    d["fps"] = d["count"] / elapsed
                    d["count"] = 0
                    d["start"] = time.time()
            else:
                frame = self._make_placeholder()

            frames.append(frame)
            pipeline_ids.append(pid)
            all_results[pid] = results
            fps_values.append(self._fps_data.get(pid, {}).get("fps", 0.0))

        if not frames:
            return None, all_results, False

        composed = self.frame_composer.compose(frames, pipeline_ids, fps_list=fps_values)
        return composed, all_results, any_fresh

    def compose_frame(self) -> Optional[np.ndarray]:
        return self._composed_frame

    def drain_results(self):
        results = []
        while self._pending_results:
            results.append(self._pending_results.popleft())
        return results

    def release_pipeline(self, pipeline_id: str) -> None:
        pipe = self._pipelines.pop(pipeline_id, None)
        if pipe is not None and hasattr(pipe.camera, 'release'):
            pipe.camera.release()

    def _make_placeholder(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def enable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.enable_task(task_name)
        return False

    def disable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.disable_task(task_name)
        return False

    def set_processor_target(self, pipeline_id: str, task_name: str, color) -> None:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            task = pipe.get_task(task_name)
            if task and hasattr(task.processor, "set_target_color"):
                task.processor.set_target_color(color)

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineCamera]:
        return self._pipelines.get(pipeline_id)

    def get_all_results(self) -> Dict[str, Dict]:
        return {
            pid: pipe.last_results
            for pid, pipe in self._pipelines.items()
        }

    def add_result_callback(self, callback: Callable[[Dict], None]) -> None:
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]) -> None:
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def stop(self) -> None:
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def release(self) -> None:
        self.stop()
        if self.ffmpeg_pusher:
            self.ffmpeg_pusher.close_sync()
            self.ffmpeg_pusher = None
        self._pipelines.clear()


class _LegacyCameraManagerShim:
    def __init__(self, vision_manager: VisionManager) -> None:
        self._vm = vision_manager

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.enable_task(camera_id, task_name)

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.disable_task(camera_id, task_name)

    def get_all_results(self) -> Dict:
        return self._vm.get_all_results()

    def add_result_callback(self, callback) -> None:
        self._vm.add_result_callback(callback)

    def remove_result_callback(self, callback) -> None:
        self._vm.remove_result_callback(callback)

    @property
    def cameras(self) -> Dict:
        return {}


class CameraManager:
    pass
