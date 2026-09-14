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
from utils.log_util import log_print

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
        self._hud = None
        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []
        self._event_bus = None

        self._fps_data: Dict[str, dict] = {}

        self._any_fresh = False
        self._pending_results: deque = deque(maxlen=5)
        self._display_frame = None

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._capture_sink: callable = None
        self._capture_seq: int = 0
        self._CAPTURE_EVERY_N: int = 4

        self._exit_icon = None
        self._exit_icon_size: int = 0
        self._exit_icon_margin: int = 0

        # 调参页等独占相机的场景: 暂停采集处理, 让出相机成为唯一读取者
        self._paused = False

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def set_capture_sink(self, sink: callable) -> None:
        self._capture_sink = sink

    def set_exit_icon(self, icon, icon_size: int = 48, margin: int = 12) -> None:
        self._exit_icon = icon
        self._exit_icon_size = icon_size
        self._exit_icon_margin = margin

    def start(self) -> None:
        if self._running:
            return

        if not os.path.exists(self._config_path):
            return

        try:
            with open(self._config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            logger.error("Failed to load vision config: %s", self._config_path)
            return

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

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _process_loop(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("vision_processing")

        while self._running:
            try:
                self._wdt_feed()
                self._wdt_count += 1
                if self._wdt_count % 100 == 0:
                    log_print(f"[WDT] viz feed #{self._wdt_count}")

                if self._paused:
                    time.sleep(0.01)
                    continue

                profiler.start("total")
                _, all_results, any_fresh = self.process_all()

                self._update_display_frame()

                if self._event_bus:
                    self._pending_results.append(all_results)

                for cb in self._result_callbacks:
                    try:
                        cb(all_results)
                    except Exception as e:
                        pass

                profiler.stop("total")
                time.sleep(0)
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
        all_results: Dict[str, Dict] = {}
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

            all_results[pid] = results

        return None, all_results, any_fresh

    def get_display_frame(self):
        return self._display_frame

    def _update_display_frame(self) -> None:
        # 显示不依赖 AI 模型加载（色块任务无模型也须可显示）
        for pid, pipe in list(self._pipelines.items()):
            raw_img = getattr(pipe.camera, "last_raw", None)
            if raw_img is None:
                continue
            fps = self.get_pipeline_fps(pid)
            if self._hud is not None:
                self._hud.draw(raw_img, fps, pipe.last_results)
            self._draw_exit_icon(raw_img)
            self._display_frame = raw_img
            self._capture_seq += 1
            if self._capture_sink is not None and self._capture_seq % self._CAPTURE_EVERY_N == 0:
                self._capture_sink(raw_img)
            return

    def set_hud(self, hud) -> None:
        """注入显示 HUD（状态栏 + 检测叠加）; None = 只画退出图标。"""
        self._hud = hud

    def set_paused(self, paused: bool) -> None:
        """暂停/恢复采集处理（调参页独占相机时置 True）。
        暂停期间不读相机、不 produce 新结果, WDT 照常喂;
        结果 >1s 陈旧由 coordinator 的归零心跳逻辑兜底。"""
        self._paused = bool(paused)

    def get_pipeline_camera(self):
        """首个 pipeline 的 HAL 相机（供调参页等独占式页面读帧）。"""
        for _pid, pipe in self._pipelines.items():
            return pipe.camera
        return None

    def _draw_exit_icon(self, img) -> None:
        if self._exit_icon is None:
            return
        try:
            import maix.image as _mi
            h = img.height()
            by = h - self._exit_icon_size - 8
            bx = self._exit_icon_margin
            img.draw_rect(bx - 2, by - 2, self._exit_icon_size + 4,
                          self._exit_icon_size + 4, color=_mi.COLOR_BLACK, thickness=-1)
            img.draw_image(bx, by, self._exit_icon)
        except Exception:
            pass

    def drain_results(self):
        results = []
        while self._pending_results:
            results.append(self._pending_results.popleft())
        return results

    def get_pipeline_fps(self, pipeline_id: str) -> float:
        fd = self._fps_data.get(pipeline_id)
        return fd.get("fps", 0.0) if fd else 0.0

    def release_pipeline(self, pipeline_id: str) -> None:
        pipe = self._pipelines.pop(pipeline_id, None)
        if pipe is not None and hasattr(pipe.camera, 'release'):
            pipe.camera.release()

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
        self._pipelines.clear()


class _LegacyCameraManagerShim:
    def __init__(self, vision_manager: VisionManager) -> None:
        self._vm = vision_manager

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.enable_task(camera_id, task_name)

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.disable_task(camera_id, task_name)

    def get_all_results(self) -> dict:
        return self._vm.get_all_results()

    def add_result_callback(self, callback) -> None:
        self._vm.add_result_callback(callback)

    def remove_result_callback(self, callback) -> None:
        self._vm.remove_result_callback(callback)

    @property
    def cameras(self) -> dict:
        return {}
