from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from framework.hal.interface import AIInference, Camera

from .task_manager import TaskManager, Task
from .processors.base import VisionResult
from .processors.registry import get_processor


class PipelineCamera:
    _PROCESSOR_REGISTRY = get_processor

    def __init__(
        self,
        pipeline_id: str,
        camera: Camera,
        task_configs: List[dict],
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        sensor_height_mm: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480,
        ai: Optional[AIInference] = None,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.camera = camera
        self.task_manager = TaskManager()
        self.focal_calculator = None

        self._last_frame: Optional[np.ndarray] = None
        self._last_results: Dict[str, VisionResult] = {}
        self._ai: Optional[AIInference] = ai

        self._init_focal_calculator(
            focal_length_mm, sensor_width_mm, sensor_height_mm,
            image_width, image_height,
        )
        self._setup_tasks(task_configs)

    @property
    def last_results(self) -> Dict[str, VisionResult]:
        return self._last_results

    def _init_focal_calculator(
        self,
        focal_length_mm: Optional[float],
        sensor_width_mm: Optional[float],
        sensor_height_mm: Optional[float],
        image_width: int,
        image_height: int,
    ) -> None:
        if focal_length_mm and sensor_width_mm and sensor_height_mm:
            from utils.focal_distance_util import CameraIntrinsics, FocalDistanceCalculator
            intrinsics = CameraIntrinsics(
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                image_width=image_width,
                image_height=image_height,
            )
            self.focal_calculator = FocalDistanceCalculator(intrinsics=intrinsics)

    def _setup_tasks(self, task_configs: List[dict]) -> None:
        for cfg in task_configs:
            task_name = cfg.get("name", "")
            task_type = cfg.get("type", "")
            task_enabled = cfg.get("enabled", True)
            try:
                processor_cls = get_processor(task_type)
            except ValueError:
                continue
            processor = processor_cls(task_name)
            if self._ai is not None and hasattr(processor, "set_ai"):
                processor.set_ai(self._ai)
            task = Task(task_name, processor, task_enabled)
            self.task_manager.register_task(task)

    def get_distance_to_target(self, real_size_mm: float, pixel_size: float) -> Optional[float]:
        if self.focal_calculator:
            return self.focal_calculator.calculate_distance(real_size_mm, pixel_size)
        return None

    def get_camera_coords(self, pixel_x: float, pixel_y: float, distance_mm: float) -> Optional[tuple]:
        if self.focal_calculator:
            return self.focal_calculator.pixel_to_camera_coords(
                pixel_x, pixel_y, distance_mm,
            )
        return None

    def process_frame(self, fps: float = 0.0) -> Tuple[Optional[np.ndarray], Dict[str, VisionResult]]:
        frame = self.camera.read()
        if frame is None:
            frame = self._last_frame
        else:
            self._last_frame = frame

        if frame is None:
            return None, {}

        env_context = {
            "fps": fps,
            "focal_calculator": self.focal_calculator,
            "camera": self.camera,
        }
        processed_frame, task_results = self.task_manager.run_tasks_serial(
            frame, context=env_context,
        )
        self._last_results = task_results
        return processed_frame, task_results

    def enable_task(self, task_name: str) -> bool:
        return self.task_manager.enable_task(task_name)

    def disable_task(self, task_name: str) -> bool:
        return self.task_manager.disable_task(task_name)

    def get_task(self, task_name: str):
        return self.task_manager.get_task(task_name)
