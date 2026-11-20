from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from framework.hal.interface import AIInference, Camera
from utils.boot_timing import thread_cpu as _thread_cpu

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
        # 分项耗时(ms): 墙上(read/convert)含等硬件与 GIL, CPU(read_cpu/tasks_cpu)是真消耗。
        # 两者差大 = 在等硬件; 接近 = 真在算。
        self.stage_ms: Dict[str, float] = {
            "read": 0.0, "read_cpu": 0.0, "convert": 0.0,
            "tasks": 0.0, "tasks_cpu": 0.0,
        }

        self._init_focal_calculator(
            focal_length_mm, sensor_width_mm, sensor_height_mm,
            image_width, image_height,
        )
        self._setup_tasks(task_configs)
        self._enable_lazy_convert()

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
                # 注册表缺失该处理器 = 流水线静默空转（全零输出）, 必须显性暴露
                import logging
                logging.getLogger(__name__).error(
                    "Task '%s': unknown processor type '%s' (module not imported?)",
                    task_name, task_type)
                continue
            processor = processor_cls(task_name)
            if self._ai is not None and hasattr(processor, "set_ai"):
                processor.set_ai(self._ai)
            task = Task(task_name, processor, task_enabled)
            self.task_manager.register_task(task)

    def _enable_lazy_convert(self) -> None:
        """让相机走"延迟转换"：不再每帧急切 image2cv（921KB 拷贝, 实测 9~16ms）。

        为什么安全: 色块任务(EXPLOSIVE/TARGET) 走 C++ find_blobs 直接吃
        `camera.last_raw`, 人质任务走 `_raw=True` 也是直接吃 maix Image ——
        三个任务在设备 + maix 可用时都不需要那份 BGR numpy。只有 cv2 回退
        路径(maix 不可用)与调参页需要, 而 LazyFrame 会在真正切片/传库时才转一次。

        平台无 `set_lazy_convert`（linux/mock, 它们本就返回真实 numpy）时静默跳过
        —— 那些平台没有"白做的拷贝"这个问题。
        """
        setter = getattr(self.camera, "set_lazy_convert", None)
        if setter is None:
            return
        try:
            setter(True)
        except Exception:
            # 开关失败绝不能影响流水线启动: 退回急切转换即改前行为。
            import logging
            logging.getLogger(__name__).warning(
                "set_lazy_convert 失败, 保持急切 image2cv: %s", exc_info=True)

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

    def _camera_stage(self, fallback_ms: float) -> Tuple[float, float]:
        """(read_ms, convert_ms) —— 优先用相机 HAL 自己拆好的分项。

        HAL 没提供 `stage_ms`（linux/mock/旧实现）时退回"整段都算取帧、转换为 0"，
        判断跟拆分前一样，不改变任何判断条件。
        """
        st = getattr(self.camera, "stage_ms", None)
        if isinstance(st, dict):
            return float(st.get("read", 0.0)), float(st.get("convert", 0.0))
        return fallback_ms, 0.0

    def process_frame(self, fps: float = 0.0) -> Tuple[Optional[np.ndarray], Dict[str, VisionResult]]:
        _t0, _c0 = time.perf_counter(), _thread_cpu()
        raw_img = self.camera.read_raw()
        _t1, _c1 = time.perf_counter(), _thread_cpu()
        read_ms, conv_ms = self._camera_stage((_t1 - _t0) * 1000.0)
        read_cpu = (_c1 - _c0) * 1000.0

        if raw_img is None:
            # 相机停帧: 不复用旧帧加工、不产出结果。
            # 显示侧仍用 camera.last_raw 画旧画面；coordinator 侧无新结果，
            # _latest_task_ts 不更新，1s 后归零心跳补上（零值不断流）。
            self.stage_ms = {"read": read_ms, "read_cpu": read_cpu, "convert": conv_ms,
                             "tasks": 0.0, "tasks_cpu": 0.0}
            return None, {}

        frame = raw_img
        self._last_frame = frame

        env_context = {
            "fps": fps,
            "focal_calculator": self.focal_calculator,
            "camera": self.camera,
        }
        processed_frame, task_results = self.task_manager.run_tasks_serial(
            frame, context=env_context,
        )
        _t2, _c2 = time.perf_counter(), _thread_cpu()
        self.stage_ms = {"read": read_ms, "read_cpu": read_cpu, "convert": conv_ms,
                         "tasks": (_t2 - _t1) * 1000.0,
                         "tasks_cpu": (_c2 - _c1) * 1000.0}
        self._last_results = task_results
        return processed_frame, task_results

    def enable_task(self, task_name: str) -> bool:
        ok = self.task_manager.enable_task(task_name)
        if ok and self._ai is not None:
            # Task.disable() 会调 processor.release()（GksTaskProcessor 会清空
            # 借用的 AI 句柄）；重启用时把共享 AI 重新注入，否则 RESCUE 在
            # 一次 STOP 后永久走“模型缺失零值”分支。
            task = self.task_manager.get_task(task_name)
            if task is not None and hasattr(task.processor, "set_ai"):
                task.processor.set_ai(self._ai)
        return ok

    def disable_task(self, task_name: str) -> bool:
        return self.task_manager.disable_task(task_name)

    def get_task(self, task_name: str):
        return self.task_manager.get_task(task_name)
