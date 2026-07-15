# -*- coding: utf-8 -*-
from typing import Dict, List, Callable, Optional, Tuple
from collections import OrderedDict
import numpy as np

from .processors.base import VisionResult, Processor
from .performance import profiler
from utils.log_util import log_print



class Task:
    """任务类，封装Processor并提供生命周期管理"""

    def __init__(self, name: str, processor: Processor, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.processor = processor
        self._last_result: Optional[VisionResult] = None
        self._execution_count = 0

    def enable(self):
        """启用任务"""
        self.enabled = True

    def disable(self):
        """禁用任务"""
        self.enabled = False
        self.processor.release()

    def execute(self, frame: np.ndarray, context: dict = None) -> Optional[VisionResult]:
        """
        执行任务

        Args:
            frame: 输入图像帧
            context: 上下文，可包含前置任务结果

        Returns:
            VisionResult 或 None（如果任务被禁用）
        """
        if not self.enabled:
            return None

        result = self.processor.process(frame, context)
        self._last_result = result
        self._execution_count += 1
        return result

    @property
    def last_result(self) -> Optional[VisionResult]:
        """获取最后一次执行结果"""
        return self._last_result

    @property
    def execution_count(self) -> int:
        """获取执行次数"""
        return self._execution_count


class TaskManager:
    """任务管理器，管理多个Task的串行执行"""

    def __init__(self):
        self.tasks: OrderedDict[str, Task] = OrderedDict()
        self.result_callbacks: List[Callable[[str, VisionResult], None]] = []
        self.draw_enabled: bool = True

    def register_task(self, task: Task):
        """注册任务"""
        self.tasks[task.name] = task

    def unregister_task(self, name: str) -> Optional[Task]:
        """注销任务"""
        return self.tasks.pop(name, None)

    def get_task(self, name: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(name)

    def enable_task(self, name: str) -> bool:
        """启用指定任务"""
        task = self.tasks.get(name)
        if task:
            task.enable()
            return True
        return False

    def disable_task(self, name: str) -> bool:
        """禁用指定任务"""
        task = self.tasks.get(name)
        if task:
            task.disable()
            return True
        return False

    def add_result_callback(self, callback: Callable[[str, VisionResult], None]):
        """添加结果回调"""
        self.result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[str, VisionResult], None]):
        """移除结果回调"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)

    def run_tasks_serial(
        self, frame: np.ndarray, context: dict = None
    ) -> Tuple[np.ndarray, Dict[str, VisionResult]]:
        """
        串行执行所有启用的Task。

        每个 task 的 processor 收到该 task 独立的完整上下文（环境变量 + 前置 task 结果）。
        task 输出的 VisionResult 存入独立的 task_results dict 返回，
        不入参 context dict。

        Args:
            frame: 输入图像帧
            context: 只读环境上下文，如 {'fps': ..., 'focal_calculator': ...}

        Returns:
            Tuple[np.ndarray, Dict[str, VisionResult]]: 处理后的帧和 {task_name: VisionResult}
        """
        if frame is None:
            return None, {}

        context = context if context is not None else {}
        processed_frame = frame
        task_results: Dict[str, VisionResult] = {}

        for task in self.tasks.values():
            if not task.enabled:
                continue

            task_context = {**context, **task_results}

            profiler.start(f"task_{task.name}_process")
            result = task.execute(frame, task_context)
            profiler.stop(f"task_{task.name}_process")

            if result is not None:
                task_results[task.name] = result

            if result is not None and self.draw_enabled:
                profiler.start(f"task_{task.name}_draw")
                processed_frame = task.processor.draw_result(processed_frame, result)
                profiler.stop(f"task_{task.name}_draw")

            for callback in self.result_callbacks:
                try:
                    callback(task.name, result)
                except Exception as e:
                    log_print(f"Error in result callback: {e}")

        return processed_frame, task_results

    def get_all_results(self) -> Dict[str, Optional[VisionResult]]:
        """获取所有任务的最后结果"""
        return {name: task.last_result for name, task in self.tasks.items()}

    def clear_all_results(self):
        """清除所有任务的最后结果"""
        for task in self.tasks.values():
            task._last_result = None

    def release(self):
        for task in self.tasks.values():
            task.processor.release()
        self.tasks.clear()
