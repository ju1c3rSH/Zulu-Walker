# -*- coding: utf-8 -*-
from typing import Dict, List, Callable, Optional, Tuple
from collections import OrderedDict
import numpy as np

from .processors.base import VisionResult, Processor
from .performance import profiler


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
        self, frame: np.ndarray, all_results: dict = None
    ) -> Tuple[np.ndarray, Dict[str, VisionResult]]:
        """
        串行执行所有启用的Task

        前一任务的结果可以传入一个可被包装的all_results变量供后续任务使用，虽然不传也没什么影响

        Args:
            frame: 输入图像帧
            all_results: 所有任务的结果，可包含 'fps'、'focal_calculator' 等

        Returns:
            Tuple[np.ndarray, Dict[str, VisionResult]]: 处理后的帧和所有结果
        """
        if frame is None:
            return None, {}

        if all_results is None:
            all_results = {}

        processed_frame = frame  # 显示用帧，累积绘制

        for task in self.tasks.values():
            if task.enabled:
                timer_name = f"task_{task.name}_process"
                profiler.start(timer_name)
                result = task.execute(frame, all_results)  # 始终用原始帧处理
                profiler.stop(timer_name)

                all_results[task.name] = result

                # 在帧上绘制结果
                if result is not None and self.draw_enabled:
                    draw_timer_name = f"task_{task.name}_draw"
                    profiler.start(draw_timer_name)
                    processed_frame = task.processor.draw_result(processed_frame, result)
                    profiler.stop(draw_timer_name)
                #TODO 这种写法可能不适合串行任务
                # 触发回调
                for callback in self.result_callbacks:
                    try:
                        callback(task.name, result)
                    except Exception as e:
                        print(f"Error in result callback: {e}")

        return processed_frame, all_results

    def get_all_results(self) -> Dict[str, Optional[VisionResult]]:
        """获取所有任务的最后结果"""
        return {name: task.last_result for name, task in self.tasks.items()}

    def clear_all_results(self):
        """清除所有任务的最后结果"""
        for task in self.tasks.values():
            task._last_result = None
