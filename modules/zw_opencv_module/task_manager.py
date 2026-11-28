# -*- coding: utf-8 -*-
from typing import Dict, List, Callable, Optional, Tuple
from collections import OrderedDict
import numpy as np

from .processors.base import VisionResult, Processor
from .performance import profiler
from utils.log_util import log_print



class Task:
    """Wrap a Processor with enable/disable state and execution bookkeeping."""

    def __init__(self, name: str, processor: Processor, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.processor = processor
        self._last_result: Optional[VisionResult] = None
        self._execution_count = 0

    def enable(self):
        """Enable the task."""
        self.enabled = True

    def disable(self):
        """Disable the task and release its processor."""
        self.enabled = False
        self.processor.release()

    def execute(self, frame: np.ndarray, context: dict = None) -> Optional[VisionResult]:
        """Run the processor on a frame and cache the result.

        Args:
            frame: Input image.
            context: Environment context, possibly including earlier results.

        Returns:
            The processor result, or None when the task is disabled.
        """
        if not self.enabled:
            return None

        result = self.processor.process(frame, context)
        self._last_result = result
        self._execution_count += 1
        return result

    @property
    def last_result(self) -> Optional[VisionResult]:
        """Return the most recent execution result."""
        return self._last_result

    @property
    def execution_count(self) -> int:
        """Return the number of times the task has run."""
        return self._execution_count


class TaskManager:
    """Register tasks and run the enabled ones in sequence."""

    def __init__(self):
        self.tasks: OrderedDict[str, Task] = OrderedDict()
        self.result_callbacks: List[Callable[[str, VisionResult], None]] = []
        self.draw_enabled: bool = False

    def register_task(self, task: Task):
        """Register a task under its name."""
        self.tasks[task.name] = task

    def unregister_task(self, name: str) -> Optional[Task]:
        """Unregister and return the named task."""
        return self.tasks.pop(name, None)

    def get_task(self, name: str) -> Optional[Task]:
        """Return the named task, if present."""
        return self.tasks.get(name)

    def enable_task(self, name: str) -> bool:
        """Enable a task by name; return whether it existed."""
        task = self.tasks.get(name)
        if task:
            task.enable()
            return True
        return False

    def disable_task(self, name: str) -> bool:
        """Disable a task by name; return whether it existed."""
        task = self.tasks.get(name)
        if task:
            task.disable()
            return True
        return False

    def add_result_callback(self, callback: Callable[[str, VisionResult], None]):
        """Add a callback invoked with each task result."""
        self.result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[str, VisionResult], None]):
        """Remove a previously added result callback."""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)

    def run_tasks_serial(
        self, frame: np.ndarray, context: dict = None
    ) -> Tuple[np.ndarray, Dict[str, VisionResult]]:
        """Run all enabled tasks in sequence on one frame.

        Each processor receives a context merging the shared environment with
        the results of earlier tasks. Results are collected in a separate dict
        instead of being written back into ``context``.

        Args:
            frame: Input image.
            context: Read-only environment values such as fps and focal calculator.

        Returns:
            The annotated frame and a mapping of task names to results.
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
        """Return the last result of every task."""
        return {name: task.last_result for name, task in self.tasks.items()}

    def clear_all_results(self):
        """Clear the cached result of every task."""
        for task in self.tasks.values():
            task._last_result = None

    def release(self):
        for task in self.tasks.values():
            task.processor.release()
        self.tasks.clear()
