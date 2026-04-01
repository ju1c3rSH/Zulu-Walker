# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class VisionResult:
    """视觉任务结果"""

    def __init__(
        self,
        task_name: str,
        result_data: Any = None,
        success: bool = False,
        error_message: str = "",
    ):
        self.task_name = task_name
        self.result_data = result_data
        self.success = success
        self.error_message = error_message

    def __repr__(self):
        return f"VisionResult(task={self.task_name}, success={self.success}, data={self.result_data})"


class Processor(ABC):
    """处理器抽象基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """
        处理帧

        Args:
            frame: 输入图像帧
            context: 上下文，可包含前置任务结果

        Returns:
            VisionResult: 处理结果
        """
        pass

    def draw_result(
        self, frame: np.ndarray, result: VisionResult
    ) -> np.ndarray:
        """
        在帧上绘制结果

        Args:
            frame: 输入图像帧
            result: 处理结果

        Returns:
            np.ndarray: 绘制后的帧
        """
        return frame
