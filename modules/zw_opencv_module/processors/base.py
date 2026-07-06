# -*- coding: utf-8 -*-
"""
Processor 基类和协议定义。

Protocol 规范：
  ColorTrackable — 实现了 set_target_color(color: Color) 的 Processor
    自动满足此协议。使用 isinstance(processor, ColorTrackable) 检测，
    运行时通过 @runtime_checkable 反射检查方法是否存在。
    注意：Protocol 不检查方法签名，请确保实现正确。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from ..models.color import Color
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
        self._getters: dict[str, Callable[[], Any]] = {}

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

    def register_getter(self, key: str, getter: Callable[[], Any]) -> None:
        self._getters[key] = getter

    def get(self, key: str) -> Any:
        getter = self._getters.get(key)
        return getter() if getter else None

    def clear_getters(self) -> None:
        self._getters.clear()

@runtime_checkable
class ColorTrackable(Protocol):
    @abstractmethod
    def set_target_color(self,color:Color) -> None: ...