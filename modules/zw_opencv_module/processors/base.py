# -*- coding: utf-8 -*-
"""Define the processor base class, result container, and tracking protocol.

A Processor implementing set_target_color(color: Color) satisfies
ColorTrackable automatically. Use isinstance(processor, ColorTrackable) to
test for it; @runtime_checkable only checks that the method exists, not its
signature, so implementations must keep the signature correct.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from ..models.color import Color
import numpy as np


class VisionResult:
    """Hold the outcome of one vision task."""

    def __init__(
        self,
        task_name: str,
        result_data: Any = None,
        success: bool = False,
        error_message: str = "",
        error_code: str = "",
    ):
        self.task_name = task_name
        self.result_data = result_data
        self.success = success
        self.error_message = error_message
        # Stable failure identifier (see the E_* constants in
        # framework/hal/interface/ai.py); empty when the task succeeded.
        self.error_code = error_code

    def __repr__(self):
        return f"VisionResult(task={self.task_name}, success={self.success}, data={self.result_data})"


class Processor(ABC):
    """Abstract base class for vision processors."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._getters: dict[str, Callable[[], Any]] = {}

    @abstractmethod
    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """Process one frame and return its result.

        Args:
            context: May carry results from upstream tasks.
        """
        pass

    def draw_result(
        self, frame: np.ndarray, result: VisionResult
    ) -> np.ndarray:
        """Draw the result onto the frame.

        The default implementation leaves the frame unchanged.
        """
        return frame

    def register_getter(self, key: str, getter: Callable[[], Any]) -> None:
        self._getters[key] = getter

    def get(self, key: str) -> Any:
        getter = self._getters.get(key)
        return getter() if getter else None

    def clear_getters(self) -> None:
        self._getters.clear()

    def release(self) -> None:
        """Release resources held by the processor, such as models or detectors.

        Subclasses override this when they own such resources.
        """
        pass

@runtime_checkable
class ColorTrackable(Protocol):
    @abstractmethod
    def set_target_color(self,color:Color) -> None: ...