from typing import TYPE_CHECKING

import numpy as np

from .base import AbstractModelHandler

if TYPE_CHECKING:
    from hal.interface import AIInference, Detection


class ModelHandlerRegistry:
    _registry: dict[str, type[AbstractModelHandler]] = {}

    @classmethod
    def register(cls, model_type: str):
        def wrapper(handler_cls: type[AbstractModelHandler]):
            cls._registry[model_type] = handler_cls
            return handler_cls
        return wrapper

    @classmethod
    def get(cls, model_type: str, ai) -> AbstractModelHandler:
        handler_cls = cls._registry.get(model_type)
        if handler_cls is None:
            handler_cls = cls._registry.get("default")
        if handler_cls is None:
            return _NullHandler(ai)
        return handler_cls(ai)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


class _NullHandler(AbstractModelHandler):
    def draw(
        self, frame: np.ndarray, detections: list
    ) -> np.ndarray:
        return frame
