from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hal.interface.ai import Detection

logger = logging.getLogger(__name__)


class MockAI:
    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}
        self._active_name: str = ""
        self._model_type: str = ""

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def active_model(self) -> str:
        return self._active_name

    @property
    def model_type(self) -> str:
        if self._active_name and self._active_name in self._registry:
            return self._registry[self._active_name].get("type", "")
        return ""

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        self._registry[nick_name] = dict(path=model_path, type=model_type, kwargs=kwargs)
        logger.info("MockAI: registered model '%s' (%s, type=%s)", nick_name, model_path, model_type)
        if not self._active_name:
            self.switch(nick_name)
        return True

    def remove(self, nick_name: str) -> None:
        self._registry.pop(nick_name, None)
        if self._active_name == nick_name:
            self._active_name = ""

    def switch(self, nick_name: str) -> bool:
        if nick_name not in self._registry:
            logger.warning("MockAI: model '%s' not registered", nick_name)
            return False
        self._active_name = nick_name
        logger.info("MockAI: switched to '%s' (mock)", nick_name)
        return True

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        return self.add("default", model_path, model_type=model_type, **kwargs)

    def unload(self) -> None:
        self.remove(self._active_name)

    @property
    def loaded(self) -> bool:
        return bool(self._active_name)

    @property
    def input_width(self) -> int:
        return 0

    @property
    def input_height(self) -> int:
        return 0

    @property
    def labels(self) -> list[str]:
        return []

    @property
    def model_path(self) -> str:
        if self._active_name and self._active_name in self._registry:
            return self._registry[self._active_name]["path"]
        return ""

    def detect(self, frame: np.ndarray, **kwargs) -> list[Detection]:
        logger.warning("MockAI: detect() called but no NPU available")
        return []

    def classify(self, frame: np.ndarray, **kwargs) -> list[tuple[int, float]]:
        logger.warning("MockAI: classify() called but no NPU available")
        return []

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()
