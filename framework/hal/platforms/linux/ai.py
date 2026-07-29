from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from framework.hal.interface.ai import Detection

logger = logging.getLogger(__name__)


class LinuxAI:
    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}
        self._active_name: str = ""
        self._model_type: str = ""
        self._model_path: str = ""

    # ------------------------------------------------------------------ #
    #  Registry API
    # ------------------------------------------------------------------ #

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def active_model(self) -> str:
        return self._active_name

    @property
    def model_type(self) -> str:
        return self._model_type

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        if nick_name in self._registry:
            logger.warning("Model '%s' already registered, overwriting", nick_name)
        self._registry[nick_name] = {
            "path": model_path,
            "type": model_type,
            "kwargs": kwargs,
        }
        logger.info("Registered model '%s' -> %s (type=%s)", nick_name, model_path, model_type)
        return True

    def remove(self, nick_name: str) -> None:
        if nick_name not in self._registry:
            logger.warning("Attempt to remove unknown model '%s'", nick_name)
            return
        if self._active_name == nick_name:
            self.unload()
        del self._registry[nick_name]
        logger.info("Removed model '%s'", nick_name)

    def switch(self, nick_name: str) -> bool:
        if nick_name not in self._registry:
            logger.error("Cannot switch to unknown model '%s'", nick_name)
            return False
        if self._active_name == nick_name:
            return True
        logger.warning("LinuxAI: model switching is a no-op (no NPU available)")
        self._active_name = nick_name
        info = self._registry[nick_name]
        self._model_path = info["path"]
        self._model_type = info.get("type", "")
        return True

    # ------------------------------------------------------------------ #
    #  Convenience aliases
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        self.add("default", model_path, model_type=model_type, **kwargs)
        return self.switch("default")

    def unload(self) -> None:
        self._active_name = ""
        self._model_path = ""

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        return False

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
        return self._model_path

    # ------------------------------------------------------------------ #
    #  Inference (stub)
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray, **kwargs) -> list[Detection]:
        logger.warning("LinuxAI.detect() called but no NPU is available")
        return []

    def segment(self, frame: np.ndarray, **kwargs) -> list:
        logger.warning("LinuxAI.segment() called but no NPU is available")
        return []

    def classify(self, frame: np.ndarray, **kwargs) -> list[tuple[int, float]]:
        logger.warning("LinuxAI.classify() called but no NPU is available")
        return []

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()
