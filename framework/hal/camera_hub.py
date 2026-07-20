from __future__ import annotations

import importlib
import logging
from typing import Dict, Optional

from framework.hal.interface import Camera

logger = logging.getLogger(__name__)


class CameraHub:
    _instance: Optional["CameraHub"] = None
    _platform_module = None

    def __init__(self, platform: str) -> None:
        self._platform = platform
        self._cameras: Dict[str, Camera] = {}

    @classmethod
    def init_instance(cls, platform: str) -> "CameraHub":
        if cls._instance is not None:
            return cls._instance
        cls._instance = cls(platform)
        cls._platform_module = importlib.import_module(f"framework.hal.platforms.{platform}")
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["CameraHub"]:
        return cls._instance

    def open(
        self,
        camera_id: str,
        source,
        width: int = 640,
        height: int = 480,
        **kwargs,
    ) -> Optional[Camera]:
        if camera_id in self._cameras:
            return self._cameras[camera_id]
        factory = getattr(self._platform_module, "create_camera")
        cam = factory(
            source=source,
            width=width,
            height=height,
            camera_id=camera_id,
            **kwargs,
        )
        if cam is None:
            logger.warning("Camera '%s' could not be created; not registered", camera_id)
            return None
        self._cameras[camera_id] = cam
        return cam

    def get(self, camera_id: str) -> Optional[Camera]:
        return self._cameras.get(camera_id)

    def close(self, camera_id: str) -> None:
        cam = self._cameras.pop(camera_id, None)
        if cam is not None:
            try:
                cam.release()
            except Exception as e:
                logger.error("Error releasing camera %s: %s", camera_id, e)

    def release_all(self) -> None:
        for cid in list(self._cameras.keys()):
            self.close(cid)

    def list_ids(self) -> list[str]:
        return list(self._cameras.keys())
