from __future__ import annotations

from typing import Optional

import numpy as np


class MaixCam2Camera:
    def __init__(self, *args, **kwargs) -> None:
        self._camera_id = kwargs.get("camera_id", "maixcam2")
        self._opened = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return 30.0

    def read(self) -> Optional[np.ndarray]:
        return None

    def set(self, prop_id: int, value) -> bool:
        return False

    def release(self) -> None:
        self._opened = False
