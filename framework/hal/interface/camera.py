from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Camera(Protocol):
    @property
    def camera_id(self) -> str:
        ...

    def read(self) -> Optional[np.ndarray]:
        ...

    @property
    def fps(self) -> float:
        ...

    def release(self) -> None:
        ...

    def set(self, prop_id: int, value) -> bool:
        ...
