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

    def read_raw(self) -> Optional[np.ndarray]:
        """Return one frame as BGR ndarray, from the same frame as ``last_raw``.

        Both views must refer to the same frame: ``context["camera"].last_raw``
        feeds the zero-copy maix C++ path, the ndarray the cv2 fallback. Return
        None when no new frame exists; callers must not reuse a stale frame.
        """
        ...

    @property
    def last_raw(self):
        """Most recent raw frame (a maix Image; None on platforms without maix)."""
        ...

    @property
    def fps(self) -> float:
        ...

    def release(self) -> None:
        ...

    def set(self, prop_id: int, value) -> bool:
        ...
