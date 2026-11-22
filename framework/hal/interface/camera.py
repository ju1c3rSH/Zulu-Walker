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
        """取一帧并返回与 last_raw 同帧的 BGR numpy（主流水线用它）。

        返回的 numpy 帧与 last_raw 必须是同一帧：context["camera"].last_raw
        供 maix C++ 零拷贝路径使用，numpy 帧供 cv2 回退路径。
        停帧（无新帧）时返回 None，调用方不得复用旧帧。
        """
        ...

    @property
    def last_raw(self):
        """最近一次读取的原始帧（maix Image；无 maix 的平台返回 None）。"""
        ...

    @property
    def fps(self) -> float:
        ...

    def release(self) -> None:
        ...

    def set(self, prop_id: int, value) -> bool:
        ...
