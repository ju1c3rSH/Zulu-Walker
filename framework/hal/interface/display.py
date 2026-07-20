from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Display(Protocol):
    def show(self, frame: np.ndarray) -> bool:
        ...

    def close(self) -> None:
        ...
