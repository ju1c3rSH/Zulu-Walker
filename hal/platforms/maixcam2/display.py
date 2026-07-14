from __future__ import annotations

import numpy as np


class MaixCam2Display:
    def show(self, frame: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        pass
