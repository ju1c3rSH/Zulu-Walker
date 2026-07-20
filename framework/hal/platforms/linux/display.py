from __future__ import annotations

import cv2
import numpy as np


class LinuxDisplay:
    _window_name = "Zulu-Walker"

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass
