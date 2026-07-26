from __future__ import annotations

import maix.display
import maix.image
import maix.err
import numpy as np


class MaixCam2Display:
    def __init__(self) -> None:
        self._disp: maix.display.Display = maix.display.Display(
            open=True,
        )

    def show(self, frame: np.ndarray) -> bool:
        if self._disp is None:
            return False
        try:
            img = maix.image.cv2image(frame, bgr=True, copy=False)
            err = self._disp.show(img)
            return err == maix.err.Err.ERR_NONE
        except Exception:
            return False

    def close(self) -> None:
        if self._disp is not None:
            try:
                self._disp.close()
            except Exception:
                pass
            self._disp = None
