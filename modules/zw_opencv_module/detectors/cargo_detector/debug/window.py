from enum import Enum
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import ParamDef


class PreviewMode(Enum):
    ORIGINAL = 0
    MASK = 1
    RESULT = 2


class CargoDebugWindow:
    def __init__(
        self,
        title: str = "Cargo Debug",
        param_defs: Optional[List[ParamDef]] = None,
        on_change: Optional[Callable[[str, int], None]] = None,
    ):
        self.title = title
        self.param_defs = param_defs or []
        self.on_change = on_change

        self.preview_mode = PreviewMode.RESULT
        self.enabled = True
        self._window_created = False

        self._raw_params: Dict[str, int] = {}
        for p in self.param_defs:
            self._raw_params[p.name] = p.default

        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._result: Optional[np.ndarray] = None

    def setup(self):
        if self._window_created:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 640, 480)

        cv2.createTrackbar(
            "Preview", self.title, self.preview_mode.value, 2,
            lambda v: self._set_preview_mode(v),
        )
        cv2.createTrackbar(
            "Enable", self.title, 1 if self.enabled else 0, 1,
            lambda v: self._set_enabled(v),
        )

        for p in self.param_defs:
            cv2.createTrackbar(
                p.display, self.title,
                self._raw_params[p.name],
                p.max_val,
                lambda v, d=p: self._on_trackbar(d, v),
            )

        self._window_created = True

    def _set_preview_mode(self, value: int):
        self.preview_mode = PreviewMode(value)

    def _set_enabled(self, value: int):
        self.enabled = bool(value)

    def _on_trackbar(self, pdef: ParamDef, raw_value: int):
        if pdef.odd and raw_value % 2 == 0:
            raw_value = max(pdef.min_val, raw_value - 1)
            if self._window_created:
                cv2.setTrackbarPos(pdef.display, self.title, raw_value)

        if self._raw_params.get(pdef.name) != raw_value:
            self._raw_params[pdef.name] = raw_value
            if self.on_change:
                self.on_change(pdef.name, raw_value)

    def update(self, frame: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
               result: Optional[np.ndarray] = None):
        if frame is not None:
            self._frame = frame
        self._mask = mask
        self._result = result

    def refresh(self):
        if not self._window_created:
            return

        preview = self._build_preview()
        if preview is not None:
            cv2.imshow(self.title, preview)

    def _build_preview(self) -> Optional[np.ndarray]:
        if self.preview_mode == PreviewMode.ORIGINAL:
            return self._frame.copy() if self._frame is not None else None
        elif self.preview_mode == PreviewMode.MASK:
            if self._mask is not None:
                return cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR)
            return None
        elif self.preview_mode == PreviewMode.RESULT:
            return self._result.copy() if self._result is not None else None
        return None

    def set_param(self, name: str, raw_value: int):
        if name in self._raw_params:
            self._raw_params[name] = raw_value
            if self._window_created:
                for p in self.param_defs:
                    if p.name == name:
                        cv2.setTrackbarPos(p.display, self.title, raw_value)
                        break

    def get_raw_params(self) -> Dict[str, int]:
        return self._raw_params.copy()

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.title)
            self._window_created = False
