from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .param_def import ParamDef

_MAX_PREVIEW_STEPS = 10


class BaseDebugWindow:
    def __init__(
        self,
        title: str = "Debug",
        method_count: int = 1,
        param_defs: Optional[List[ParamDef]] = None,
        on_change: Optional[Callable[[str, int], None]] = None,
        on_method_change: Optional[Callable[[int], None]] = None,
    ):
        self.title = title
        self.method_count = method_count
        self.param_defs = param_defs or []
        self.on_change = on_change
        self.on_method_change = on_method_change

        self.preview_index = 0
        self.enabled = True
        self.method_index = 0
        self._window_created = False

        self._raw_params: Dict[str, int] = {}
        for p in self.param_defs:
            self._raw_params[p.name] = p.default

        self._frame: Optional[np.ndarray] = None
        self._result: Optional[np.ndarray] = None
        self._intermediate_steps: Dict[int, np.ndarray] = {}

    def setup(self):
        if self._window_created:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1024, 768)

        cv2.createTrackbar(
            "Preview", self.title, self.preview_index, _MAX_PREVIEW_STEPS,
            lambda v: self._set_preview(v),
        )
        cv2.createTrackbar(
            "Enable", self.title, 1 if self.enabled else 0, 1,
            lambda v: self._set_enabled(v),
        )
        cv2.createTrackbar(
            "Method", self.title, self.method_index,
            max(1, self.method_count - 1),
            lambda v: self._set_method(v),
        )

        for p in self.param_defs:
            cv2.createTrackbar(
                p.display, self.title,
                self._raw_params[p.name],
                p.max_val,
                lambda v, d=p: self._on_trackbar(d, v),
            )

        self._window_created = True

    def _set_preview(self, value: int):
        self.preview_index = value

    def _set_enabled(self, value: int):
        self.enabled = bool(value)

    def _set_method(self, value: int):
        if value == self.method_index:
            return
        self.method_index = value
        self.preview_index = 0
        if self._window_created:
            cv2.setTrackbarPos("Preview", self.title, 0)
        if self.on_method_change:
            self.on_method_change(value)

    def _on_trackbar(self, pdef: ParamDef, raw_value: int):
        if pdef.odd and raw_value % 2 == 0:
            raw_value = max(pdef.min_val, raw_value - 1)
            if self._window_created:
                cv2.setTrackbarPos(pdef.display, self.title, raw_value)

        if self._raw_params.get(pdef.name) != raw_value:
            self._raw_params[pdef.name] = raw_value
            if self.on_change:
                self.on_change(pdef.name, raw_value)

    def update(self, frame: Optional[np.ndarray] = None,
               result: Optional[np.ndarray] = None,
               intermediates: Optional[Dict[int, np.ndarray]] = None):
        if frame is not None:
            self._frame = frame
        if result is not None:
            self._result = result
        if intermediates is not None:
            self._intermediate_steps = intermediates

    def refresh(self):
        if not self._window_created:
            return
        preview = self._build_preview()
        if preview is not None:
            self._draw_params_info(preview)
            cv2.imshow(self.title, preview)

    def _draw_params_info(self, frame: np.ndarray):
        line_count = len(self.param_defs) + 1
        line_height = 14
        pad = 8
        panel_w = 220
        panel_h = min(line_count * line_height + pad * 2, frame.shape[0] - 10, 400)
        if panel_w > frame.shape[1] - 10:
            panel_w = frame.shape[1] - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

        y = 5 + pad + line_height
        idx = 1
        for p in self.param_defs:
            raw = self._raw_params.get(p.name, p.default)
            if p.scale != 1.0:
                text = f"#{idx:02d} {p.display}: {raw * p.scale:.2f}"
            else:
                text = f"#{idx:02d} {p.display}: {raw}"
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += line_height
            idx += 1

        status = "ENABLED" if self.enabled else "DISABLED"
        color = (0, 255, 0) if self.enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _build_preview(self) -> Optional[np.ndarray]:
        if self.preview_index == 0 and self._frame is not None:
            return self._frame.copy()
        step = self.preview_index - 1
        if step in self._intermediate_steps:
            img = self._intermediate_steps[step]
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()
        if self._result is not None:
            return self._result.copy()
        if self._frame is not None:
            return self._frame.copy()
        return None

    def set_param(self, name: str, raw_value: int):
        if name in self._raw_params:
            self._raw_params[name] = raw_value
            if self._window_created:
                for p in self.param_defs:
                    if p.name == name:
                        cv2.setTrackbarPos(p.display, self.title, raw_value)
                        break

    def set_method_index(self, index: int):
        self.method_index = index
        if self._window_created:
            cv2.setTrackbarPos("Method", self.title, index)

    def get_raw_params(self) -> Dict[str, int]:
        return self._raw_params.copy()

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.title)
            self._window_created = False
