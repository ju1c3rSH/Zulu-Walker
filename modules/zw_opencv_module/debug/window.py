# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum

import cv2
import numpy as np
import yaml

from .param_panel import ParamPanel, METHOD_PARAMS
from utils.log_util import log_print



class PreviewMode(Enum):
    ORIGINAL = 0
    EDGE = 1
    RESULT = 2


class DebugWindow:
    def __init__(self, config_path: str = None, on_params_change: Callable = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "debug_params.yaml"
        )
        self.on_params_change = on_params_change

        self.param_panel: ParamPanel = None

        self.enabled = True
        self.preview_mode = PreviewMode.EDGE
        self._window_created = False

        self._current_frame: Optional[np.ndarray] = None
        self._result_frame: Optional[np.ndarray] = None
        self._edge_frame: Optional[np.ndarray] = None

        self._save_pending = False
        self._last_save_time = 0

        self.window_name = "Edge Detection Debug"

        self._init_param_panel()
        self._load_config()

    def _init_param_panel(self):
        self.current_method = "edge_drawing_quads"

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and "current_method" in data:
                        method_name = data["current_method"]
                        if method_name in METHOD_PARAMS:
                            self.current_method = method_name
            except Exception as e:
                log_print(f"[DebugWindow] Failed to load current_method: {e}")

        if self.current_method in METHOD_PARAMS:
            self.param_panel = ParamPanel(
                method_name=self.current_method,
                params_def=METHOD_PARAMS[self.current_method],
                window_name=self.window_name,
                on_change=self._on_panel_change,
            )

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    return

                if "enabled" in data:
                    self.enabled = data["enabled"]

                if "methods" in data and self.current_method in data["methods"]:
                    params = data["methods"][self.current_method]
                    if self.param_panel:
                        self.param_panel.load_params(params)

        except Exception as e:
            log_print(f"[DebugWindow] Failed to load config: {e}")

    def _save_config(self):
        self._save_pending = True

    def _do_save(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            data = {
                "current_method": self.current_method,
                "enabled": self.enabled,
                "methods": {
                    self.current_method: self.param_panel.get_raw_params() if self.param_panel else {}
                },
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            log_print(f"[DebugWindow] Config saved to {self.config_path}")

        except Exception as e:
            log_print(f"[DebugWindow] Failed to save config: {e}")

    def _on_panel_change(self, method_name: str, params: Dict[str, Any]):
        if self.on_params_change:
            self.on_params_change(method_name, params)
        self._save_config()

    def update_frame(self, frame: np.ndarray, edge_frame: np.ndarray = None, result_frame: np.ndarray = None):
        self._current_frame = frame
        self._edge_frame = edge_frame
        self._result_frame = result_frame

    def setup_window(self):
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        cv2.createTrackbar(
            "Preview", self.window_name,
            self.preview_mode.value, 2,
            lambda val: self._on_mode_change(val)
        )

        cv2.createTrackbar(
            "Enable", self.window_name,
            1 if self.enabled else 0, 1,
            lambda val: self._on_enable_change(val)
        )

        if self.param_panel:
            self.param_panel.create_trackbars()

        self._window_created = True

        if self.param_panel and self.on_params_change:
            self.on_params_change(self.param_panel.method_name, self.param_panel.get_params())

    def _on_mode_change(self, value: int):
        self.preview_mode = PreviewMode(value)

    def _on_enable_change(self, value: int):
        self.enabled = bool(value)
        self._save_config()

    def update_gui(self):
        if not self._window_created:
            return

        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        preview = self._generate_preview(
            self._current_frame,
            self._edge_frame,
            self._result_frame
        )

        if preview is not None:
            self._draw_params_info(preview)
            cv2.imshow(self.window_name, preview)

    def destroy_window(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    def _generate_preview(self, frame: np.ndarray, edge: np.ndarray, result: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        scale = min(320 / w, 240 / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if self.preview_mode == PreviewMode.ORIGINAL:
            return frame.copy()

        elif self.preview_mode == PreviewMode.EDGE:
            if edge is not None:
                if edge.shape[:2] != frame.shape[:2]:
                    edge = cv2.resize(edge, (frame.shape[1], frame.shape[0]))
                return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
            return frame.copy()

        elif self.preview_mode == PreviewMode.RESULT:
            if result is not None:
                if result.shape[:2] != frame.shape[:2]:
                    result = cv2.resize(result, (frame.shape[1], frame.shape[0]))
                return result.copy()
            return frame.copy()

        return frame.copy()

    def _draw_params_info(self, frame: np.ndarray):
        y = 20

        if self.param_panel:
            for pdef in self.param_panel.params_def:
                value = self.param_panel.params.get(pdef.name, 0)
                if pdef.scale != 1.0:
                    display_value = value * pdef.scale
                    text = f"{pdef.display_name}: {display_value:.1f}"
                else:
                    text = f"{pdef.display_name}: {value}"

                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                y += 12

        status = "ENABLED" if self.enabled else "DISABLED"
        color = (0, 255, 0) if self.enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def get_params(self) -> Dict[str, Any]:
        if self.param_panel:
            return self.param_panel.get_params()
        return {}

    def is_enabled(self) -> bool:
        return self.enabled
