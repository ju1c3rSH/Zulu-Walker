# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any, Optional, Callable

import cv2
import numpy as np
import yaml

from ..param_utils import CAMERA_PARAM_DEFS, load_camera_params, read_camera_params_from_capture


class CameraDebugWindow:
    def __init__(
        self,
        cap: cv2.VideoCapture,
        config_path: str = None,
        on_params_change: Callable[[Dict[str, Any]], None] = None
    ):
        self.cap = cap
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "camera_params.yaml"
        )
        self.on_params_change = on_params_change

        self.window_name = "Camera Params Debug"
        self._window_created = False

        self._save_pending = False
        self._last_save_time = 0

        self.params: Dict[str, int] = {}
        self._prop_map: Dict[str, int] = {}
        self._exposure_offset = 13

        for display_name, key, cap_prop, min_val, max_val in CAMERA_PARAM_DEFS:
            self._prop_map[key] = cap_prop

        self.params = read_camera_params_from_capture(self.cap)
        loaded, _ = load_camera_params(self.config_path)
        self.params.update(loaded)

    def _save_config(self):
        self._save_pending = True

    def _do_save(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            data = {"camera_params": self.params.copy()}
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"[CameraDebugWindow] Config saved to {self.config_path}")
        except Exception as e:
            print(f"[CameraDebugWindow] Failed to save config: {e}")

    def setup_window(self):
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 500)

        for display_name, key, cap_prop, min_val, max_val in CAMERA_PARAM_DEFS:
            if key not in self.params:
                continue
            if key == "exposure":
                tb_min = 0
                tb_max = max_val - min_val
                tb_default = self.params[key] - min_val
                cv2.createTrackbar(
                    display_name, self.window_name, tb_default, tb_max,
                    lambda val, k=key, mn=min_val: self._on_trackbar_exposure(k, val, mn)
                )
            else:
                cv2.createTrackbar(
                    display_name, self.window_name, self.params[key], max_val,
                    lambda val, k=key: self._on_trackbar(k, val)
                )

            self._apply_param(key)

        self._window_created = True

        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def _apply_param(self, key: str):
        cap_prop = self._prop_map.get(key)
        if cap_prop is not None:
            try:
                self.cap.set(cap_prop, self.params[key])
            except Exception:
                pass

    def _on_trackbar(self, key: str, value: int):
        self.params[key] = value
        self._apply_param(key)
        self._save_config()
        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def _on_trackbar_exposure(self, key: str, tb_value: int, min_val: int):
        real_value = tb_value + min_val
        self.params[key] = real_value
        self._apply_param(key)
        self._save_config()
        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def update_gui(self):
        if not self._window_created:
            return

        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        info = np.zeros((300, 400, 3), dtype=np.uint8)
        y = 25
        for display_name, key, cap_prop, min_val, max_val in CAMERA_PARAM_DEFS:
            if key not in self.params:
                continue
            text = f"{display_name}: {self.params[key]}"
            cv2.putText(info, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25

        cv2.imshow(self.window_name, info)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def destroy_window(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
