# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any, Optional, Callable, Tuple

import cv2
import numpy as np
import yaml


class UVDebugWindow:
    def __init__(
        self,
        config_path: str = None,
        on_params_change: Callable[[Dict[str, Any]], None] = None
    ):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "uv_params.yaml"
        )
        self.on_params_change = on_params_change

        self.window_name = "UV Detection Debug"

        self._window_created = False

        self._current_frame: Optional[np.ndarray] = None
        self._uv_mask: Optional[np.ndarray] = None

        self._save_pending = False
        self._last_save_time = 0

        self.params = {
            "uv_h_min1": 130,
            "uv_h_max1": 145,
            "uv_s_min1": 90,
            "uv_s_max1": 255,
            "uv_v_min1": 190,
            "uv_v_max1": 255,
            "uv_h_min2": 130,
            "uv_h_max2": 155,
            "uv_s_min2": 0,
            "uv_s_max2": 50,
            "uv_v_min2": 236,
            "uv_v_max2": 255,
            "uv_min_area": 0,
            "uv_adaptive_enabled": 1,
            "uv_v_percentile": 95,
            "uv_v_floor": 90,
            "uv_s_min": 80,
            "uv_h_low": 130,
            "uv_h_high": 160,
            "uv_s_gate": 80,
            "uv_contrast_ratio_min": 115,
            "uv_contrast_dilate": 30,
        }

        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "uv_params" in data:
                    for name, value in data["uv_params"].items():
                        if name in self.params:
                            self.params[name] = int(value)
        except Exception as e:
            print(f"[UVDebugWindow] Failed to load config: {e}")

    def _save_config(self):
        self._save_pending = True

    def _do_save(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            data = {
                "uv_params": self.params.copy()
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"[UVDebugWindow] Config saved to {self.config_path}")

        except Exception as e:
            print(f"[UVDebugWindow] Failed to save config: {e}")

    def setup_window(self):
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        cv2.createTrackbar("H Min1", self.window_name,
                           self.params["uv_h_min1"], 180,
                           lambda val: self._on_trackbar("uv_h_min1", val))
        cv2.createTrackbar("H Max1", self.window_name,
                           self.params["uv_h_max1"], 180,
                           lambda val: self._on_trackbar("uv_h_max1", val))
        cv2.createTrackbar("S Min1", self.window_name,
                           self.params["uv_s_min1"], 255,
                           lambda val: self._on_trackbar("uv_s_min1", val))
        cv2.createTrackbar("S Max1", self.window_name,
                           self.params["uv_s_max1"], 255,
                           lambda val: self._on_trackbar("uv_s_max1", val))
        cv2.createTrackbar("V Min1", self.window_name,
                           self.params["uv_v_min1"], 255,
                           lambda val: self._on_trackbar("uv_v_min1", val))
        cv2.createTrackbar("V Max1", self.window_name,
                           self.params["uv_v_max1"], 255,
                           lambda val: self._on_trackbar("uv_v_max1", val))

        cv2.createTrackbar("H Min2", self.window_name,
                           self.params["uv_h_min2"], 180,
                           lambda val: self._on_trackbar("uv_h_min2", val))
        cv2.createTrackbar("H Max2", self.window_name,
                           self.params["uv_h_max2"], 180,
                           lambda val: self._on_trackbar("uv_h_max2", val))
        cv2.createTrackbar("S Min2", self.window_name,
                           self.params["uv_s_min2"], 255,
                           lambda val: self._on_trackbar("uv_s_min2", val))
        cv2.createTrackbar("S Max2", self.window_name,
                           self.params["uv_s_max2"], 255,
                           lambda val: self._on_trackbar("uv_s_max2", val))
        cv2.createTrackbar("V Min2", self.window_name,
                           self.params["uv_v_min2"], 255,
                           lambda val: self._on_trackbar("uv_v_min2", val))
        cv2.createTrackbar("V Max2", self.window_name,
                           self.params["uv_v_max2"], 255,
                           lambda val: self._on_trackbar("uv_v_max2", val))

        cv2.createTrackbar("UV MinArea", self.window_name,
                           self.params["uv_min_area"], 100,
                           lambda val: self._on_trackbar("uv_min_area", val))

        cv2.createTrackbar("Adaptive", self.window_name,
                           self.params["uv_adaptive_enabled"], 1,
                           lambda val: self._on_trackbar("uv_adaptive_enabled", val))
        cv2.createTrackbar("V Percentile", self.window_name,
                           self.params["uv_v_percentile"], 99,
                           lambda val: self._on_trackbar("uv_v_percentile", val))
        cv2.createTrackbar("V Floor", self.window_name,
                           self.params["uv_v_floor"], 100,
                           lambda val: self._on_trackbar("uv_v_floor", val))
        cv2.createTrackbar("S Min", self.window_name,
                           self.params["uv_s_min"], 50,
                           lambda val: self._on_trackbar("uv_s_min", val))
        cv2.createTrackbar("H Low", self.window_name,
                           self.params["uv_h_low"], 180,
                           lambda val: self._on_trackbar("uv_h_low", val))
        cv2.createTrackbar("H High", self.window_name,
                           self.params["uv_h_high"], 180,
                           lambda val: self._on_trackbar("uv_h_high", val))
        cv2.createTrackbar("S Gate", self.window_name,
                           self.params["uv_s_gate"], 255,
                           lambda val: self._on_trackbar("uv_s_gate", val))
        cv2.createTrackbar("ContrastR", self.window_name,
                           self.params["uv_contrast_ratio_min"], 300,
                           lambda val: self._on_trackbar("uv_contrast_ratio_min", val))
        cv2.createTrackbar("CtrDilate", self.window_name,
                           self.params["uv_contrast_dilate"], 50,
                           lambda val: self._on_trackbar("uv_contrast_dilate", val))

        self._window_created = True

        if self.on_params_change:
            self.on_params_change(self.get_color_ranges(), self.params["uv_min_area"], self.params)

    def _on_trackbar(self, name: str, value: int):
        self.params[name] = value

        if self.on_params_change:
            self.on_params_change(self.get_color_ranges(), self.params["uv_min_area"], self.params)

        self._save_config()

    def update_frame(self, frame: np.ndarray, uv_mask: np.ndarray = None):
        self._current_frame = frame
        self._uv_mask = uv_mask

    def update_gui(self):
        if not self._window_created:
            return

        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        preview = self._generate_preview()

        if preview is not None:
            cv2.imshow(self.window_name, preview)

    def _generate_preview(self) -> Optional[np.ndarray]:
        if self._current_frame is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        frame = self._current_frame

        h, w = frame.shape[:2]
        scale = min(320 / w, 240 / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.params["uv_adaptive_enabled"]:
            v_channel = hsv[:, :, 2]
            v_min = max(int(np.percentile(v_channel, self.params["uv_v_percentile"])),
                        self.params["uv_v_floor"])
            h_lo = self.params["uv_h_low"]
            h_hi = self.params["uv_h_high"]
            s_min = self.params["uv_s_min"]
            uv_ranges = [(np.array([h_lo, s_min, v_min]), np.array([h_hi, 255, 255]))]
        else:
            uv_ranges = self.get_color_ranges()

        mask = None
        for lower, upper in uv_ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.params["uv_min_area"]]

        preview_left = frame.copy()
        purple_overlay = np.zeros_like(frame)
        purple_overlay[mask > 0] = (180, 50, 180)
        cv2.addWeighted(purple_overlay, 0.5, preview_left, 0.5, 0, preview_left)

        preview_right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(preview_right, (cx, cy), 5, (0, 0, 255), -1)

        preview = np.hstack([preview_left, preview_right])

        self._draw_params_info(preview)

        return preview

    def _draw_params_info(self, frame: np.ndarray):
        y = 20

        cv2.putText(frame,
                    f"R1: H[{self.params['uv_h_min1']}-{self.params['uv_h_max1']}] "
                    f"S[{self.params['uv_s_min1']}-{self.params['uv_s_max1']}] "
                    f"V[{self.params['uv_v_min1']}-{self.params['uv_v_max1']}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y += 15

        cv2.putText(frame,
                    f"R2: H[{self.params['uv_h_min2']}-{self.params['uv_h_max2']}] "
                    f"S[{self.params['uv_s_min2']}-{self.params['uv_s_max2']}] "
                    f"V[{self.params['uv_v_min2']}-{self.params['uv_v_max2']}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y += 15

        cv2.putText(frame,
                    f"Min Area: {self.params['uv_min_area']}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y += 15

        adaptive = self.params["uv_adaptive_enabled"]
        mode_str = "ADAPTIVE" if adaptive else "STATIC"
        color = (0, 255, 255) if adaptive else (200, 200, 200)
        cv2.putText(frame,
                    f"Mode: {mode_str}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if adaptive:
            y += 15
            cv2.putText(frame,
                        f"V%={self.params['uv_v_percentile']} Vf={self.params['uv_v_floor']} "
                        f"S={self.params['uv_s_min']} H=[{self.params['uv_h_low']}-{self.params['uv_h_high']}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y += 12
            cr = self.params['uv_contrast_ratio_min'] / 100.0
            cv2.putText(frame,
                        f"S_gate={self.params['uv_s_gate']} CR={cr:.1f} "
                        f"Dilate={self.params['uv_contrast_dilate']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y += 12
            cv2.putText(frame, "R1/R2 sliders inactive",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    def get_color_ranges(self) -> list:
        return [
            (np.array([self.params["uv_h_min1"], self.params["uv_s_min1"], self.params["uv_v_min1"]]),
             np.array([self.params["uv_h_max1"], self.params["uv_s_max1"], self.params["uv_v_max1"]])),
            (np.array([self.params["uv_h_min2"], self.params["uv_s_min2"], self.params["uv_v_min2"]]),
             np.array([self.params["uv_h_max2"], self.params["uv_s_max2"], self.params["uv_v_max2"]])),
        ]

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def destroy_window(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
