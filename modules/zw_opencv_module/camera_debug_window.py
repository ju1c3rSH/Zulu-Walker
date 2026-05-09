# -*- coding: utf-8 -*-
"""
摄像头参数调试窗口

独立窗口用于实时调节摄像头硬件参数（曝光、增益、锐度等）。

使用方法:
    python run.py debug --debug-cam
"""
import os
import time
from typing import Dict, Any, Optional, Callable

import cv2
import numpy as np
import yaml

from .param_utils import CAMERA_PARAM_DEFS, load_camera_params


class CameraDebugWindow:
    """
    摄像头参数调试窗口

    功能：
    - 摄像头硬件参数滑动条（曝光、增益、锐度等）
    - 通过 cv2.CAP_PROP_* 实时设置
    - 参数自动保存到 YAML
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        config_path: str = None,
        on_params_change: Callable[[Dict[str, Any]], None] = None
    ):
        self.cap = cap
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "camera_params.yaml"
        )
        self.on_params_change = on_params_change

        self.window_name = "Camera Params Debug"
        self._window_created = False

        # 保存节流
        self._save_pending = False
        self._last_save_time = 0

        # 参数字典
        self.params: Dict[str, int] = {}
        # CAP_PROP 映射
        self._prop_map: Dict[str, int] = {}
        # 曝光偏移（trackbar 不支持负数，用 offset 转换）
        self._exposure_offset = 13  # -13 → 0, -1 → 12

        for display_name, key, cap_prop, min_val, max_val, default in CAMERA_PARAM_DEFS:
            self._prop_map[key] = cap_prop
            self.params[key] = default

        self._load_config()

    def _load_config(self):
        """从 YAML 加载配置"""
        loaded, _ = load_camera_params(self.config_path)
        self.params.update(loaded)

    def _save_config(self):
        """保存配置（节流）"""
        self._save_pending = True

    def _do_save(self):
        """执行保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            data = {"camera_params": self.params.copy()}
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"[CameraDebugWindow] Config saved to {self.config_path}")
        except Exception as e:
            print(f"[CameraDebugWindow] Failed to save config: {e}")

    def setup_window(self):
        """创建窗口和滑动条（必须在主线程调用）"""
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 500)

        for display_name, key, cap_prop, min_val, max_val, default in CAMERA_PARAM_DEFS:
            # 曝光参数需要偏移转换（负数 → 非负数）
            if key == "exposure":
                tb_min = 0
                tb_max = max_val - min_val  # 0..12
                tb_default = self.params[key] - min_val  # -5 - (-13) = 8
                cv2.createTrackbar(
                    display_name, self.window_name, tb_default, tb_max,
                    lambda val, k=key, mn=min_val: self._on_trackbar_exposure(k, val, mn)
                )
            else:
                cv2.createTrackbar(
                    display_name, self.window_name, self.params[key], max_val,
                    lambda val, k=key: self._on_trackbar(k, val)
                )

            # 应用初始值到摄像头
            self._apply_param(key)

        self._window_created = True

        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def _apply_param(self, key: str):
        """将参数值应用到摄像头"""
        cap_prop = self._prop_map.get(key)
        if cap_prop is not None:
            try:
                self.cap.set(cap_prop, self.params[key])
            except Exception:
                pass

    def _on_trackbar(self, key: str, value: int):
        """普通滑动条回调"""
        self.params[key] = value
        self._apply_param(key)
        self._save_config()
        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def _on_trackbar_exposure(self, key: str, tb_value: int, min_val: int):
        """曝光滑动条回调（带偏移转换）"""
        real_value = tb_value + min_val  # 8 + (-13) = -5
        self.params[key] = real_value
        self._apply_param(key)
        self._save_config()
        if self.on_params_change:
            self.on_params_change(self.params.copy())

    def update_gui(self):
        """更新 GUI（必须在主线程调用）"""
        if not self._window_created:
            return

        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        # 显示当前参数信息面板
        info = np.zeros((300, 400, 3), dtype=np.uint8)
        y = 25
        for display_name, key, cap_prop, min_val, max_val, default in CAMERA_PARAM_DEFS:
            text = f"{display_name}: {self.params[key]}"
            cv2.putText(info, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25

        cv2.imshow(self.window_name, info)

    def get_params(self) -> Dict[str, Any]:
        """获取所有参数"""
        return self.params.copy()

    def destroy_window(self):
        """销毁窗口"""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
