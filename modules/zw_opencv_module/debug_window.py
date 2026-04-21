# -*- coding: utf-8 -*-
"""
边缘检测参数调试窗口

提供实时参数调整和预览功能，用于优化边缘检测效果。

注意：OpenCV GUI 在 Linux/X11 上必须在主线程中运行，
因此本模块不使用独立线程，而是在主循环中调用 update_gui()。
"""
import os
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum

import cv2
import numpy as np
import yaml

from .param_panel import ParamPanel, METHOD_PARAMS


class PreviewMode(Enum):
    """预览模式"""
    ORIGINAL = 0      # 原始图像
    EDGE = 1          # EdgeDrawing边缘
    RESULT = 2        # 检测结果叠加


class DebugWindow:
    """
    边缘检测参数调试窗口

    功能：
    - 参数面板
    - 参数自动保存到YAML
    - 多阶段预览（原图/Canny/结果）
    - 开启/关闭检测

    注意：必须在主线程中调用 setup_window() 和 update_gui()
    """

    def __init__(self, config_path: str = None, on_params_change: Callable = None):
        """
        初始化调试窗口

        Args:
            config_path: 配置文件路径
            on_params_change: 参数变化回调函数 (method_name, params_dict)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "debug_params.yaml"
        )
        self.on_params_change = on_params_change

        # 参数面板
        self.param_panel: ParamPanel = None

        # 状态
        self.enabled = True
        self.preview_mode = PreviewMode.EDGE
        self._window_created = False

        # 帧缓冲
        self._current_frame: Optional[np.ndarray] = None
        self._result_frame: Optional[np.ndarray] = None
        self._edge_frame: Optional[np.ndarray] = None

        # 保存节流
        self._save_pending = False
        self._last_save_time = 0

        # 窗口名称
        self.window_name = "Edge Detection Debug"

        # 初始化参数面板
        self._init_param_panel()

        # 从YAML加载
        self._load_config()

    def _init_param_panel(self):
        """初始化参数面板"""
        # 只使用 edge_contour_ellipse 方法
        method_name = "edge_contour_ellipse"
        if method_name in METHOD_PARAMS:
            self.param_panel = ParamPanel(
                method_name=method_name,
                params_def=METHOD_PARAMS[method_name],
                window_name=self.window_name,
                on_change=self._on_panel_change,
            )

    def _load_config(self):
        """从YAML加载配置"""
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    return

                # 加载启用状态
                if "enabled" in data:
                    self.enabled = data["enabled"]

                # 加载参数
                if "methods" in data and "edge_contour_ellipse" in data["methods"]:
                    params = data["methods"]["edge_contour_ellipse"]
                    if self.param_panel:
                        self.param_panel.load_params(params)

        except Exception as e:
            print(f"[DebugWindow] Failed to load config: {e}")

    def _save_config(self):
        """保存配置到YAML（节流）"""
        self._save_pending = True

    def _do_save(self):
        """执行保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            data = {
                "enabled": self.enabled,
                "methods": {
                    "edge_contour_ellipse": self.param_panel.get_raw_params() if self.param_panel else {}
                },
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"[DebugWindow] Config saved to {self.config_path}")

        except Exception as e:
            print(f"[DebugWindow] Failed to save config: {e}")

    def _on_panel_change(self, method_name: str, params: Dict[str, Any]):
        """参数面板变化回调"""
        # 触发外部回调
        if self.on_params_change:
            self.on_params_change(method_name, params)

        # 保存配置
        self._save_config()

    def update_frame(self, frame: np.ndarray, edge_frame: np.ndarray = None, result_frame: np.ndarray = None):
        """
        更新帧数据

        Args:
            frame: 原始帧
            edge_frame: EdgeDrawing边缘帧
            result_frame: 检测结果帧
        """
        self._current_frame = frame
        self._edge_frame = edge_frame
        self._result_frame = result_frame

    def setup_window(self):
        """
        创建窗口和滑动条（必须在主线程调用）

        只调用一次。
        """
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        # 创建预览模式滑动条
        cv2.createTrackbar(
            "Preview", self.window_name,
            self.preview_mode.value, 2,
            lambda val: self._on_mode_change(val)
        )

        # 创建启用/禁用滑动条
        cv2.createTrackbar(
            "Enable", self.window_name,
            1 if self.enabled else 0, 1,
            lambda val: self._on_enable_change(val)
        )

        # 创建参数滑动条
        if self.param_panel:
            self.param_panel.create_trackbars()

        self._window_created = True

    def _on_mode_change(self, value: int):
        """预览模式切换"""
        self.preview_mode = PreviewMode(value)

    def _on_enable_change(self, value: int):
        """启用/禁用切换"""
        self.enabled = bool(value)
        self._save_config()

    def update_gui(self):
        """
        更新 GUI 显示（必须在主线程调用）

        每帧调用一次。
        """
        if not self._window_created:
            return

        # 处理保存
        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        # 生成预览
        preview = self._generate_preview(
            self._current_frame,
            self._edge_frame,
            self._result_frame
        )

        if preview is not None:
            # 绘制参数信息
            self._draw_params_info(preview)
            cv2.imshow(self.window_name, preview)

    def destroy_window(self):
        """销毁窗口"""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    def _generate_preview(self, frame: np.ndarray, edge: np.ndarray, result: np.ndarray) -> Optional[np.ndarray]:
        """生成预览图像"""
        if frame is None:
            # 返回黑色图像
            return np.zeros((240, 320, 3), dtype=np.uint8)

        # 降采样到 320x240
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
        """在预览上绘制参数信息"""
        y = 20

        # 显示当前方法的参数
        if self.param_panel:
            for pdef in self.param_panel.params_def:
                value = self.param_panel.params.get(pdef.name, 0)
                # 应用缩放因子显示
                if pdef.scale != 1.0:
                    display_value = value * pdef.scale
                    text = f"{pdef.display_name}: {display_value:.1f}"
                else:
                    text = f"{pdef.display_name}: {value}"

                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                y += 12

        # 状态
        status = "ENABLED" if self.enabled else "DISABLED"
        color = (0, 255, 0) if self.enabled else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def get_params(self) -> Dict[str, Any]:
        """
        获取参数

        Returns:
            参数字典
        """
        if self.param_panel:
            return self.param_panel.get_params()
        return {}

    def is_enabled(self) -> bool:
        """检测是否启用"""
        return self.enabled
