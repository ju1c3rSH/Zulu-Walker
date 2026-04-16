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
from .circle_target_detector import DetectMethod


class PreviewMode(Enum):
    """预览模式"""
    ORIGINAL = 0      # 原始图像
    CANNY = 1         # Canny边缘
    RESULT = 2        # 检测结果叠加


class DebugWindow:
    """
    边缘检测参数调试窗口

    功能：
    - 检测方法选择器
    - 各方法的独立参数面板
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

        # 当前检测方法
        self.current_method = DetectMethod.EDGE_CONTOUR_ELLIPSE

        # 参数面板（按方法名索引）
        self.param_panels: Dict[str, ParamPanel] = {}

        # 状态
        self.enabled = True
        self.preview_mode = PreviewMode.CANNY
        self._window_created = False

        # 帧缓冲
        self._current_frame: Optional[np.ndarray] = None
        self._result_frame: Optional[np.ndarray] = None
        self._canny_frame: Optional[np.ndarray] = None

        # 保存节流
        self._save_pending = False
        self._last_save_time = 0

        # 窗口名称
        self.window_name = "Edge Detection Debug"

        # 初始化参数面板
        self._init_param_panels()

        # 从YAML加载
        self._load_config()

    def _init_param_panels(self):
        """初始化各检测方法的参数面板"""
        for method_name, params_def in METHOD_PARAMS.items():
            panel = ParamPanel(
                method_name=method_name,
                params_def=params_def,
                window_name=self.window_name,
                on_change=self._on_panel_change,
            )
            self.param_panels[method_name] = panel

    def _load_config(self):
        """从YAML加载配置"""
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    return

                # 加载当前方法
                if "current_method" in data:
                    try:
                        self.current_method = DetectMethod(data["current_method"])
                    except ValueError:
                        pass

                # 加载启用状态
                if "enabled" in data:
                    self.enabled = data["enabled"]

                # 加载各方法的参数
                if "methods" in data:
                    for method_name, params in data["methods"].items():
                        if method_name in self.param_panels:
                            self.param_panels[method_name].load_params(params)

        except Exception as e:
            print(f"[DebugWindow] Failed to load config: {e}")

    def _save_config(self):
        """保存配置到YAML（节流）"""
        self._save_pending = True

    def _do_save(self):
        """执行保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            methods_data = {}
            for method_name, panel in self.param_panels.items():
                methods_data[method_name] = panel.get_raw_params()

            data = {
                "current_method": self.current_method.value,
                "enabled": self.enabled,
                "methods": methods_data,
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

        except Exception as e:
            print(f"[DebugWindow] Failed to save config: {e}")

    def _on_panel_change(self, method_name: str, params: Dict[str, Any]):
        """参数面板变化回调"""
        # 触发外部回调
        if self.on_params_change:
            self.on_params_change(method_name, params)

        # 保存配置
        self._save_config()

    def update_frame(self, frame: np.ndarray, canny_frame: np.ndarray = None, result_frame: np.ndarray = None):
        """
        更新帧数据

        Args:
            frame: 原始帧
            canny_frame: Canny边缘帧
            result_frame: 检测结果帧
        """
        self._current_frame = frame
        self._canny_frame = canny_frame
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

        # 创建方法选择滑动条
        method_names = [m.value for m in DetectMethod]
        current_idx = method_names.index(self.current_method.value)
        cv2.createTrackbar(
            "Method", self.window_name,
            current_idx, len(method_names) - 1,
            lambda val: self._on_method_change(val)
        )

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

        # 创建当前方法的参数滑动条
        current_panel = self.param_panels.get(self.current_method.value)
        if current_panel:
            current_panel.create_trackbars()

        self._window_created = True

    def _on_method_change(self, value: int):
        """检测方法切换"""
        method_names = [m.value for m in DetectMethod]
        if 0 <= value < len(method_names):
            new_method = DetectMethod(method_names[value])
            if new_method != self.current_method:
                self.current_method = new_method
                self._save_config()

                # 重新创建参数滑动条
                # OpenCV 不支持删除滑动条，所以需要重建窗口
                self._rebuild_window()

    def _rebuild_window(self):
        """重建窗口（切换方法时）"""
        # 销毁旧窗口
        cv2.destroyWindow(self.window_name)
        self._window_created = False

        # 重新创建窗口
        self.setup_window()

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
            self._canny_frame,
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

    def _generate_preview(self, frame: np.ndarray, canny: np.ndarray, result: np.ndarray) -> Optional[np.ndarray]:
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

        elif self.preview_mode == PreviewMode.CANNY:
            if canny is not None:
                if canny.shape[:2] != frame.shape[:2]:
                    canny = cv2.resize(canny, (frame.shape[1], frame.shape[0]))
                return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
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

        # 显示当前方法
        cv2.putText(frame, f"Method: {self.current_method.value}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y += 15

        # 显示当前方法的参数
        panel = self.param_panels.get(self.current_method.value)
        if panel:
            for pdef in panel.params_def:
                value = panel.params.get(pdef.name, 0)
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

    def get_current_method(self) -> DetectMethod:
        """获取当前检测方法"""
        return self.current_method

    def get_params(self, method: DetectMethod = None) -> Dict[str, Any]:
        """
        获取指定方法的参数

        Args:
            method: 检测方法，None 表示当前方法

        Returns:
            参数字典
        """
        target = method or self.current_method
        panel = self.param_panels.get(target.value)
        if panel:
            return panel.get_params()
        return {}

    def is_enabled(self) -> bool:
        """检测是否启用"""
        return self.enabled
