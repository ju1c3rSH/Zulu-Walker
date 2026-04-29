# -*- coding: utf-8 -*-
"""
UV 检测调试窗口

独立窗口用于调试 UV 点检测参数，显示 UV 二值化结果。

使用方法:
    python run.py debug --debug-uv
"""
import os
import time
from typing import Dict, Any, Optional, Callable, Tuple

import cv2
import numpy as np
import yaml


class UVDebugWindow:
    """
    UV 检测调试窗口

    功能：
    - UV 颜色阈值参数滑动条
    - UV 二值化预览
    - 参数自动保存到 YAML
    """

    def __init__(
        self,
        config_path: str = None,
        on_params_change: Callable[[Dict[str, Any]], None] = None
    ):
        """
        初始化 UV 调试窗口

        Args:
            config_path: 配置文件路径
            on_params_change: 参数变化回调函数
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "uv_params.yaml"
        )
        self.on_params_change = on_params_change

        # 窗口名称
        self.window_name = "UV Detection Debug"

        # 状态
        self._window_created = False

        # 帧缓冲
        self._current_frame: Optional[np.ndarray] = None
        self._uv_mask: Optional[np.ndarray] = None

        # 保存节流
        self._save_pending = False
        self._last_save_time = 0

        # UV 参数（默认值）
        self.params = {
            # 第一段范围（紫色）- H, S, V 都有 Min 和 Max
            "uv_h_min1": 130,
            "uv_h_max1": 145,
            "uv_s_min1": 90,
            "uv_s_max1": 255,
            "uv_v_min1": 190,
            "uv_v_max1": 255,
            # 第二段范围（蓝紫）- H, S, V 都有 Min 和 Max
            "uv_h_min2": 130,
            "uv_h_max2": 155,
            "uv_s_min2": 0,
            "uv_s_max2": 50,
            "uv_v_min2": 236,
            "uv_v_max2": 255,
            # UV 最小面积阈值
            "uv_min_area": 2,
        }

        # 从 YAML 加载
        self._load_config()

    def _load_config(self):
        """从 YAML 加载配置"""
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
        """保存配置到 YAML（节流）"""
        self._save_pending = True

    def _do_save(self):
        """执行保存"""
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
        """
        创建窗口和滑动条（必须在主线程调用）
        """
        if self._window_created:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        # 创建滑动条
        # 第一段范围 - H, S, V 都有 Min 和 Max
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

        # 第二段范围 - H, S, V 都有 Min 和 Max
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

        # UV 最小面积阈值
        cv2.createTrackbar("UV MinArea", self.window_name,
                           self.params["uv_min_area"], 100,
                           lambda val: self._on_trackbar("uv_min_area", val))

        self._window_created = True

        # 初始化时触发一次回调
        if self.on_params_change:
            self.on_params_change(self.get_color_ranges(), self.params["uv_min_area"])

    def _on_trackbar(self, name: str, value: int):
        """滑动条回调"""
        self.params[name] = value

        # 触发回调（传递颜色范围和最小面积）
        if self.on_params_change:
            self.on_params_change(self.get_color_ranges(), self.params["uv_min_area"])

        # 保存配置
        self._save_config()

    def update_frame(self, frame: np.ndarray, uv_mask: np.ndarray = None):
        """
        更新帧数据

        Args:
            frame: 原始帧（BGR）
            uv_mask: UV mask（可选，如果不提供则自动计算）
        """
        self._current_frame = frame
        self._uv_mask = uv_mask

    def update_gui(self):
        """
        更新 GUI 显示（必须在主线程调用）
        """
        if not self._window_created:
            return

        # 处理保存
        if self._save_pending and time.time() - self._last_save_time > 0.5:
            self._do_save()
            self._save_pending = False
            self._last_save_time = time.time()

        # 生成预览
        preview = self._generate_preview()

        if preview is not None:
            cv2.imshow(self.window_name, preview)

    def _generate_preview(self) -> Optional[np.ndarray]:
        """生成预览图像"""
        if self._current_frame is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        frame = self._current_frame

        # 降采样到 320x240
        h, w = frame.shape[:2]
        scale = min(320 / w, 240 / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # 计算 UV mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        uv_ranges = self.get_color_ranges()

        mask = None
        for lower, upper in uv_ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 显示 UV mask（彩色叠加）
        preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在 mask 上绘制检测到的 UV 点轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.params["uv_min_area"]:
                # 绘制轮廓（绿色）
                cv2.drawContours(preview, [cnt], -1, (0, 255, 0), 2)
                # 计算并绘制质心（红色）
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(preview, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制参数信息
        self._draw_params_info(preview)

        return preview

    def _draw_params_info(self, frame: np.ndarray):
        """在预览上绘制参数信息"""
        y = 20

        # 显示第一段范围 - H, S, V 都有 Min 和 Max
        cv2.putText(frame,
                    f"R1: H[{self.params['uv_h_min1']}-{self.params['uv_h_max1']}] "
                    f"S[{self.params['uv_s_min1']}-{self.params['uv_s_max1']}] "
                    f"V[{self.params['uv_v_min1']}-{self.params['uv_v_max1']}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y += 15

        # 显示第二段范围 - H, S, V 都有 Min 和 Max
        cv2.putText(frame,
                    f"R2: H[{self.params['uv_h_min2']}-{self.params['uv_h_max2']}] "
                    f"S[{self.params['uv_s_min2']}-{self.params['uv_s_max2']}] "
                    f"V[{self.params['uv_v_min2']}-{self.params['uv_v_max2']}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y += 15

        # 显示最小面积
        cv2.putText(frame,
                    f"Min Area: {self.params['uv_min_area']}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def get_color_ranges(self) -> list:
        """
        获取 UV 颜色范围列表

        Returns:
            [(lower, upper), ...] 格式的颜色范围列表
        """
        return [
            # 第一段范围（紫色）- H, S, V 都有 Min 和 Max
            (np.array([self.params["uv_h_min1"], self.params["uv_s_min1"], self.params["uv_v_min1"]]),
             np.array([self.params["uv_h_max1"], self.params["uv_s_max1"], self.params["uv_v_max1"]])),
            # 第二段范围（蓝紫）- H, S, V 都有 Min 和 Max
            (np.array([self.params["uv_h_min2"], self.params["uv_s_min2"], self.params["uv_v_min2"]]),
             np.array([self.params["uv_h_max2"], self.params["uv_s_max2"], self.params["uv_v_max2"]])),
        ]

    def get_params(self) -> Dict[str, Any]:
        """
        获取所有参数

        Returns:
            参数字典
        """
        return self.params.copy()

    def destroy_window(self):
        """销毁窗口"""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False