# -*- coding: utf-8 -*-
"""
CargoProcessor - 货物检测处理器

封装 CargoDetector，实现目标颜色过滤和坐标偏差计算。
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any
from .base import Processor, VisionResult

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cargo_detector import CargoDetector
from cargos import CargoItem, Cargos


class CargoProcessor(Processor):
    """货物检测处理器"""

    def __init__(self, name: str = "cargo_detect"):
        super().__init__(name)
        self.detector = CargoDetector()
        self.target_color: Optional[str] = None
        self._frame_width: int = 640
        self._frame_height: int = 480

    def set_target_color(self, color: Optional[str]):
        """
        设置目标颜色

        Args:
            color: 颜色名称 ('Red', 'Green', 'Blue') 或 None 表示检测所有颜色
        """
        self.target_color = color

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """
        检测货物并计算坐标偏差

        Args:
            frame: 输入图像帧
            context: 上下文（未使用）

        Returns:
            VisionResult: 包含检测结果和坐标偏差
        """
        if frame is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="Empty frame provided",
            )

        try:
            # 更新帧尺寸
            self._frame_height, self._frame_width = frame.shape[:2]

            # 检测货物（传入目标颜色进行过滤）
            cargos = self.detector.detect_cargo_shape(frame, target_color=self.target_color)

            # 找到目标货物
            target_cargo = self._find_target_cargo(cargos)

            if target_cargo is None:
                error_msg = f"Target color cargo not found"
                if self.target_color:
                    error_msg = f"{self.target_color} color cargo not found"
                return VisionResult(
                    task_name=self.name,
                    result_data={
                        "cargos": cargos,
                        "target_color": self.target_color,
                        "percent_error_x": 0,
                        "percent_error_y": 0,
                        "target_found": False,
                    },
                    success=False,
                    error_message=error_msg,
                )

            # 计算坐标偏差
            percent_error_x, percent_error_y = self._calculate_position_error(target_cargo)

            return VisionResult(
                task_name=self.name,
                result_data={
                    "cargos": cargos,
                    "target_cargo": target_cargo,
                    "target_color": self.target_color,
                    "percent_error_x": percent_error_x,
                    "percent_error_y": percent_error_y,
                    "target_found": True,
                },
                success=True,
            )

        except Exception as e:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

    def _find_target_cargo(self, cargos: Cargos) -> Optional[CargoItem]:
        if not cargos.payload:
            return None

        # 如果没有设置目标颜色，返回面积最大的货物
        if self.target_color is None:
            return max(cargos.payload, key=lambda c: c.area or 0)

        # 根据目标颜色过滤（CargoItem 已有 color 属性）
        matching = [c for c in cargos.payload if c.color == self.target_color]
        if matching:
            return max(matching, key=lambda c: c.area or 0)
        return None

    def _find_cargo_by_color(self, cargos: Cargos, color: str) -> Optional[CargoItem]:
        """已废弃，颜色过滤在检测阶段完成"""
        matching = [c for c in cargos.payload if c.color == color]
        if matching:
            return max(matching, key=lambda c: c.area or 0)
        return None

    def _calculate_position_error(self, cargo: CargoItem) -> tuple:
        """
        计算货物相对于屏幕中心的坐标偏差

        Args:
            cargo: 目标货物

        Returns:
            (percent_error_x, percent_error_y): 归一化到 [-100, 100] 的坐标偏差
        """
        # 屏幕中心
        center_x = self._frame_width // 2
        center_y = self._frame_height // 2

        # 货物中心
        cargo_x, cargo_y = cargo.center_coordinates

        # 像素偏差
        pixel_error_x = cargo_x - center_x
        pixel_error_y = cargo_y - center_y

        # 归一化到 [-100, 100]
        percent_error_x = int((pixel_error_x * 200) / self._frame_width)
        percent_error_y = int((pixel_error_y * 200) / self._frame_height)

        return percent_error_x, percent_error_y

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """
        在帧上绘制检测结果

        Args:
            frame: 输入图像帧
            result: 检测结果

        Returns:
            np.ndarray: 绘制后的帧
        """
        if frame is None:
            return frame

        if result.result_data is None:
            return frame

        cargos = result.result_data.get("cargos")
        target_cargo = result.result_data.get("target_cargo")
        target_found = result.result_data.get("target_found", False)

        # 绘制所有检测到的货物轮廓
        if cargos and cargos.payload:
            for cargo in cargos.payload:
                self._draw_cargo(frame, cargo, is_target=(cargo == target_cargo))

        # 绘制屏幕中心十字线
        self._draw_center_cross(frame)

        # 绘制状态信息
        self._draw_status_info(frame, result, target_found)

        return frame

    def _draw_cargo(self, frame: np.ndarray, cargo: CargoItem, is_target: bool = False):
        """绘制单个货物"""
        color = (0, 255, 0) if is_target else (128, 128, 128)
        thickness = 2 if is_target else 1

        # 绘制轮廓
        if cargo.contour_points is not None:
            cv2.drawContours(frame, [cargo.contour_points], -1, color, thickness)

        # 绘制中心点
        cx, cy = cargo.center_coordinates
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # 绘制边界框
        if cargo.bounding_box:
            x_min, y_min, x_max, y_max = cargo.bounding_box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

        # 如果是目标货物，绘制坐标偏差
        if is_target:
            # 绘制从屏幕中心到货物中心的连线
            center_x = self._frame_width // 2
            center_y = self._frame_height // 2
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

    def _draw_center_cross(self, frame: np.ndarray):
        """绘制屏幕中心十字线"""
        center_x = self._frame_width // 2
        center_y = self._frame_height // 2

        # 水平线
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        # 垂直线
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

    def _draw_status_info(self, frame: np.ndarray, result: VisionResult, target_found: bool):
        """绘制状态信息"""
        # 背景
        cv2.rectangle(frame, (5, 5), (350, 100), (0, 0, 0), -1)

        # 目标颜色
        target_color = result.result_data.get("target_color", "Any")
        color_text = f"Target: {target_color or 'Any'}"
        cv2.putText(frame, color_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if target_found:
            # 坐标偏差
            error_x = result.result_data.get("percent_error_x", 0)
            error_y = result.result_data.get("percent_error_y", 0)
            error_text = f"Error: X={error_x:+d}, Y={error_y:+d}"
            cv2.putText(frame, error_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # 检测到的货物数量
            cargos = result.result_data.get("cargos")
            if cargos:
                count_text = f"Cargos: {cargos.number}"
                cv2.putText(frame, count_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Target not found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
