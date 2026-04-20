# -*- coding: utf-8 -*-
"""
参数面板模块

提供参数定义和滑动条管理功能，用于调试窗口。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import cv2


@dataclass
class ParamDef:
    """参数定义"""
    name: str           # 参数名（与detector属性对应）
    display_name: str   # 显示名称
    default: int        # 默认值
    min_val: int        # 最小值
    max_val: int        # 最大值
    step: int = 1       # 步进
    scale: float = 1.0  # 缩放因子（实际值 = 滑动条值 * scale）
    is_odd: bool = False  # 是否必须为奇数（如核大小）


# 各检测方法的参数定义
METHOD_PARAMS: Dict[str, List[ParamDef]] = {
    "contour_ellipse": [
        ParamDef("min_area_threshold", "Min Area", 150, 10, 2000, 10),
        ParamDef("min_contour_points", "Min Points", 15, 5, 50, 1),
    ],
    "edge_contour_ellipse": [
        ParamDef("edge_canny_threshold1", "Canny Th1", 50, 0, 255, 1),
        ParamDef("edge_canny_threshold2", "Canny Th2", 150, 0, 255, 1),
        ParamDef("morph_type", "Morph Type", 1, 0, 4, 1),
        ParamDef("morph_kernel", "Morph Kernel", 3, 1, 15, 2, is_odd=True),
        ParamDef("morph_iterations", "Morph Iter", 1, 1, 10, 1),
        ParamDef("blur_kernel", "Blur Kernel", 5, 1, 15, 2, is_odd=True),
        ParamDef("blur_sigma", "Blur Sigma", 10, 1, 50, 1, scale=0.1),
        ParamDef("min_area_threshold", "Min Area", 150, 10, 2000, 10),
        ParamDef("min_contour_points", "Min Points", 15, 5, 50, 1),
    ],
    "test_line_quad": [
        ParamDef("blur_kernel", "Blur Kernel", 5, 1, 15, 2, is_odd=True),
        ParamDef("blur_sigma", "Blur Sigma", 10, 1, 50, 1, scale=0.1),
        ParamDef("min_area_threshold", "Min Area", 150, 10, 2000, 10),
        ParamDef("min_contour_points", "Min Points", 15, 5, 50, 1),
    ],
}


class ParamPanel:
    """
    参数面板类

    管理一组参数定义和对应的滑动条。
    """

    def __init__(
        self,
        method_name: str,
        params_def: List[ParamDef],
        window_name: str,
        on_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        初始化参数面板

        Args:
            method_name: 方法名称
            params_def: 参数定义列表
            window_name: OpenCV 窗口名称
            on_change: 参数变化回调 (method_name, params_dict)
        """
        self.method_name = method_name
        self.params_def = params_def
        self.window_name = window_name
        self.on_change = on_change

        # 参数值
        self.params: Dict[str, int] = {}
        for pdef in params_def:
            self.params[pdef.name] = pdef.default

        # 滑动条是否已创建
        self._trackbars_created = False

    def load_params(self, params: Dict[str, int]):
        """从字典加载参数值"""
        for name, value in params.items():
            if name in self.params:
                # 查找参数定义，反向应用 scale
                pdef = self.get_param_def(name)
                if pdef and pdef.scale != 1.0:
                    self.params[name] = int(value / pdef.scale)
                else:
                    self.params[name] = int(value)

    def create_trackbars(self):
        """创建滑动条（必须在主线程调用）"""
        if self._trackbars_created:
            return

        for pdef in self.params_def:
            cv2.createTrackbar(
                pdef.display_name,
                self.window_name,
                int(self.params[pdef.name]),
                int(pdef.max_val),
                lambda val, d=pdef: self._on_trackbar(d, val)
            )

        self._trackbars_created = True

    def _on_trackbar(self, pdef: ParamDef, value: int):
        """滑动条回调"""
        # 特殊处理：核大小必须为奇数
        if pdef.is_odd and value % 2 == 0:
            value = max(pdef.min_val, value - 1)
            cv2.setTrackbarPos(pdef.display_name, self.window_name, int(value))

        if self.params.get(pdef.name) != value:
            self.params[pdef.name] = value

            # 触发回调
            if self.on_change:
                self.on_change(self.method_name, self.get_params())

    def get_params(self) -> Dict[str, Any]:
        """
        获取当前参数（应用缩放因子）

        Returns:
            参数字典（包含转换后的值）
        """
        result = {}
        for pdef in self.params_def:
            value = self.params[pdef.name]
            # 应用缩放因子
            if pdef.scale != 1.0:
                result[pdef.name] = value * pdef.scale
            else:
                result[pdef.name] = value
        return result

    def get_raw_params(self) -> Dict[str, int]:
        """获取原始参数值（用于持久化）"""
        return self.params.copy()

    def set_param(self, name: str, value: int):
        """设置单个参数值"""
        if name in self.params:
            self.params[name] = int(value)
            # 更新滑动条位置
            for pdef in self.params_def:
                if pdef.name == name:
                    cv2.setTrackbarPos(pdef.display_name, self.window_name, int(value))
                    break

    def get_param_def(self, name: str) -> Optional[ParamDef]:
        """获取参数定义"""
        for pdef in self.params_def:
            if pdef.name == name:
                return pdef
        return None
