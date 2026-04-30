# -*- coding: utf-8 -*-
"""检测方法基类"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ...circle import CircleTargets
from ..detect_method import DetectMethod


class DetectionMethodBase(ABC):
    """检测方法抽象基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        target_color: Optional[str] = None,
    ) -> CircleTargets:
        """
        执行检测

        Args:
            frame: BGR 格式输入图像
            target_color: 目标颜色

        Returns:
            CircleTargets: 检测结果
        """
        pass

    @property
    @abstractmethod
    def method(self) -> DetectMethod:
        """返回对应的检测方法枚举"""
        pass

    def get_edge_preview(self) -> Optional[np.ndarray]:
        """获取边缘预览图像（可选实现）"""
        return None
