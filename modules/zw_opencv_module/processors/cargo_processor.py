import cv2
import numpy as np
from .base import Processor, VisionResult
from ..detectors.cargo_detector import CargoDetector
from ..models import Color

class TrackCargoProcessor(Processor):
    """货物追踪处理器（stub — 后续用 CargoDetector 实现）"""

    def __init__(self, name: str = "cargo_detect"):
        super().__init__(name)
        self.detector = CargoDetector(name)
        self._frame_width = 640
        self._frame_height = 480
        self._log_interval = 300
        self._getters :dict[str,callable] = {}
        self.target_color :Color = None       
        
        cv2.ocl.setUseOpenCL(True)
        
    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        return VisionResult(
            task_name=self.name,
            success=False,
            error_message="Not implemented",
        )
    
    def draw_result(self, frame, result):
        return super().draw_result(frame, result)
    
    def set_target_color(self,color:Color):
        self.target_color = color
        
    def register_getter(self, key: str, getter: callable):
        """注册一个 lambda，运行时读取外部状态"""
        self._getters[key] = getter

    def get(self, key: str):
        """调用已注入的 lambda 获取状态"""
        getter = self._getters.get(key)
        return getter() if getter else None