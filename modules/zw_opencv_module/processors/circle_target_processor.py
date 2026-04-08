import cv2
import numpy as np
from typing import  Optional, Dict, Any
from ..circle_target_detector import CircleTargetDetector
from ..circle import CircleTargetItem, CircleTargets

from .base import Processor, VisionResult
#给
class CircleTargetProcessor(Processor):
    def __init__(self, name: str = "circle_target_detect"):
        super().__init__(name)
        self.detector = CircleTargetDetector()
        self.target_max_radius = 10  # 最大圆半径,单位为cm
        self.target_min_radius = 2   # 最小圆半径,单位为cm
        self.target_color : Optional[str] = None  # 目标颜色
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
        if frame is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="Empty frame provided",
            )
        try:
            
            self._frame_height, self._frame_width = frame.shape[:2]
            
            targets = self.detector.detect_circle_targets(frame, target_color=self.target_color)
            
            if targets is not None and len(targets) > 0:
                return VisionResult(
                    task_name=self.name,
                    success=True,
                    result_data={"targets": targets},
                )
            else:    
                return VisionResult(
                    task_name=self.name,
                    success=False,
                    error_message="No circle targets detected",
                    result_data={"targets": []},
                )
                
                
        except Exception as e:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

        
        
    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None:
            return frame
        if result.success and result.result_data and "targets" in result.result_data:
            targets: CircleTargets = result.result_data["targets"]
            for target in targets.targets:
                color = (0, 255, 0)  # 默认绿色
                if target.color == "Red":
                    color = (0, 0, 255)
                elif target.color == "Green":
                    color = (0, 255, 0)
                elif target.color == "Blue":
                    color = (255, 0, 0)
                
                cv2.circle(frame, target.center, target.radius, color, 2)
                cv2.putText(frame, f"{target.color} ({target.radius:.1f}cm)", 
                            (target.center[0] - 20, target.center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_targets(self, frame: np.ndarray, targets: CircleTargets) -> np.ndarray:
        """在帧上绘制检测到的圆形目标"""
        for target in targets.targets:
            color = (0, 255, 0)  # 默认绿色
            if target.color == "Red":
                color = (0, 0, 255)
            elif target.color == "Green":
                color = (0, 255, 0)
            elif target.color == "Blue":
                color = (255, 0, 0)
            
            cv2.circle(frame, target.center, target.radius, color, 2)
            cv2.putText(frame, f"{target.color} ({target.radius:.1f}cm)", 
                        (target.center[0] - 20, target.center[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def _is_target_in_vision_range(self, frame: np.ndarray) -> bool:
        """通过判定摄像头视界内是否存在一个纯黑的矩形来判断目标是否在视野范围内"""
        
        
        