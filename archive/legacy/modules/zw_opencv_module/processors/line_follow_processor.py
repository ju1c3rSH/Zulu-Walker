from typing import Optional

import cv2
import numpy as np

from .registry import register_processor
from .base import Processor, VisionResult
from ..detectors.line_detector import LineDetector


@register_processor("LineFollowProcessor")
class LineFollowProcessor(Processor):
    def __init__(self, name: str = "line_follow"):
        super().__init__(name)
        self.detector = LineDetector()
    
    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        result = self.detector.detect(frame)
        
        return VisionResult(
            task_name=self.name,
            success=result.target_found,
            result_data={
                "target_found": result.target_found,
                "percent_error_x": result.percent_error_x,
                "percent_error_y": 0,
                "coordinate": (result.center_x, result.center_y) if result.target_found else None,
            },
        )
    
    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame
        
        data = result.result_data
        target_found = data.get("target_found", False)
        h, w = frame.shape[:2]
        
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        
        if target_found:
            coord = data.get("coordinate")
            if coord:
                cx, cy = coord
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.line(frame, (int(cx), 0), (int(cx), h), (0, 0, 255), 1)
            
            pe_x = data.get("percent_error_x", 0)
            info = f"ERR x:{pe_x:+d}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "LINE LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        debug = self.detector.get_debug_frame()
        if debug is not None:
            dbg_small = cv2.resize(debug, (160, 120))
            frame[0:120, w-160:w] = dbg_small
        
        return frame
    
    def release(self) -> None:
        pass
