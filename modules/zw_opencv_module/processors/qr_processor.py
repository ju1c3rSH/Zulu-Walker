
import cv2
import numpy as np
from .base import Processor, VisionResult


class QRCodeProcessor(Processor):
    """QR 码解码处理器"""

    def __init__(self, name: str):
        super().__init__(name)
        self.qr_detector = cv2.QRCodeDetector()

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        try:
            data, points, _ = self.qr_detector.detectAndDecode(frame)

            result = {}
            if points is not None and data:
                points = points[0].astype(int)
                for i in range(len(points)):
                    cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 2)
                result['qr_data'] = data
                result['qr_points'] = points.tolist()

            return VisionResult(
                task_name=self.name,
                result_data={"result": result},
                success=bool(data),
                error_message="" if data else "No QR code detected",
            )
        except Exception as e:
            return VisionResult(
                task_name=self.name,
                result_data={},
                success=False,
                error_message=str(e),
            )