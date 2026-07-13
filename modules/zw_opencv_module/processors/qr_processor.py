import gc
import cv2
import numpy as np
from .base import Processor, VisionResult

try:
    import pyzbar.pyzbar as pyzbar
    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False


class QRCodeProcessor(Processor):
    """QR 码解码处理器（pyzbar 主解码 + cv2.QRCodeDetector 兜底）"""

    def __init__(self, name: str):
        super().__init__(name)

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        try:
            if _HAS_PYZBAR:
                result = self._process_with_pyzbar(frame)
                if result.success:
                    return result

            result = self._process_fallback(frame)
            if result.success:
                return result

            return VisionResult(
                task_name=self.name,
                result_data={"result": {}},
                success=False,
                error_message="No QR code detected",
            )

        except Exception as e:
            return VisionResult(
                task_name=self.name,
                result_data={"result": {}},
                success=False,
                error_message=str(e),
            )

    def _process_with_pyzbar(self, frame: np.ndarray) -> VisionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)

        if not decoded:
            return VisionResult(
                task_name=self.name,
                result_data={"result": {}},
                success=False,
            )

        qr_codes = [obj for obj in decoded if obj.type == "QRCODE"]
        if not qr_codes:
            return VisionResult(
                task_name=self.name,
                result_data={"result": {}},
                success=False,
            )

        obj = qr_codes[0]
        data = obj.data.decode("utf-8")
        polygon = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.int32)

        for i in range(len(polygon)):
            cv2.line(
                frame,
                tuple(polygon[i]),
                tuple(polygon[(i + 1) % len(polygon)]),
                (0, 255, 0), 2,
            )
        cv2.putText(
            frame, data,
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
        )

        result = {
            "qr_data": data,
            "qr_points": polygon.tolist(),
        }
        return VisionResult(
            task_name=self.name,
            result_data={"result": result},
            success=True,
        )

    def _process_fallback(self, frame: np.ndarray) -> VisionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(sharpened)

        if points is not None and data:
            pts = points[0].astype(int)
            for i in range(len(pts)):
                cv2.line(
                    frame,
                    tuple(pts[i]),
                    tuple(pts[(i + 1) % len(pts)]),
                    (0, 255, 0), 2,
                )
            cv2.putText(
                frame, data,
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )
            result = {
                "qr_data": data,
                "qr_points": pts.tolist(),
            }
            return VisionResult(
                task_name=self.name,
                result_data={"result": result},
                success=True,
            )

        return VisionResult(
            task_name=self.name,
            result_data={"result": {}},
            success=False,
            error_message="No QR code detected",
        )

    def release(self) -> None:
        gc.collect()
