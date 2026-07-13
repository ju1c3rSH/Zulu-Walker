import gc
import os
import cv2
import numpy as np
from .base import Processor, VisionResult


_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ai", "qrcode"
)


def _try_create_wechat_detector():
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return None
    detect_pt = os.path.join(_MODEL_DIR, "detect.prototxt")
    detect_cm = os.path.join(_MODEL_DIR, "detect.caffemodel")
    if not os.path.isfile(detect_pt):
        print(f"[QRCodeProcessor] WeChat QR model file missing: {detect_pt}")
        return None
    if not os.path.isfile(detect_cm):
        print(f"[QRCodeProcessor] WeChat QR model file missing: {detect_cm}")
        return None
    sr_pt = os.path.join(_MODEL_DIR, "sr.prototxt")
    sr_cm = os.path.join(_MODEL_DIR, "sr.caffemodel")
    try:
        return cv2.wechat_qrcode_WeChatQRCode(
            detect_pt, detect_cm, sr_pt, sr_cm
        )
    except TypeError:
        pass
    except Exception as e:
        print(f"[QRCodeProcessor] Failed to load WeChatQRCode: {e}")
        return None
    try:
        return cv2.wechat_qrcode_WeChatQRCode(detect_pt, detect_cm)
    except Exception as e:
        print(f"[QRCodeProcessor] Failed to load WeChatQRCode: {e}")
        return None


class QRCodeProcessor(Processor):
    """QR 码解码处理器（WeChatQRCode / 回退 QRCodeDetector）"""

    def __init__(self, name: str):
        super().__init__(name)
        self._detector = None
        self._detector_loaded = False
        self._use_wechat = False

    def _ensure_detector(self):
        if self._detector_loaded:
            return
        self._detector_loaded = True
        detector = _try_create_wechat_detector()
        if detector is not None:
            self._detector = detector
            self._use_wechat = True
            print(f"[QRCodeProcessor] Using WeChatQRCode detector")
        else:
            self._detector = cv2.QRCodeDetector()
            self._use_wechat = False
            print(f"[QRCodeProcessor] WeChatQRCode unavailable, using QRCodeDetector fallback")

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        self._ensure_detector()
        try:
            if self._use_wechat:
                return self._process_wechat(frame)
            return self._process_fallback(frame)
        except Exception as e:
            return VisionResult(
                task_name=self.name,
                result_data={},
                success=False,
                error_message=str(e),
            )

    def _process_wechat(self, frame: np.ndarray) -> VisionResult:
        results, points_list = self._detector.detectAndDecode(frame)
        result = {}
        if results and len(results) > 0:
            data = results[0]
            points = None
            if points_list and len(points_list) > 0:
                pts = points_list[0]
                if pts is not None and len(pts) > 0:
                    pts_i = np.array(pts, dtype=np.int32)
                    for i in range(len(pts_i)):
                        cv2.line(
                            frame,
                            tuple(pts_i[i]),
                            tuple(pts_i[(i + 1) % len(pts_i)]),
                            (0, 255, 0), 2,
                        )
                    cv2.putText(
                        frame, data,
                        tuple(pts_i[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                    )
                    points = pts_i
            result["qr_data"] = data
            if points is not None:
                result["qr_points"] = points.tolist()
            return VisionResult(
                task_name=self.name,
                result_data={"result": result},
                success=True,
            )
        return VisionResult(
            task_name=self.name,
            result_data={"result": result},
            success=False,
            error_message="No QR code detected",
        )

    def _process_fallback(self, frame: np.ndarray) -> VisionResult:
        data, points, _ = self._detector.detectAndDecode(frame)
        result = {}
        if points is not None and data:
            points = points[0].astype(int)
            for i in range(len(points)):
                cv2.line(
                    frame,
                    tuple(points[i]),
                    tuple(points[(i + 1) % len(points)]),
                    (0, 255, 0), 2,
                )
            cv2.putText(
                frame, data,
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )
            result["qr_data"] = data
            result["qr_points"] = points.tolist()
        return VisionResult(
            task_name=self.name,
            result_data={"result": result},
            success=bool(data),
            error_message="" if data else "No QR code detected",
        )

    def release(self) -> None:
        self._detector = None
        self._detector_loaded = False
        gc.collect()
