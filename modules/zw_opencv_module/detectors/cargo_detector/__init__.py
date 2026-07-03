
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque

from ...models.color import Color
from ...models.cargo import CargoItem
from .detection import DetectMethod


class CargoDetector:
    def __init__(self, name: str = "cargo_detect"):
        self.name = name

        self.detect_method = DetectMethod.FAST_CIRCLE

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf_q_base = 0.3
        self._kf_q_vel_base = 0.3

        self.tracking_initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self._last_predict_time = None

        self.roi_size = 150
        self.max_roi_miss = 5
        self.min_circularity = 0.75
        self.min_area = 100
        self.smooth_window = 5
        self.kernel_open = 5
        self.kernel_close = 7

        self.last_center = None
        self.roi_miss_count = 0
        self._last_mask = None
        self._last_morphed = None

        self._center_history = deque(maxlen=self.smooth_window)
        self._radius_history = deque(maxlen=self.smooth_window)

        self.color_ranges = {
            Color.RED: [
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            Color.GREEN: [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            Color.BLUE: [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        }

        self._methods: Dict[DetectMethod, Any] = {}
        self._init_detection_methods()

    def _init_detection_methods(self):
        from .detection.methods.fast_circle import FastCircleDetectionWithColorMethod
        self._methods = {
            DetectMethod.FAST_CIRCLE: FastCircleDetectionWithColorMethod(detector=self),
        }

    def set_detect_method(self, method: DetectMethod):
        self.detect_method = method

    def get_detect_method(self) -> DetectMethod:
        return self.detect_method

    @staticmethod
    def get_supported_methods():
        return [DetectMethod.FAST_CIRCLE]

    def get_method_params(self, method: DetectMethod) -> dict:
        return {
            "roi_size": self.roi_size,
            "max_roi_miss": self.max_roi_miss,
            "min_circularity": self.min_circularity,
            "min_area": self.min_area,
            "kernel_open": self.kernel_open,
            "kernel_close": self.kernel_close,
            "smooth_window": self.smooth_window,
        }

    def update_params(self, params: dict):
        for key in ("roi_size", "max_roi_miss", "min_area", "kernel_open", "kernel_close", "smooth_window"):
            if key in params:
                setattr(self, key, int(params[key]))
        if "min_circularity" in params:
            self.min_circularity = float(params["min_circularity"])

    def detect_cargo(self, frame, target_color: Color) -> Optional[CargoItem]:
        method = self._methods.get(self.detect_method)
        if method is not None:
            return method.detect(frame, target_color)
        return None