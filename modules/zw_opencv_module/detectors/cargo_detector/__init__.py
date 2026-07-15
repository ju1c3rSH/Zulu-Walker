
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque

try:
    cv2.ximgproc.createEdgeDrawing
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False

from ...models.color import Color
from ...models.cargo import CargoItem
from .detection import DetectMethod


class _TrackingState:

    __slots__ = (
        "kf", "tracking_initialized", "lost_frames", "_last_predict_time",
        "last_center", "roi_miss_count", "_center_history", "_radius_history",
        "_ema_s", "_ema_v",
    )

    def __init__(self, smooth_window: int = 5):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.tracking_initialized = False
        self.lost_frames = 0
        self._last_predict_time = None

        self.last_center: Optional[Tuple[float, float]] = None
        self.roi_miss_count = 0

        self._center_history: deque = deque(maxlen=smooth_window)
        self._radius_history: deque = deque(maxlen=smooth_window)

        self._ema_s = None
        self._ema_v = None

    def resize_histories(self, smooth_window: int):
        self._center_history = deque(self._center_history, maxlen=smooth_window)
        self._radius_history = deque(self._radius_history, maxlen=smooth_window)

    def reset_prediction(self):
        self.last_center = None
        self.roi_miss_count = 0
        self.tracking_initialized = False
        self.lost_frames = 0
        self._last_predict_time = None
        self._center_history.clear()
        self._radius_history.clear()
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)


class CargoDetector:
    def __init__(self, name: str = "cargo_detect"):
        self.name = name

        # EdgeDrawing 圆检测参数（必须在 _init_edge_drawing 之前赋值，_update_ed_params 会读取）
        self.blur_kernel = 5
        self.blur_sigma = 1.5
        self.ed_min_path_length = 50
        self.ed_gradient_threshold = 36
        self.ed_nfa_validation = True
        self.edge_morph_kernel = 3
        self.edge_morph_iterations = 1
        self.color_match_threshold = 0.35

        # EdgeDrawing 初始化（若不可用则默认使用 FAST_CIRCLE）
        self.ed = None
        self._init_edge_drawing()
        self.detect_method = (
            DetectMethod.EDGE_DRAWING_CIRCLE
            if self.ed is not None
            else DetectMethod.FAST_CIRCLE
        )

        self._kf_q_base = 0.2
        self._kf_q_vel_base = 0.15
        self.max_lost_frames = 10

        self.roi_size = 150
        self.max_roi_miss = 5
        self.min_circularity = 0.75
        self.min_area = 100
        self.smooth_window = 5
        self.kernel_open = 5
        self.kernel_close = 7

        self._last_mask = None
        self._last_morphed = None
        self._last_edge_preview = None

        # 按颜色隔离的跟踪状态
        self._tracking: Dict[Color, _TrackingState] = {}
        self._active_tracking: Optional[_TrackingState] = None
        self._active_color: Optional[Color] = None

        self.color_ranges = {
            Color.RED: [
                (np.array([0, 15, 0]), np.array([10, 255, 255])),
                (np.array([170, 15, 0]), np.array([180, 255, 255])),
            ],
            Color.GREEN: [(np.array([40, 15, 15]), np.array([80, 255, 255]))],
            Color.BLUE: [(np.array([100, 15, 15]), np.array([130, 255, 255]))],
        }

        self._methods: Dict[DetectMethod, Any] = {}
        self._init_detection_methods()

    def _init_edge_drawing(self):
        if not _HAS_XIMGPROC:
            return
        self.ed = cv2.ximgproc.createEdgeDrawing()
        self._update_ed_params()

    def _update_ed_params(self):
        if self.ed is None:
            return
        ed_params = self.ed.Params()
        ed_params.MinPathLength = self.ed_min_path_length
        ed_params.GradientThresholdValue = self.ed_gradient_threshold
        ed_params.NFAValidation = self.ed_nfa_validation
        self.ed.setParams(ed_params)

    def _init_detection_methods(self):
        from .detection.methods.fast_circle import FastCircleDetectionWithColorMethod
        from .detection.methods.edge_drawing_circle import EdgeDrawingCircleMethod
        self._methods = {
            DetectMethod.FAST_CIRCLE: FastCircleDetectionWithColorMethod(detector=self),
            DetectMethod.EDGE_DRAWING_CIRCLE: EdgeDrawingCircleMethod(detector=self),
        }

    def _get_tracking(self, color: Color) -> _TrackingState:
        if color not in self._tracking:
            self._tracking[color] = _TrackingState(smooth_window=self.smooth_window)
        ts = self._tracking[color]
        self._active_tracking = ts
        self._active_color = color
        return ts

    def set_detect_method(self, method: DetectMethod) -> bool:
        if method == self.detect_method:
            return True
        if method == DetectMethod.EDGE_DRAWING_CIRCLE and self.ed is None:
            print("EdgeDrawing method is not available currently, please check your python lib installation.")
            return False
        self.detect_method = method
        self._reset_all_tracking()
        return True

    def _reset_all_tracking(self):
        for ts in self._tracking.values():
            ts.reset_prediction()
        self._active_tracking = None
        self._active_color = None

    def is_edge_drawing_available(self) -> bool:
        return self.ed is not None

    def get_detect_method(self) -> DetectMethod:
        return self.detect_method

    @staticmethod
    def get_supported_methods():
        return [DetectMethod.FAST_CIRCLE, DetectMethod.EDGE_DRAWING_CIRCLE]

    def get_method_params(self, method: DetectMethod) -> dict:
        return {
            "roi_size": self.roi_size,
            "max_roi_miss": self.max_roi_miss,
            "min_circularity": self.min_circularity,
            "min_area": self.min_area,
            "kernel_open": self.kernel_open,
            "kernel_close": self.kernel_close,
            "smooth_window": self.smooth_window,
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "ed_min_path_length": self.ed_min_path_length,
            "ed_gradient_threshold": self.ed_gradient_threshold,
            "ed_nfa_validation": self.ed_nfa_validation,
            "edge_morph_kernel": self.edge_morph_kernel,
            "edge_morph_iterations": self.edge_morph_iterations,
            "color_match_threshold": self.color_match_threshold,
        }

    def update_params(self, params: dict):
        int_keys = (
            "roi_size", "max_roi_miss", "min_area", "kernel_open", "kernel_close",
            "smooth_window", "blur_kernel", "ed_min_path_length",
            "ed_gradient_threshold", "edge_morph_kernel", "edge_morph_iterations",
        )
        for key in int_keys:
            if key in params:
                setattr(self, key, int(params[key]))

        float_keys = ("min_circularity", "blur_sigma", "color_match_threshold")
        for key in float_keys:
            if key in params:
                setattr(self, key, float(params[key]))

        if "ed_nfa_validation" in params:
            self.ed_nfa_validation = bool(params["ed_nfa_validation"])

        for ts in self._tracking.values():
            ts.resize_histories(self.smooth_window)

        self._update_ed_params()

    def detect_cargo(self, frame, target_color: Color) -> Optional[CargoItem]:
        method = self._methods.get(self.detect_method)
        if method is not None:
            return method.detect(frame, target_color)
        return None
