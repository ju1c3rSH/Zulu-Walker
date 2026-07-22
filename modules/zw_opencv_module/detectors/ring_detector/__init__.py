import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque
from utils.log_util import log_print

try:
    cv2.ximgproc.createEdgeDrawing
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False

from ...models.color import Color
from ...models.ring import RingTarget
from .detection import RingDetectMethod

_METHOD_TO_CONFIG_KEY = {
    RingDetectMethod.FAST_RING: "FAST_RING",
    RingDetectMethod.EDGE_DRAWING_RING: "EDGE_DRAWING_RING",
    RingDetectMethod.HEURISTIC_RING: "HEURISTIC_RING",
}


class _TrackingState:

    __slots__ = (
        "kf", "tracking_initialized", "lost_frames", "_last_predict_time",
        "last_center", "roi_miss_count", "_center_history",
        "_ring_outer_radius",
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

    def resize_histories(self, smooth_window: int):
        self._center_history = deque(self._center_history, maxlen=smooth_window)

    def reset_prediction(self):
        self.last_center = None
        self.roi_miss_count = 0
        self.tracking_initialized = False
        self.lost_frames = 0
        self._last_predict_time = None
        self._center_history.clear()
        self._ring_outer_radius = None
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)


class RingDetector:

    def __init__(self, name: str = "ring_detect"):
        self.name = name

        self.ed = None
        # ED params must be assigned BEFORE _init_edge_drawing() calls _update_ed_params()
        self.ed_min_path_length = 20
        self.ed_gradient_threshold = 36
        self.ed_nfa_validation = False
        self.edge_morph_kernel = 5
        self.edge_morph_iterations = 2
        self._init_edge_drawing()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if _HAS_XIMGPROC:
            self.detect_method = RingDetectMethod.HEURISTIC_RING
        else:
            self.detect_method = RingDetectMethod.FAST_RING

        self.q_base = 0.2
        self.q_vel_base = 0.15
        self.max_lost_frames = 10
        self.roi_size = 350
        self.max_roi_miss = 5
        self.min_area = 1500
        self.smooth_window = 5
        self.kalman_enabled = False
        self.force_global = False
        self._ring_log_frame = 0
        self.ring_gap_px = 10
        self.max_outer_radius = 300

        self.blur_kernel = 3
        self.blur_sigma = 1.5

        self._last_mask = None
        self._last_morphed = None
        self._last_edge_preview = None
        self._last_alt_img = None
        self.force_stage = 0
        self.color_blob_min_area = 80

        self._tracking: Dict[Color, _TrackingState] = {}
        self._active_tracking: Optional[_TrackingState] = None
        self._active_color: Optional[Color] = None
        self._last_ring_meta: Dict[Color, dict] = {}

        self.color_ranges = {
            Color.RED: [
                (np.array([0, 60, 60]), np.array([6, 255, 255])),
                (np.array([156, 60, 60]), np.array([180, 255, 255])),
            ],
            Color.GREEN: [(np.array([40, 60, 45]), np.array([80, 255, 255]))],
            Color.BLUE: [(np.array([100, 40, 35]), np.array([140, 255, 255]))],
        }

        self._methods: Dict[RingDetectMethod, Any] = {}
        self._init_detection_methods()
        self._load_params_from_config()

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
        from .detection.methods.fast_ring import FastRingMethod
        from .detection.methods.edge_drawing_ring import EdgeDrawingRingMethod
        from .detection.methods.heuristic_ring import HeuristicRingMethod
        self._methods = {
            RingDetectMethod.FAST_RING: FastRingMethod(detector=self),
            RingDetectMethod.EDGE_DRAWING_RING: EdgeDrawingRingMethod(detector=self),
            RingDetectMethod.HEURISTIC_RING: HeuristicRingMethod(detector=self),
        }

    def _load_params_from_config(self):
        try:
            from .debug.config import RingConfig
            import os
            config = RingConfig()
            data = config.load()

            method_key = _METHOD_TO_CONFIG_KEY.get(self.detect_method)
            sections = [method_key, "SHARED"] if method_key else ["SHARED"]

            for section in sections:
                if section not in data or not isinstance(data[section], dict):
                    continue
                defs = config.get_param_defs(section)
                pdef_map = {p.name: p for p in defs}
                for param_name, raw_value in data[section].items():
                    pdef = pdef_map.get(param_name)
                    if pdef is None:
                        continue
                    actual = raw_value * pdef.scale
                    if pdef.scale == 1.0:
                        actual = int(actual)
                    if hasattr(self, param_name):
                        setattr(self, param_name, actual)

            self._update_ed_params()
            log_print(f"[RingDetector] parameters loaded from config "
                      f"(path={config.path}, exists={os.path.exists(config.path)})")
        except Exception as e:
            log_print(f"[RingDetector] failed to load params from config: {e}")

    def _get_tracking(self, color: Color) -> _TrackingState:
        if color not in self._tracking:
            self._tracking[color] = _TrackingState(smooth_window=self.smooth_window)
        ts = self._tracking[color]
        self._active_tracking = ts
        self._active_color = color
        return ts

    def set_detect_method(self, method: RingDetectMethod) -> bool:
        if method == self.detect_method:
            return True
        if method in (RingDetectMethod.EDGE_DRAWING_RING, RingDetectMethod.HEURISTIC_RING) and self.ed is None:
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

    def get_detect_method(self) -> RingDetectMethod:
        return self.detect_method

    @staticmethod
    def get_supported_methods():
        return [RingDetectMethod.FAST_RING, RingDetectMethod.EDGE_DRAWING_RING, RingDetectMethod.HEURISTIC_RING]

    def get_method_params(self, method: RingDetectMethod) -> dict:
        return {
            "roi_size": self.roi_size,
            "max_roi_miss": self.max_roi_miss,
            "min_area": self.min_area,
            "smooth_window": self.smooth_window,
            "q_base": self.q_base,
            "q_vel_base": self.q_vel_base,
            "max_lost_frames": self.max_lost_frames,
            "ring_gap_px": self.ring_gap_px,
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "ed_min_path_length": self.ed_min_path_length,
            "ed_gradient_threshold": self.ed_gradient_threshold,
            "ed_nfa_validation": self.ed_nfa_validation,
            "edge_morph_kernel": self.edge_morph_kernel,
            "edge_morph_iterations": self.edge_morph_iterations,
            "color_blob_min_area": self.color_blob_min_area,
        }

    def update_params(self, params: dict):
        int_keys = (
            "roi_size", "max_roi_miss", "min_area",
            "smooth_window", "blur_kernel", "ed_min_path_length",
            "ed_gradient_threshold", "edge_morph_kernel", "edge_morph_iterations",
            "max_lost_frames", "ring_gap_px", "color_blob_min_area",
        )
        for key in int_keys:
            if key in params:
                setattr(self, key, int(params[key]))

        float_keys = ("blur_sigma", "q_base", "q_vel_base")
        for key in float_keys:
            if key in params:
                setattr(self, key, float(params[key]))

        if "ed_nfa_validation" in params:
            self.ed_nfa_validation = bool(params["ed_nfa_validation"])

        for ts in self._tracking.values():
            ts.resize_histories(self.smooth_window)

        self._update_ed_params()

    def detect_ring(self, frame, target_color: Color) -> Optional[RingTarget]:
        method = self._methods.get(self.detect_method)
        if method is not None:
            return method.detect(frame, target_color)
        return None
