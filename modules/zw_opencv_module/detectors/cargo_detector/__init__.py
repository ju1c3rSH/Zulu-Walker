
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
from ...models.cargo import CargoItem
from .detection import DetectMethod

_METHOD_TO_CONFIG_KEY = {
    DetectMethod.FAST_CIRCLE: "FAST_CIRCLE",
    DetectMethod.EDGE_DRAWING_CIRCLE: "EDGE_DRAWING_CIRCLE",
    DetectMethod.HEURISTIC_EDGE: "HEURISTIC_EDGE",
}


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
        self.ed_nfa_validation = False
        self.edge_morph_kernel = 3
        self.edge_morph_iterations = 1
        self.color_match_threshold = 0.20
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.color_match_threshold_per_color = {Color.GREEN: 0.30}
        self.stage2_color_threshold_per_color = {Color.GREEN: 0.22}

        # FastCircle 方法内部可调参数
        # sv_percentile: 粗筛掩码内取 S/V 通道的第 N 百分位作为自适应下界
        #   值越低 → 下界越低 → 细筛更宽松（包容更多光照变化）
        #   值越高 → 下界越高 → 细筛更严格（精确但易漏）
        #   与 config.py ParamDef 和 params.yaml 保持同一值
        self.sv_percentile = 15
        self.ema_alpha = 0.3
        self.coarse_min_pixels = 50
        self.coarse_ratio_threshold = 0.30
        self.sv_min_samples = 10
        self.sv_fallback_s = 50
        self.sv_fallback_v = 50
        self.ellipse_min_contour_points = 5
        self.ellipse_max_axis_ratio = 1.5

        # EdgeDrawingCircle 方法内部可调参数
        self.edge_min_pixels = 20
        self.low_light_min_pixels = 50
        self.relaxed_s = 10
        self.relaxed_v = 35
        self.low_light_s_divider = 3
        self.low_light_v_divider = 3
        self.score_weight_color = 0.5
        self.score_weight_circularity = 0.3
        self.score_weight_area = 0.2

        # HeuristicEdge 方法内部可调参数
        self.stage2_color_threshold = 0.3
        self.stage2_min_area_ratio = 0.5

        # EdgeDrawing 初始化（若不可用则默认使用 FAST_CIRCLE）
        self.ed = None
        self._init_edge_drawing()
        self.detect_method = DetectMethod.HEURISTIC_EDGE

        self._kf_q_base = 0.2
        self._kf_q_vel_base = 0.15
        self.max_lost_frames = 10

        self.roi_size = 150
        self.max_roi_miss = 5
        self.min_circularity = 0.5
        self.min_area = 4000
        self.smooth_window = 5
        self.kernel_open = 5
        self.kernel_close = 7

        self._last_mask = None
        self._last_morphed = None
        self._last_edge_preview = None
        self._last_alt_img = None
        self.force_stage = 0
        self.color_blob_min_area = 2000

        self._last_cargo_meta: Dict[Color, dict] = {}

        # 按颜色隔离的跟踪状态
        self._tracking: Dict[Color, _TrackingState] = {}
        self._active_tracking: Optional[_TrackingState] = None
        self._active_color: Optional[Color] = None

        self.color_ranges = {
            Color.RED: [
                (np.array([0, 15, 50]), np.array([10, 255, 255])),
                (np.array([170, 15, 50]), np.array([180, 255, 255])),
            ],
            Color.GREEN: [(np.array([45, 20, 30]), np.array([75, 255, 255]))],
            Color.BLUE: [(np.array([100, 60, 40]), np.array([130, 255, 255]))],
        }

        self._methods: Dict[DetectMethod, Any] = {}
        self._init_detection_methods()

        # 从 config.py ParamDef + params.yaml 统一加载覆盖默认值
        # 确保 CargoDetector 与 Debug UI 使用同一套参数定义
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
        from .detection.methods.fast_circle import FastCircleDetectionWithColorMethod
        from .detection.methods.edge_drawing_circle import EdgeDrawingCircleMethod
        from .detection.methods.heuristic_edge_circle import HeuristicEdgeCircleMethod
        self._methods = {
            DetectMethod.FAST_CIRCLE: FastCircleDetectionWithColorMethod(detector=self),
            DetectMethod.EDGE_DRAWING_CIRCLE: EdgeDrawingCircleMethod(detector=self),
            DetectMethod.HEURISTIC_EDGE: HeuristicEdgeCircleMethod(detector=self),
        }

    def _load_params_from_config(self):
        """只加载当前 detect_method 对应的 config section + SHARED。

        避免不同方法间的参数串扰。
        """
        try:
            from .debug.config import CargoConfig
            import os
            config = CargoConfig()
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
            log_print(f"[CargoDetector] parameters loaded from config "
                      f"(path={config.path}, exists={os.path.exists(config.path)})")
        except Exception as e:
            log_print(f"[CargoDetector] failed to load params from config: {e}")

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
        if method in (DetectMethod.EDGE_DRAWING_CIRCLE, DetectMethod.HEURISTIC_EDGE) and self.ed is None:
            log_print("EdgeDrawing method is not available currently, please check your python lib installation.")
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
        return [DetectMethod.FAST_CIRCLE, DetectMethod.EDGE_DRAWING_CIRCLE, DetectMethod.HEURISTIC_EDGE]

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
            "stage2_color_threshold": self.stage2_color_threshold,
            "stage2_min_area_ratio": self.stage2_min_area_ratio,
        }

    def update_params(self, params: dict):
        int_keys = (
            "roi_size", "max_roi_miss", "min_area", "kernel_open", "kernel_close",
            "smooth_window", "blur_kernel", "ed_min_path_length",
            "ed_gradient_threshold", "edge_morph_kernel", "edge_morph_iterations",
            "sv_percentile", "coarse_min_pixels", "sv_min_samples",
            "sv_fallback_s", "sv_fallback_v", "ellipse_min_contour_points",
            "edge_min_pixels", "low_light_min_pixels", "relaxed_s", "relaxed_v",
            "low_light_s_divider", "low_light_v_divider",
        )
        for key in int_keys:
            if key in params:
                setattr(self, key, int(params[key]))

        float_keys = (
            "min_circularity", "blur_sigma", "color_match_threshold",
            "ema_alpha", "coarse_ratio_threshold", "ellipse_max_axis_ratio",
            "score_weight_color", "score_weight_circularity", "score_weight_area",
            "stage2_color_threshold", "stage2_min_area_ratio",
        )
        for key in float_keys:
            if key in params:
                setattr(self, key, float(params[key]))

        for ts in self._tracking.values():
            ts.resize_histories(self.smooth_window)

        self._update_ed_params()

    def detect_cargo(self, frame, target_color: Color) -> Optional[CargoItem]:
        method = self._methods.get(self.detect_method)
        if method is not None:
            return method.detect(frame, target_color)
        return None
