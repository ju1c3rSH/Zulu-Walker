# -*- coding: utf-8 -*-
"""Load, save, and apply detection and camera parameters shared by the app and debugger."""
import os
import sys
from typing import Dict, Any, Tuple, Optional
import yaml
import cv2

# Allow running this file directly as well as importing it as a module.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .detectors.circle_target_detector import DetectMethod, CircleTargetDetector
from utils.log_util import log_print


def get_default_params() -> Dict[str, Dict[str, Any]]:
    """Return default parameters for each detection method."""

    return {
        "edge_contour_ellipse": {
            "ed_min_path_length": 50,
            "ed_gradient_threshold": 20,
            "ed_nfa_validation": True,
            "morph_type": 1,
            "morph_kernel": 3,
            "morph_iterations": 1,
            "blur_kernel": 5,
            "blur_sigma": 1.0,
            "min_area_threshold": 150,
            "min_contour_points": 15,
        },
        "edge_drawing_quads": {
            "ed_min_path_length": 164,
            "ed_gradient_threshold": 90,
            "ed_nfa_validation": True,
            "morph_type": 4,
            "morph_kernel": 3,
            "morph_iterations": 1,
            "blur_kernel": 5,
            "blur_sigma": 38.0,
            "min_area_threshold_quad": 1680,
            "quad_aspect_ratio": 1.51,
            "uv_min_area": 0,
            "enable_color_filter": True,
        },
        "contour_ellipse": {
            "min_area_threshold": 150,
            "min_contour_points": 15,
        },
        "test_line_quad": {
            "blur_kernel": 5,
            "blur_sigma": 1.0,
            "min_area_threshold_quad": 150,
        },
    }


def load_detect_params(config_path: str) -> Tuple[DetectMethod, Dict[str, Dict[str, Any]]]:
    """Load detection parameters from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The active method and the parameters for every method.
    """
    default_params = get_default_params()

    if not os.path.exists(config_path):
        # Fall back to defaults with edge_drawing_quads active.
        return DetectMethod.EDGE_DRAWING_QUADS, default_params

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not data:
                return DetectMethod.EDGE_DRAWING_QUADS, default_params

        # Resolve the active method, defaulting to edge_drawing_quads.
        current_method = DetectMethod.EDGE_DRAWING_QUADS
        if "current_method" in data:
            try:
                current_method = DetectMethod(data["current_method"])
            except ValueError:
                pass

        # Merge stored parameters over the defaults.
        methods_params = default_params.copy()
        if "methods" in data:
            for method_name, params in data["methods"].items():
                if method_name in methods_params and params:
                    methods_params[method_name].update(params)

        return current_method, methods_params

    except Exception as e:
        log_print(f"[param_utils] Failed to load config: {e}")
        return DetectMethod.EDGE_DRAWING_QUADS, default_params


def apply_params_to_detector(
    detector: CircleTargetDetector,
    method: DetectMethod,
    params: Dict[str, Any]
):
    """Select the detection method on a detector and set its parameters."""
    detector.set_detect_method(method)
    detector.set_method_params(method, params)


def get_config_path() -> str:
    """Return the default debug parameter file path."""
    return os.path.join(
        os.path.dirname(__file__),
        "config",
        "debug_params.yaml"
    )


def get_uv_config_path() -> str:
    """Return the UV parameter file path."""
    return os.path.join(
        os.path.dirname(__file__),
        "config",
        "uv_params.yaml"
    )


def load_uv_params(config_path: str = None) -> Dict[str, Any]:
    """Load UV parameters from YAML, using defaults when values are absent.

    Args:
        config_path: Optional YAML path; defaults to the UV config file.

    Returns:
        UV parameters with stored values overriding the defaults.
    """
    if config_path is None:
        config_path = get_uv_config_path()

    default_uv_params = {
        "uv_h_min1": 130,
        "uv_h_max1": 145,
        "uv_s_min1": 90,
        "uv_s_max1": 255,
        "uv_v_min1": 190,
        "uv_v_max1": 255,
        "uv_h_min2": 130,
        "uv_h_max2": 155,
        "uv_s_min2": 0,
        "uv_s_max2": 50,
        "uv_v_min2": 235,
        "uv_v_max2": 255,
        "uv_min_area": 0,
        # Adaptive UV detection thresholds.
        "uv_adaptive_enabled": 1,
        "uv_v_percentile": 94,
        "uv_v_floor": 90,
        "uv_s_min": 80,
        "uv_h_low": 121,
        "uv_h_high": 165,
        "uv_s_gate": 80,
        "uv_contrast_ratio_min": 115,  # Stored as x100; actual value 1.15.
        "uv_contrast_dilate": 30,
    }

    if not os.path.exists(config_path):
        return default_uv_params

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "uv_params" in data:
                default_uv_params.update(data["uv_params"])
            return default_uv_params
    except Exception as e:
        log_print(f"[param_utils] Failed to load UV config: {e}")
        return default_uv_params


def apply_uv_params_to_detector(detector: CircleTargetDetector, uv_params: Dict[str, Any]):
    """Apply UV detection parameters to a detector."""
    import numpy as np

    # Set the two UV color ranges.
    uv_ranges = [
        # First range.
        (np.array([uv_params["uv_h_min1"], uv_params["uv_s_min1"], uv_params["uv_v_min1"]]),
         np.array([uv_params["uv_h_max1"], uv_params["uv_s_max1"], uv_params["uv_v_max1"]])),
        # Second range.
        (np.array([uv_params["uv_h_min2"], uv_params["uv_s_min2"], uv_params["uv_v_min2"]]),
         np.array([uv_params["uv_h_max2"], uv_params["uv_s_max2"], uv_params["uv_v_max2"]])),
    ]
    detector.color_ranges["UV"] = uv_ranges

    # Set the minimum UV region area.
    if "uv_min_area" in uv_params:
        detector.uv_min_area = int(uv_params["uv_min_area"])

    # Adaptive UV detection parameters.
    if "uv_adaptive_enabled" in uv_params:
        detector.uv_adaptive_enabled = bool(int(uv_params["uv_adaptive_enabled"]))
    if "uv_v_percentile" in uv_params:
        detector.uv_v_percentile = int(uv_params["uv_v_percentile"])
    if "uv_v_floor" in uv_params:
        detector.uv_v_floor = int(uv_params["uv_v_floor"])
    if "uv_s_min" in uv_params:
        detector.uv_s_min = int(uv_params["uv_s_min"])
    if "uv_h_low" in uv_params:
        detector.uv_h_range = (int(uv_params["uv_h_low"]), detector.uv_h_range[1])
    if "uv_h_high" in uv_params:
        detector.uv_h_range = (detector.uv_h_range[0], int(uv_params["uv_h_high"]))
    if "uv_s_gate" in uv_params:
        detector.uv_s_gate = int(uv_params["uv_s_gate"])
    if "uv_contrast_ratio_min" in uv_params:
        detector.uv_contrast_ratio_min = int(uv_params["uv_contrast_ratio_min"]) / 100.0
    if "uv_contrast_dilate" in uv_params:
        detector.uv_contrast_dilate = int(uv_params["uv_contrast_dilate"])


# Camera hardware parameters.

# (display name, key, CAP_PROP, min, max)
CAMERA_PARAM_DEFS = [
    ("Brightness",  "brightness",  cv2.CAP_PROP_BRIGHTNESS,        0,   255),
    ("Contrast",    "contrast",    cv2.CAP_PROP_CONTRAST,          0,   255),
    ("Saturation",  "saturation",  cv2.CAP_PROP_SATURATION,        0,   255),
    ("Sharpness",   "sharpness",   cv2.CAP_PROP_SHARPNESS,         0,   255),
    ("Gain",        "gain",        cv2.CAP_PROP_GAIN,              0,   255),
    ("Exposure",    "exposure",    cv2.CAP_PROP_EXPOSURE,         -13,    -1),
    ("AutoExp",     "auto_exp",    cv2.CAP_PROP_AUTO_EXPOSURE,     0,     3),
    ("WbAuto",      "wb_auto",     cv2.CAP_PROP_AUTO_WB,           0,     1),
    ("WbTemp",      "wb_temp",     cv2.CAP_PROP_WB_TEMPERATURE, 2000, 10000),
    ("Gamma",       "gamma",       cv2.CAP_PROP_GAMMA,             0,   500),
    ("Backlight",   "backlight",   cv2.CAP_PROP_BACKLIGHT,         0,     2),
]


def get_camera_config_path() -> str:
    """Return the camera hardware parameter file path."""
    return os.path.join(os.path.dirname(__file__), "config", "camera_params.yaml")


def load_camera_params(config_path: str = None) -> Tuple[Dict[str, Any], set]:
    """Load user-configured camera parameters from YAML.

    Args:
        config_path: Optional YAML path; defaults to camera_params.yaml.

    Returns:
        The parameters and the keys explicitly set in the YAML. Returns an
        empty dict and set when no file exists; no hardcoded defaults are used.
    """
    if config_path is None:
        config_path = get_camera_config_path()

    if not os.path.exists(config_path):
        return {}, set()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "camera_params" in data:
                params = data["camera_params"]
                return params, set(params.keys())
        return {}, set()
    except Exception as e:
        log_print(f"[param_utils] Failed to load camera config: {e}")
        return {}, set()


def read_camera_params_from_capture(cap) -> Dict[str, Any]:
    """Read current values from the capture; unsupported properties are skipped."""
    result = {}

    # Read auto_exp first because it governs how exposure behaves.
    auto_exp_def = next((d for d in CAMERA_PARAM_DEFS if d[1] == "auto_exp"), None)
    if auto_exp_def:
        _, key, cap_prop, min_val, max_val = auto_exp_def
        try:
            raw = cap.get(cap_prop)
            if raw is not None:
                result[key] = max(min_val, min(max_val, int(round(raw))))
        except Exception:
            pass

    for display_name, key, cap_prop, min_val, max_val in CAMERA_PARAM_DEFS:
        if key == "auto_exp":
            continue
        try:
            raw = cap.get(cap_prop)
            if raw is not None:
                result[key] = max(min_val, min(max_val, int(round(raw))))
        except Exception:
            pass

    log_print(f"[param_utils] Read camera HW params: {result}")
    return result


def save_camera_params(params: Dict[str, Any], config_path: str = None):
    """Save camera parameters to YAML."""
    if config_path is None:
        config_path = get_camera_config_path()
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        data = {"camera_params": params}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        log_print(f"[param_utils] Failed to save camera config: {e}")


def apply_camera_params_to_capture(cap, params: Dict[str, Any], user_keys: set = None):
    """Apply camera hardware parameters to a VideoCapture.

    Args:
        cap: Capture to configure.
        params: Parameter values keyed by name.
        user_keys: Keys the user explicitly set in YAML.
    """
    if user_keys is None:
        user_keys = set()

    # Set auto_exp first.
    if "auto_exp" in params:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, int(params["auto_exp"]))
        except Exception:
            pass

    # Switching to manual exposure without an explicit value keeps the current one.
    switching_to_manual = "auto_exp" in params and int(params["auto_exp"]) == 1
    exposure_explicit = "exposure" in user_keys
    if switching_to_manual and not exposure_explicit:
        try:
            current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            if current_exposure is not None:
                params = dict(params)
                params["exposure"] = int(current_exposure)
        except Exception:
            pass

    for _, key, cap_prop, _, _ in CAMERA_PARAM_DEFS:
        if key not in params or key == "auto_exp":
            continue
        try:
            cap.set(cap_prop, int(params[key]))
        except Exception:
            pass
