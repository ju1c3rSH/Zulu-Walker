# -*- coding: utf-8 -*-
"""
检测参数工具模块

提供参数加载、保存和应用功能，供主程序和调试器共用。
"""
import os
import sys
from typing import Dict, Any, Tuple, Optional
import yaml
import cv2

# 支持直接运行和模块运行
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .circle_target_detector import DetectMethod, CircleTargetDetector


def get_default_params() -> Dict[str, Dict[str, Any]]:
    """获取各检测方法的默认参数"""
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
            "uv_min_area": 5,
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
    """
    从 YAML 加载检测参数

    Args:
        config_path: 配置文件路径

    Returns:
        Tuple[DetectMethod, Dict]: 当前方法和所有方法的参数
    """
    default_params = get_default_params()

    if not os.path.exists(config_path):
        # 返回默认值 - 默认使用 edge_drawing_quads
        return DetectMethod.EDGE_DRAWING_QUADS, default_params

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not data:
                return DetectMethod.EDGE_DRAWING_QUADS, default_params

        # 解析当前方法 - 默认使用 edge_drawing_quads
        current_method = DetectMethod.EDGE_DRAWING_QUADS
        if "current_method" in data:
            try:
                current_method = DetectMethod(data["current_method"])
            except ValueError:
                pass

        # 解析各方法参数
        methods_params = default_params.copy()
        if "methods" in data:
            for method_name, params in data["methods"].items():
                if method_name in methods_params and params:
                    methods_params[method_name].update(params)

        return current_method, methods_params

    except Exception as e:
        print(f"[param_utils] Failed to load config: {e}")
        return DetectMethod.EDGE_DRAWING_QUADS, default_params


def apply_params_to_detector(
    detector: CircleTargetDetector,
    method: DetectMethod,
    params: Dict[str, Any]
):
    """
    将参数应用到检测器

    Args:
        detector: 检测器实例
        method: 检测方法
        params: 参数字典
    """
    detector.set_detect_method(method)
    detector.set_method_params(method, params)


def get_config_path() -> str:
    """获取默认配置文件路径"""
    return os.path.join(
        os.path.dirname(__file__),
        "config",
        "debug_params.yaml"
    )


def get_uv_config_path() -> str:
    """获取 UV 参数配置文件路径"""
    return os.path.join(
        os.path.dirname(__file__),
        "config",
        "uv_params.yaml"
    )


def load_uv_params(config_path: str = None) -> Dict[str, Any]:
    """
    从 YAML 加载 UV 参数

    Args:
        config_path: 配置文件路径

    Returns:
        UV 参数字典
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
        "uv_min_area": 2,
        # 自适应 UV 检测
        "uv_adaptive_enabled": 0,
        "uv_v_percentile": 94,
        "uv_v_floor": 12,
        "uv_s_min": 20,
        "uv_h_low": 121,
        "uv_h_high": 165,
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
        print(f"[param_utils] Failed to load UV config: {e}")
        return default_uv_params


def apply_uv_params_to_detector(detector: CircleTargetDetector, uv_params: Dict[str, Any]):
    """
    将 UV 参数应用到检测器

    Args:
        detector: 检测器实例
        uv_params: UV 参数字典
    """
    import numpy as np

    # 更新 UV 颜色范围
    uv_ranges = [
        # 第一段范围
        (np.array([uv_params["uv_h_min1"], uv_params["uv_s_min1"], uv_params["uv_v_min1"]]),
         np.array([uv_params["uv_h_max1"], uv_params["uv_s_max1"], uv_params["uv_v_max1"]])),
        # 第二段范围
        (np.array([uv_params["uv_h_min2"], uv_params["uv_s_min2"], uv_params["uv_v_min2"]]),
         np.array([uv_params["uv_h_max2"], uv_params["uv_s_max2"], uv_params["uv_v_max2"]])),
    ]
    detector.color_ranges["UV"] = uv_ranges

    # 更新 UV 最小面积
    if "uv_min_area" in uv_params:
        detector.uv_min_area = int(uv_params["uv_min_area"])

    # 自适应 UV 检测参数
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


# ── 摄像头硬件参数 ──────────────────────────────────────────────

# (显示名, key, CAP_PROP, min, max)
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
    """获取摄像头硬件参数配置文件路径"""
    return os.path.join(os.path.dirname(__file__), "config", "camera_params.yaml")


def load_camera_params(config_path: str = None) -> Tuple[Dict[str, Any], set]:
    """
    从 YAML 加载用户配置的摄像头硬件参数

    Args:
        config_path: 配置文件路径，默认使用 camera_params.yaml

    Returns:
        (摄像头参数字典, 用户在 YAML 中显式指定的 key 集合)
        无 YAML 时返回空字典 + 空集合，不使用硬编码默认值
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
        print(f"[param_utils] Failed to load camera config: {e}")
        return {}, set()


def read_camera_params_from_capture(cap) -> Dict[str, Any]:
    """从摄像头硬件读取当前参数值，不支持的参数会被跳过"""
    result = {}

    # 先读 auto_exp（曝光模式），因为它影响 exposure 的行为
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

    print(f"[param_utils] Read camera HW params: {result}")
    return result


def save_camera_params(params: Dict[str, Any], config_path: str = None):
    """保存摄像头参数到 YAML"""
    if config_path is None:
        config_path = get_camera_config_path()
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        data = {"camera_params": params}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f"[param_utils] Failed to save camera config: {e}")


def apply_camera_params_to_capture(cap, params: Dict[str, Any], user_keys: set = None):
    """
    将摄像头硬件参数应用到 VideoCapture

    Args:
        cap: cv2.VideoCapture 实例
        params: 参数字典
        user_keys: 用户在 YAML 中显式指定的 key 集合
    """
    if user_keys is None:
        user_keys = set()

    # 先设 auto_exp（曝光模式）
    if "auto_exp" in params:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, int(params["auto_exp"]))
        except Exception:
            pass

    # 切手动模式时，若未显式指定 exposure，沿用自动模式下的当前值
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
