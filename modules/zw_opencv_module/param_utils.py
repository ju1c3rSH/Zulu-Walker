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

        # 强制使用代码中写死的 quad_aspect_ratio，忽略配置文件中的值
        if "edge_drawing_quads" in methods_params:
            methods_params["edge_drawing_quads"]["quad_aspect_ratio"] = 1.51

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
        "uv_v_min2": 236,
        "uv_v_max2": 255,
        "uv_min_area": 2,
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


# ── 摄像头硬件参数 ──────────────────────────────────────────────

# (显示名, key, CAP_PROP, min, max, default)
CAMERA_PARAM_DEFS = [
    ("Brightness",  "brightness",  cv2.CAP_PROP_BRIGHTNESS,        0,   255,  128),
    ("Contrast",    "contrast",    cv2.CAP_PROP_CONTRAST,          0,   255,  128),
    ("Saturation",  "saturation",  cv2.CAP_PROP_SATURATION,        0,   255,  128),
    ("Sharpness",   "sharpness",   cv2.CAP_PROP_SHARPNESS,         0,   255,  128),
    ("Gain",        "gain",        cv2.CAP_PROP_GAIN,              0,   255,    0),
    ("Exposure",    "exposure",    cv2.CAP_PROP_EXPOSURE,         -13,    -1,   -5),
    ("AutoExp",     "auto_exp",    cv2.CAP_PROP_AUTO_EXPOSURE,     0,     3,    1),
    ("WbAuto",      "wb_auto",     cv2.CAP_PROP_AUTO_WB,           0,     1,    1),
    ("WbTemp",      "wb_temp",     cv2.CAP_PROP_WB_TEMPERATURE, 2000, 10000, 4600),
    ("Gamma",       "gamma",       cv2.CAP_PROP_GAMMA,             0,   500,  100),
    ("Backlight",   "backlight",   cv2.CAP_PROP_BACKLIGHT,         0,     2,    1),
]


def get_camera_config_path() -> str:
    """获取摄像头硬件参数配置文件路径"""
    return os.path.join(os.path.dirname(__file__), "config", "camera_params.yaml")


def load_camera_params(config_path: str = None) -> Dict[str, Any]:
    """
    从 YAML 加载摄像头硬件参数

    Args:
        config_path: 配置文件路径，默认使用 camera_params.yaml

    Returns:
        摄像头参数字典
    """
    if config_path is None:
        config_path = get_camera_config_path()

    defaults = {key: default for _, key, _, _, _, default in CAMERA_PARAM_DEFS}

    if not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "camera_params" in data:
                defaults.update(data["camera_params"])
        return defaults
    except Exception as e:
        print(f"[param_utils] Failed to load camera config: {e}")
        return defaults


def apply_camera_params_to_capture(cap, params: Dict[str, Any]):
    """
    将摄像头硬件参数应用到 VideoCapture

    Args:
        cap: cv2.VideoCapture 实例
        params: 参数字典
    """
    for _, key, cap_prop, _, _, _ in CAMERA_PARAM_DEFS:
        if key in params:
            try:
                cap.set(cap_prop, int(params[key]))
            except Exception:
                pass
