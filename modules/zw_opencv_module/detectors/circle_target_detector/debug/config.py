import os
from typing import Dict, List, Optional

import yaml

from ..._shared.param_def import ParamDef


CIRCLE_METHOD_PARAM_DEFS: Dict[str, List[ParamDef]] = {
    "CONTOUR_ELLIPSE": [
        ParamDef("blur_kernel",                "Blur Kernel",           5,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",                 "Blur Sigma",           38,    5,  500,   5,  scale=0.01),
        ParamDef("min_contour_points",         "Min Contour Pts",      15,    5,   50,   5),
        ParamDef("min_area_threshold_ellipse", "Min Area Ellipse",    100,   10, 2000,  10),
        ParamDef("max_aspect_ratio",           "Max Aspect Ratio",    200,  100,  500,  10,  scale=0.01),
        ParamDef("min_circularity",            "Min Circularity",      40,   10,  100,   1,  scale=0.01),
    ],
    "EDGE_CONTOUR_ELLIPSE": [
        ParamDef("ed_min_path_length",         "ED MinPath",          164,   10,  300,  10),
        ParamDef("ed_gradient_threshold",      "ED Grad",              90,    5,  200,   1),
        ParamDef("morph_type",                 "Morph Type",            4,    0,    4,   1),
        ParamDef("morph_kernel",               "Morph Kernel",          3,    1,   31,   2,  odd=True),
        ParamDef("morph_iterations",           "Morph Iter",            1,    1,    5,   1),
        ParamDef("blur_kernel",                "Blur Kernel",           5,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",                 "Blur Sigma",           38,    5,  500,   5,  scale=0.01),
        ParamDef("min_area_threshold_quad",    "Min Area Quad",      1680,  100, 5000,  50),
        ParamDef("min_area_threshold_ellipse", "Min Area Ellipse",    100,   10, 2000,  10),
        ParamDef("min_contour_points",         "Min Contour Pts",      15,    5,   50,   5),
        ParamDef("max_aspect_ratio",           "Max Aspect Ratio",    200,  100,  500,  10,  scale=0.01),
        ParamDef("min_circularity",            "Min Circularity",      40,   10,  100,   1,  scale=0.01),
    ],
    "EDGE_DRAWING_QUADS": [
        ParamDef("ed_min_path_length",         "ED MinPath",          164,   10,  300,  10),
        ParamDef("ed_gradient_threshold",      "ED Grad",              90,    5,  200,   1),
        ParamDef("morph_type",                 "Morph Type",            4,    0,    4,   1),
        ParamDef("morph_kernel",               "Morph Kernel",          3,    1,   31,   2,  odd=True),
        ParamDef("morph_iterations",           "Morph Iter",            1,    1,    5,   1),
        ParamDef("blur_kernel",                "Blur Kernel",           5,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",                 "Blur Sigma",           38,    5,  500,   5,  scale=0.01),
        ParamDef("min_area_threshold_quad",    "Min Area Quad",      1680,  100, 5000,  50),
        ParamDef("quad_aspect_ratio",          "Quad Aspect Ratio",   151,  100,  300,  10,  scale=0.01),
        ParamDef("uv_min_area",                "UV Min Area",           0,    0, 1000,  10),
        ParamDef("enable_color_filter",        "Color Filter",          1,    0,    1,   1),
        ParamDef("uv_adaptive_enabled",        "UV Adaptive",           1,    0,    1,   1),
        ParamDef("uv_s_gate",                  "UV S Gate",            80,   20,  200,   5),
        ParamDef("uv_s_min",                   "UV S Min",             80,   10,  200,   5),
        ParamDef("uv_v_floor",                 "UV V Floor",           90,   30,  200,   5),
        ParamDef("uv_v_percentile",            "UV V Percentile",      95,   50,  100,   5),
        ParamDef("uv_contrast_ratio_min",      "UV Contrast Ratio",   115,  100,  300,   5,  scale=0.01),
        ParamDef("uv_contrast_dilate",         "UV Contrast Dilate",   30,    5,  100,   5),
    ],
    "TEST_LINE_QUAD": [
        ParamDef("ed_min_path_length",         "ED MinPath",          164,   10,  300,  10),
        ParamDef("ed_gradient_threshold",      "ED Grad",              90,    5,  200,   1),
        ParamDef("blur_kernel",                "Blur Kernel",           5,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",                 "Blur Sigma",           38,    5,  500,   5,  scale=0.01),
        ParamDef("min_area_threshold_quad",    "Min Area Quad",      1680,  100, 5000,  50),
        ParamDef("min_area_threshold_ellipse", "Min Area Ellipse",    100,   10, 2000,  10),
        ParamDef("min_contour_points",         "Min Contour Pts",      15,    5,   50,   5),
        ParamDef("min_circularity",            "Min Circularity",      40,   10,  100,   1,  scale=0.01),
        ParamDef("quad_aspect_ratio",          "Quad Aspect Ratio",   151,  100,  300,  10,  scale=0.01),
        ParamDef("uv_min_area",                "UV Min Area",           0,    0, 1000,  10),
        ParamDef("enable_color_filter",        "Color Filter",          1,    0,    1,   1),
        ParamDef("uv_adaptive_enabled",        "UV Adaptive",           1,    0,    1,   1),
        ParamDef("uv_s_gate",                  "UV S Gate",            80,   20,  200,   5),
        ParamDef("uv_s_min",                   "UV S Min",             80,   10,  200,   5),
        ParamDef("uv_v_floor",                 "UV V Floor",           90,   30,  200,   5),
        ParamDef("uv_v_percentile",            "UV V Percentile",      95,   50,  100,   5),
        ParamDef("uv_contrast_ratio_min",      "UV Contrast Ratio",   115,  100,  300,   5,  scale=0.01),
        ParamDef("uv_contrast_dilate",         "UV Contrast Dilate",   30,    5,  100,   5),
    ],
}

SHARED_PARAM_DEFS: List[ParamDef] = [
    ParamDef("max_lost_frames",    "Max Lost",         10,    1,   50,   1),
    ParamDef("_kf_q_base",        "KF Q Base",         30,    1,  100,   5,  scale=0.01),
    ParamDef("_kf_q_vel_base",    "KF Q Vel Base",     30,    1,  100,   5,  scale=0.01),
    ParamDef("uv_max_lost_frames","UV Max Lost",       10,    1,   50,   1),
    ParamDef("_uv_q_base",        "UV Q Base",         50,    1,  100,   5,  scale=0.01),
    ParamDef("_uv_q_vel_base",    "UV Q Vel Base",     50,    1,  100,   5,  scale=0.01),
]


class CircleTargetConfig:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "params.yaml")
        self.path = os.path.abspath(path)

    def load(self) -> Dict[str, Dict[str, int]]:
        if not os.path.exists(self.path):
            return self._all_defaults()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return self._all_defaults()
            result = self._all_defaults()
            for method_key in result:
                if method_key in data and isinstance(data[method_key], dict):
                    for param_key in result[method_key]:
                        if param_key in data[method_key] and isinstance(data[method_key][param_key], int):
                            result[method_key][param_key] = data[method_key][param_key]
            return result
        except Exception:
            return self._all_defaults()

    def save(self, params_by_method: Dict[str, Dict[str, int]]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(params_by_method, f, default_flow_style=False)

    def _all_defaults(self) -> Dict[str, Dict[str, int]]:
        result = {}
        for method_key, defs in CIRCLE_METHOD_PARAM_DEFS.items():
            result[method_key] = {p.name: p.default for p in defs}
        result["SHARED"] = {p.name: p.default for p in SHARED_PARAM_DEFS}
        return result

    def get_param_defs(self, method_key: str) -> List[ParamDef]:
        defs = list(CIRCLE_METHOD_PARAM_DEFS.get(method_key, []))
        defs.extend(SHARED_PARAM_DEFS)
        return defs
