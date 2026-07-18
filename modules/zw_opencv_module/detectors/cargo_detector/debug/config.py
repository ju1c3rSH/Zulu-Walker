import os
from typing import Dict, List, Optional

import yaml

from ..._shared.param_def import ParamDef


CARGO_METHOD_PARAM_DEFS: Dict[str, List[ParamDef]] = {
    "FAST_CIRCLE": [
        ParamDef("roi_size",                "ROI Size",         150,   50,  500,  10),
        ParamDef("max_roi_miss",            "Max ROI Miss",       5,    1,   30,   1),
        ParamDef("min_circularity",         "Min Circ",          50,   10,  100,   1,  scale=0.01),
        ParamDef("min_area",                "Min Area",         100,   10, 2000,  10),
        ParamDef("kernel_open",             "Kernel Open",        5,    1,   31,   2,  odd=True),
        ParamDef("kernel_close",            "Kernel Close",       7,    1,   31,   2,  odd=True),
        ParamDef("smooth_window",           "Smooth Win",         5,    1,   20,   1),
        ParamDef("sv_percentile",           "SV Percentile",     15,    1,   50,   1),
        ParamDef("ema_alpha",               "EMA Alpha",         30,    5,  100,   5,  scale=0.01),
        ParamDef("coarse_min_pixels",       "Coarse MinPx",      50,   10,  500,  10),
        ParamDef("coarse_ratio_threshold",  "Coarse Ratio",      30,   10,   80,   5,  scale=0.01),
        ParamDef("sv_min_samples",          "SV Min Samples",    10,    5,  100,   5),
        ParamDef("sv_fallback_s",           "SV Fallback S",     50,   10,  200,  10),
        ParamDef("sv_fallback_v",           "SV Fallback V",     50,   10,  200,  10),
        ParamDef("ellipse_min_contour_points", "Ellipse MinPts",  5,    5,   50,   5),
        ParamDef("ellipse_max_axis_ratio",  "Ellipse AxisR",    150,  100,  300,  10,  scale=0.01),
    ],
    "EDGE_DRAWING_CIRCLE": [
        ParamDef("blur_kernel",             "Blur Kernel",        5,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",              "Blur Sigma",       150,    5,  500,   5,  scale=0.01),
        ParamDef("ed_min_path_length",      "ED MinPath",        50,   10,  300,  10),
        ParamDef("ed_gradient_threshold",   "ED Grad",           36,    5,  200,   1),
        ParamDef("edge_morph_kernel",       "ED Close K",         3,    1,   15,   2,  odd=True),
        ParamDef("edge_morph_iterations",   "ED Close Iter",      1,    1,    5,   1),
        ParamDef("color_match_threshold",   "Color Match",       20,   10,  100,   1,  scale=0.01),
        ParamDef("min_area",                "Min Area",        4000,   10, 10000,  10),
        ParamDef("min_circularity",         "Min Circ",          50,   10,  100,   1,  scale=0.01),
        ParamDef("edge_min_pixels",         "Edge MinPx",        20,    5,  200,   5),
        ParamDef("ellipse_min_contour_points", "Ellipse MinPts",  5,    5,   50,   5),
        ParamDef("ellipse_max_axis_ratio",  "Ellipse AxisR",    150,  100,  300,  10,  scale=0.01),
        ParamDef("low_light_min_pixels",    "LowLight MinPx",    50,   10,  500,  10),
        ParamDef("relaxed_s",               "Relaxed S",         10,    5,   50,   5),
        ParamDef("relaxed_v",               "Relaxed V",         35,    2,   50,   1),
        ParamDef("low_light_s_divider",     "LowLight S Div",     3,    1,   10,   1),
        ParamDef("low_light_v_divider",     "LowLight V Div",     3,    1,   10,   1),
        ParamDef("score_weight_color",      "Score Color",       50,   10,  100,   5,  scale=0.01),
        ParamDef("score_weight_circularity","Score Circ",        30,   10,  100,   5,  scale=0.01),
        ParamDef("score_weight_area",       "Score Area",        20,   10,  100,   5,  scale=0.01),
        ParamDef("color_blob_min_area",     "Blob Min Area",   2000,  100,10000, 100),
    ],
}

SHARED_PARAM_DEFS: List[ParamDef] = [
    ParamDef("max_lost_frames",             "Max Lost",          10,    1,   50,   1),
    ParamDef("_kf_q_base",                 "KF Q Base",         20,    1,  100,   5,  scale=0.01),
    ParamDef("_kf_q_vel_base",             "KF Q Vel Base",     15,    1,  100,   5,  scale=0.01),
]


class CargoConfig:
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
            if "_method_index" in data:
                result["_method_index"] = data["_method_index"]
            return result
        except Exception:
            return self._all_defaults()

    def save(self, params_by_method: Dict[str, Dict[str, int]]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(params_by_method, f, default_flow_style=False)

    def _all_defaults(self) -> Dict[str, Dict[str, int]]:
        result = {}
        for method_key, defs in CARGO_METHOD_PARAM_DEFS.items():
            result[method_key] = {p.name: p.default for p in defs}
        result["SHARED"] = {p.name: p.default for p in SHARED_PARAM_DEFS}
        return result

    def get_param_defs(self, method_key: str) -> List[ParamDef]:
        defs = list(CARGO_METHOD_PARAM_DEFS.get(method_key, []))
        defs.extend(SHARED_PARAM_DEFS)
        return defs
