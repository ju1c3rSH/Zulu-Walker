import os
from typing import Dict, List, Optional

import yaml

from ..._shared.param_def import ParamDef


RING_METHOD_PARAM_DEFS: Dict[str, List[ParamDef]] = {
    "FAST_RING": [
        ParamDef("roi_size",              "ROI Size",         150,   50,  500,  10),
        ParamDef("max_roi_miss",          "Max ROI Miss",       5,    1,   30,   1),
        ParamDef("min_area",              "Min Area",         150,   10, 2000,  10),
        ParamDef("smooth_window",         "Smooth Win",         5,    1,   20,   1),
    ],
    "EDGE_DRAWING_RING": [
        ParamDef("blur_kernel",           "Blur Kernel",        3,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",            "Blur Sigma",       150,    5,  500,   5,  scale=0.01),
        ParamDef("ed_min_path_length",    "ED MinPath",        50,   10,  300,  10),
        ParamDef("ed_gradient_threshold", "ED Grad",           36,    5,  200,   1),
        ParamDef("edge_morph_kernel",     "ED Close K",         3,    1,   15,   2,  odd=True),
        ParamDef("edge_morph_iterations", "ED Close Iter",      1,    1,    5,   1),
        ParamDef("min_area",              "Min Area",         150,   10, 2000,  10),
    ],
    "HEURISTIC_RING": [
        ParamDef("roi_size",              "ROI Size",         150,   50,  500,  10),
        ParamDef("max_roi_miss",          "Max ROI Miss",       5,    1,   30,   1),
        ParamDef("min_area",              "Min Area",         150,   10, 2000,  10),
        ParamDef("smooth_window",         "Smooth Win",         5,    1,   20,   1),
        ParamDef("blur_kernel",           "Blur Kernel",        3,    1,   31,   2,  odd=True),
        ParamDef("blur_sigma",            "Blur Sigma",       150,    5,  500,   5,  scale=0.01),
    ],
}

SHARED_PARAM_DEFS: List[ParamDef] = [
    ParamDef("max_lost_frames",  "Max Lost",         10,    1,   50,   1),
    ParamDef("q_base",           "Q Base",           20,    1,  100,   5,  scale=0.01),
    ParamDef("q_vel_base",       "Q Vel Base",       15,    1,  100,   5,  scale=0.01),
]


class RingConfig:
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
        for method_key, defs in RING_METHOD_PARAM_DEFS.items():
            result[method_key] = {p.name: p.default for p in defs}
        result["SHARED"] = {p.name: p.default for p in SHARED_PARAM_DEFS}
        return result

    def get_param_defs(self, method_key: str) -> List[ParamDef]:
        defs = list(RING_METHOD_PARAM_DEFS.get(method_key, []))
        defs.extend(SHARED_PARAM_DEFS)
        return defs
