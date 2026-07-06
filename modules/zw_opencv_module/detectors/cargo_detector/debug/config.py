import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class ParamDef:
    name: str
    display: str
    default: int
    min_val: int
    max_val: int
    step: int = 1
    scale: float = 1.0
    odd: bool = False


CARGO_PARAM_DEFS: List[ParamDef] = [
    ParamDef("roi_size",       "ROI Size",       150,   50,  500,  10),
    ParamDef("max_roi_miss",   "Max ROI Miss",     5,    1,   30,   1),
    ParamDef("min_circularity","Min Circ",         75,   10,  100,  1,  scale=0.01),
    ParamDef("min_area",       "Min Area",        100,   10, 2000,  10),
    ParamDef("kernel_open",    "Kernel Open",       5,    1,   31,   2,  odd=True),
    ParamDef("kernel_close",   "Kernel Close",      7,    1,   31,   2,  odd=True),
    ParamDef("smooth_window",  "Smooth Win",        5,    1,   20,   1),
    # EdgeDrawing 圆检测参数
    ParamDef("blur_kernel",    "Blur Kernel",       5,    1,   31,   2,  odd=True),
    ParamDef("blur_sigma",     "Blur Sigma",      150,    5,  500,   5,  scale=0.01),
    ParamDef("ed_min_path_length", "ED MinPath",   50,   10,  300,  10),
    ParamDef("ed_gradient_threshold", "ED Grad",    36,    5,  200,   1),
    ParamDef("edge_morph_kernel", "ED Close K",      3,    1,   15,   2,  odd=True),
    ParamDef("edge_morph_iterations", "ED Close Iter", 1, 1,    5,   1),
    ParamDef("color_match_threshold", "Color Match", 60, 10,  100,   1,  scale=0.01),
]


class CargoConfig:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "config",
                "cargo_debug_params.yaml"
            )
        self.path = os.path.abspath(path)

    def load(self) -> Dict[str, int]:
        if not os.path.exists(self.path):
            return self._defaults()

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return self._defaults()
            result = self._defaults()
            for key in result:
                if key in data and isinstance(data[key], int):
                    result[key] = data[key]
            return result
        except Exception:
            return self._defaults()

    def save(self, params: Dict[str, int]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(params, f, default_flow_style=False)

    def _defaults(self) -> Dict[str, int]:
        return {p.name: p.default for p in CARGO_PARAM_DEFS}
