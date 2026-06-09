# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import cv2


@dataclass
class ParamDef:
    name: str
    display_name: str
    default: int
    min_val: int
    max_val: int
    step: int = 1
    scale: float = 1.0
    is_odd: bool = False


METHOD_PARAMS: Dict[str, List[ParamDef]] = {
    "edge_contour_ellipse": [
        ParamDef("ed_min_path_length", "ED MinPath", 155, 10, 200, 5),
        ParamDef("ed_gradient_threshold", "ED Gradient", 80, 5, 100, 1),
        ParamDef("ed_nfa_validation", "ED NFA", 1, 0, 1, 1),
        ParamDef("morph_type", "Morph Type", 1, 0, 4, 1),
        ParamDef("morph_kernel", "Morph Kernel", 3, 1, 15, 2, is_odd=True),
        ParamDef("morph_iterations", "Morph Iter", 1, 1, 10, 1),
        ParamDef("blur_kernel", "Blur Kernel", 5, 1, 15, 2, is_odd=True),
        ParamDef("blur_sigma", "Blur Sigma", 10, 1, 50, 1, scale=0.1),
        ParamDef("min_area_threshold_quad", "Min Quad Area", 150, 10, 2000, 10),
        ParamDef("min_area_threshold_ellipse", "Min Ellipse Area", 100, 10, 2000, 10),
        ParamDef("min_contour_points", "Min Points", 15, 5, 50, 1),
        ParamDef("max_aspect_ratio", "Max Aspect", 20, 10, 50, 1, scale=0.1),
        ParamDef("min_circularity", "Min Circ", 4, 1, 10, 1, scale=0.1),
    ],
    "edge_drawing_quads": [
        ParamDef("ed_min_path_length", "ED MinPath", 164, 10, 200, 5),
        ParamDef("ed_gradient_threshold", "ED Gradient", 90, 5, 100, 1),
        ParamDef("ed_nfa_validation", "ED NFA", 1, 0, 1, 1),
        ParamDef("morph_type", "Morph Type", 4, 0, 4, 1),
        ParamDef("morph_kernel", "Morph Kernel", 3, 1, 15, 2, is_odd=True),
        ParamDef("morph_iterations", "Morph Iter", 1, 1, 10, 1),
        ParamDef("blur_kernel", "Blur Kernel", 5, 1, 15, 2, is_odd=True),
        ParamDef("blur_sigma", "Blur Sigma", 38, 1, 50, 1, scale=0.1),
        ParamDef("min_area_threshold_quad", "Min Quad Area", 1680, 10, 2000, 10),
        ParamDef("uv_min_area", "UV Min Area", 0, 1, 100, 1),
        ParamDef("enable_color_filter", "Color Filter", 1, 0, 1, 1),
    ],
    "test_line_quad": [
        ParamDef("blur_kernel", "Blur Kernel", 5, 1, 15, 2, is_odd=True),
        ParamDef("blur_sigma", "Blur Sigma", 10, 1, 50, 1, scale=0.1),
        ParamDef("min_area_threshold_quad", "Min Quad Area", 150, 10, 2000, 10),
    ],
}


class ParamPanel:
    def __init__(
        self,
        method_name: str,
        params_def: List[ParamDef],
        window_name: str,
        on_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.method_name = method_name
        self.params_def = params_def
        self.window_name = window_name
        self.on_change = on_change

        self.params: Dict[str, int] = {}
        for pdef in params_def:
            self.params[pdef.name] = pdef.default

        self._trackbars_created = False

    def load_params(self, params: Dict[str, int]):
        for name, value in params.items():
            if name in self.params:
                self.params[name] = int(value)

    def create_trackbars(self):
        if self._trackbars_created:
            return

        for pdef in self.params_def:
            cv2.createTrackbar(
                pdef.display_name,
                self.window_name,
                int(self.params[pdef.name]),
                int(pdef.max_val),
                lambda val, d=pdef: self._on_trackbar(d, val)
            )

        self._trackbars_created = True

    def _on_trackbar(self, pdef: ParamDef, value: int):
        if pdef.is_odd and value % 2 == 0:
            value = max(pdef.min_val, value - 1)
            cv2.setTrackbarPos(pdef.display_name, self.window_name, int(value))

        if self.params.get(pdef.name) != value:
            self.params[pdef.name] = value
            if self.on_change:
                self.on_change(self.method_name, self.get_params())

    def get_params(self) -> Dict[str, Any]:
        result = {}
        for pdef in self.params_def:
            value = self.params[pdef.name]
            if pdef.scale != 1.0:
                result[pdef.name] = value * pdef.scale
            else:
                result[pdef.name] = value
        return result

    def get_raw_params(self) -> Dict[str, int]:
        return self.params.copy()

    def set_param(self, name: str, value: int):
        if name in self.params:
            self.params[name] = int(value)
            for pdef in self.params_def:
                if pdef.name == name:
                    cv2.setTrackbarPos(pdef.display_name, self.window_name, int(value))
                    break

    def get_param_def(self, name: str) -> Optional[ParamDef]:
        for pdef in self.params_def:
            if pdef.name == name:
                return pdef
        return None
