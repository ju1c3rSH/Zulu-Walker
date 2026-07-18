from typing import Dict, Optional

import cv2
import numpy as np

from ..._shared.base_debug_window import BaseDebugWindow


class CargoDebugWindow(BaseDebugWindow):
    _STAGE_NAMES = ["Auto", "ROI", "Global", "Blob", "Fallback"]
    _METHOD_NAMES = ["FAST_CIRCLE", "EDGE_DRAWING_CIRCLE", "HEURISTIC_EDGE"]
    _PREVIEW_LABELS = {
        0: "Original",
        1: "Edge Preview",
        2: "Color Mask",
        3: "Morphed Mask",
        4: "Alt Processed",
    }

    def __init__(self, **kwargs):
        super().__init__(title="Cargo Debug", method_count=3, **kwargs)
        self._show_circle = False
        self._cargo_data: Dict = {}
        self._stage_value = 0

    def setup(self):
        super().setup()
        cv2.createTrackbar(
            "Stage", self.title, 0, 4,
            lambda v: self._set_stage(v),
        )
        cv2.createTrackbar(
            "Circle", self.title, 0, 1,
            lambda v: self._set_circle(v),
        )

    def _set_stage(self, value: int):
        self._stage_value = value
        if self.on_change:
            self.on_change("force_stage", value)

    def _set_circle(self, value: int):
        self._show_circle = bool(value)

    def update(self, frame=None, result=None,
               intermediates=None, cargo_data=None):
        super().update(frame, result, intermediates)
        if cargo_data is not None:
            self._cargo_data = cargo_data

    def _build_preview(self) -> Optional[np.ndarray]:
        if self._show_circle and self._frame is not None:
            return self._build_circle_preview()
        preview = super()._build_preview()
        if preview is not None:
            self._draw_preview_label(preview)
        return preview

    def _draw_preview_label(self, frame: np.ndarray):
        text = self._PREVIEW_LABELS.get(self.preview_index,
                                        f"Step {self.preview_index}")
        cv2.putText(frame, text, (8, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _build_circle_preview(self) -> np.ndarray:
        preview = self._frame.copy()
        for meta in self._cargo_data.values():
            area = meta.get('area', 0) or 0
            center = meta.get('center')
            outer_r = meta.get('outer_radius', 0) or 0
            if area <= 0 or center is None or outer_r <= 0:
                continue
            cx, cy = int(center[0]), int(center[1])
            r = max(int(outer_r), 1)
            overlay = preview.copy()
            cv2.circle(overlay, (cx, cy), r, (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.3, preview, 0.7, 0, dst=preview)
            cv2.circle(preview, (cx, cy), r, (0, 200, 200), 2)
            cv2.putText(preview, f"Area: {area:.0f}",
                        (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        self._draw_preview_label(preview)
        return preview

    def _draw_params_info(self, frame: np.ndarray):
        super()._draw_params_info(frame)

        midx = self.method_index
        if 0 <= midx < len(self._METHOD_NAMES):
            text = f"Method: {self._METHOD_NAMES[midx]}"
            (tw, _), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, text,
                        (frame.shape[1] - tw - 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

        sv = self.preview_index if not self._show_circle else -1
        if sv >= 0:
            label = self._PREVIEW_LABELS.get(sv, f"Step {sv}")
        else:
            label = "Circle Overlay"
        cv2.putText(frame, label,
                    (frame.shape[1] - 160, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 0), 1)

        sv_stage = self._stage_value
        if 0 <= sv_stage < len(self._STAGE_NAMES):
            stage_text = f"Stage: {self._STAGE_NAMES[sv_stage]}"
            cv2.putText(frame, stage_text,
                        (frame.shape[1] - 160, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 255, 100), 1)
