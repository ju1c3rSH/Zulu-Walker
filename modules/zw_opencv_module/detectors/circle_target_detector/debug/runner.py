import time
from typing import Dict, Optional

import cv2
import numpy as np

from ....models.circle import ShapeType
from .. import CircleTargetDetector
from ..detection import DetectMethod
from .config import CircleTargetConfig, CIRCLE_METHOD_PARAM_DEFS, SHARED_PARAM_DEFS
from .window import CircleTargetDebugWindow


_METHOD_KEY_MAP = {
    DetectMethod.CONTOUR_ELLIPSE: "CONTOUR_ELLIPSE",
    DetectMethod.EDGE_CONTOUR_ELLIPSE: "EDGE_CONTOUR_ELLIPSE",
    DetectMethod.EDGE_DRAWING_QUADS: "EDGE_DRAWING_QUADS",
    DetectMethod.TEST_LINE_QUAD: "TEST_LINE_QUAD",
}


class CircleTargetDebugRunner:
    def __init__(self, camera_source: int = 0, width: int = 640, height: int = 480):
        self.camera_source = camera_source
        self.width = width
        self.height = height

        self.config = CircleTargetConfig()
        self.detector = CircleTargetDetector()
        self.window = CircleTargetDebugWindow(
            param_defs=self.config.get_param_defs("EDGE_DRAWING_QUADS"),
            on_change=self._on_param_changed,
            on_method_change=self._on_method_changed,
        )
        self.stream: Optional["CameraStream"] = None

        self._running = False
        self._save_pending = False
        self._last_save_time = 0.0

        self._current_method_key = "EDGE_DRAWING_QUADS"

        self._load_config()

    def _load_config(self):
        data = self.config.load()
        method_idx = data.get("_method_index", 2)
        method_keys = list(_METHOD_KEY_MAP.values())
        if 0 <= method_idx < len(method_keys):
            self._current_method_key = method_keys[method_idx]

        method_params = data.get(self._current_method_key, {})
        shared_params = data.get("SHARED", {})
        all_params = {**method_params, **shared_params}

        self.detector.update_params(all_params)

        for pdef in self._get_active_defs():
            if pdef.name in all_params:
                self.window.set_param(pdef.name, all_params[pdef.name])

        methods = self.detector.get_supported_methods()
        if 0 <= method_idx < len(methods):
            self.detector.set_detect_method(methods[method_idx])
            self.window.set_method_index(method_idx)

    def _on_param_changed(self, name: str, raw_value: int):
        for p in self._get_active_defs():
            if p.name == name:
                actual = raw_value * p.scale
                if p.scale == 1.0:
                    actual = int(actual)
                setattr(self.detector, name, actual)
                break
        self.detector._update_ed_params()
        self._save_pending = True

    def _on_method_changed(self, raw_value: int):
        methods = self.detector.get_supported_methods()
        if not (0 <= raw_value < len(methods)):
            return
        self.detector.set_detect_method(methods[raw_value])
        method_key = _METHOD_KEY_MAP[methods[raw_value]]
        self._switch_to_method(method_key)
        self._save_pending = True

    def _switch_to_method(self, method_key: str):
        self._current_method_key = method_key
        new_defs = self.config.get_param_defs(method_key)
        self.window.param_defs = new_defs

        data = self.config.load()
        method_params = data.get(method_key, {})
        shared_params = data.get("SHARED", {})
        all_params = {**method_params, **shared_params}

        if self.window._window_created:
            cv2.destroyWindow(self.window.title)
            self.window._window_created = False

        self.window._raw_params.clear()
        for p in new_defs:
            raw = all_params.get(p.name, p.default)
            self.window._raw_params[p.name] = raw

        self.detector._update_ed_params()
        self.window.setup()

    def _get_active_defs(self):
        method_defs = CIRCLE_METHOD_PARAM_DEFS.get(self._current_method_key, [])
        return list(method_defs) + list(SHARED_PARAM_DEFS)

    def run(self):
        from ....camera_stream import CameraStream

        self.stream = CameraStream(self.camera_source, self.width, self.height)
        self.window.setup()
        self._running = True

        while self._running:
            frame = self.stream.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self._process_frame(frame)
            intermediates = self._collect_intermediates()
            self.window.update(frame=frame, result=result, intermediates=intermediates)
            self.window.refresh()

            if self._save_pending and time.time() - self._last_save_time > 0.5:
                self._save_params()
                self._save_pending = False
                self._last_save_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        self._cleanup()

    def _collect_intermediates(self) -> Dict[int, np.ndarray]:
        steps = {}
        idx = 0
        if self.detector._last_canny_preview is not None:
            steps[idx] = self.detector._last_canny_preview
            idx += 1
        return steps

    def _save_params(self):
        data = self.config.load()
        raw = self.window.get_raw_params()
        shared_names = {s.name for s in SHARED_PARAM_DEFS}
        method_params = {k: v for k, v in raw.items() if k not in shared_names}
        shared_params = {k: v for k, v in raw.items() if k in shared_names}
        data[self._current_method_key] = method_params
        data["SHARED"] = shared_params
        data["_method_index"] = self.window.method_index
        self.config.save(data)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        targets = self.detector.detect_circle_targets(frame, None)
        for item in targets.targets:
            cx, cy = item.center_coordinates
            color_bgr = {
                "Red": (0, 0, 255),
                "Green": (0, 255, 0),
                "Blue": (255, 0, 0),
                "Black": (128, 128, 128),
            }.get(item.color, (0, 255, 255))
            cv2.circle(display, (int(cx), int(cy)), 6, color_bgr, 2)
            cv2.putText(
                display, f"{item.color} ({int(cx)},{int(cy)})",
                (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1,
            )
            if item.shape_type == ShapeType.QUAD and item.quad_points is not None:
                cv2.polylines(display, [item.quad_points.astype(np.int32)], True, color_bgr, 2)
            if self.detector.is_uv_spot_detected and self.detector.uv_spot_center:
                ux, uy = self.detector.uv_spot_center
                cv2.circle(display, (int(ux), int(uy)), 4, (255, 255, 0), -1)
                cv2.putText(
                    display, f"UV ({int(ux)},{int(uy)})",
                    (int(ux) + 10, int(uy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                )
        return display

    def _cleanup(self):
        self.window.close()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
