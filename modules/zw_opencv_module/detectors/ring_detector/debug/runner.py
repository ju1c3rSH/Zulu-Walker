import time
from typing import Dict, Optional

import cv2
import numpy as np

from ....models.color import Color
from .. import RingDetector
from ..detection import RingDetectMethod
from .config import RingConfig, RING_METHOD_PARAM_DEFS, SHARED_PARAM_DEFS
from .window import RingDebugWindow


_METHOD_KEY_MAP = {
    RingDetectMethod.FAST_RING: "FAST_RING",
    RingDetectMethod.EDGE_DRAWING_RING: "EDGE_DRAWING_RING",
    RingDetectMethod.HEURISTIC_RING: "HEURISTIC_RING",
}

_COLORS = [Color.RED, Color.GREEN, Color.BLUE]


class RingDebugRunner:
    def __init__(self, camera_source: int | str = 0, width: int = 640, height: int = 480):
        self.camera_source = camera_source
        self.width = width
        self.height = height

        self.config = RingConfig()
        self.detector = RingDetector()
        self.window = RingDebugWindow(
            param_defs=self.config.get_param_defs("HEURISTIC_RING"),
            on_change=self._on_param_changed,
            on_method_change=self._on_method_changed,
        )
        self.stream: Optional["CameraStream"] = None

        self._running = False
        self._save_pending = False
        self._last_save_time = 0.0

        self._current_method_key = "HEURISTIC_RING"

        self._load_config()

    def _load_config(self):
        data = self.config.load()
        method_idx = data.get("_method_index", 2)
        method_keys = list(_METHOD_KEY_MAP.values())
        if 0 <= method_idx < len(method_keys):
            new_method_key = method_keys[method_idx]
        else:
            new_method_key = self._current_method_key

        if new_method_key != self._current_method_key:
            self._switch_to_method(new_method_key)
        else:
            method_params = data.get(self._current_method_key, {})
            shared_params = data.get("SHARED", {})
            all_params = {**method_params, **shared_params}
            self._apply_params(all_params)
            for pdef in self._get_active_defs():
                if pdef.name in all_params:
                    self.window.set_param(pdef.name, all_params[pdef.name])

        methods = self.detector.get_supported_methods()
        if 0 <= method_idx < len(methods):
            ok = self.detector.set_detect_method(methods[method_idx])
            if ok:
                self.window.set_method_index(method_idx)
            else:
                self.window.set_method_index(0)

    def _apply_params(self, all_params: dict):
        """Apply raw YAML params to detector with proper scale factors."""
        for pdef in self._get_active_defs():
            raw = all_params.get(pdef.name)
            if raw is None:
                continue
            actual = raw * pdef.scale
            if pdef.scale == 1.0:
                actual = int(actual)
            setattr(self.detector, pdef.name, actual)
        self.detector._update_ed_params()

    def _on_param_changed(self, name: str, raw_value: int):
        if name == "force_stage":
            self.detector.force_stage = raw_value
            return
        for p in self._get_active_defs():
            if p.name == name:
                actual = raw_value * p.scale
                if p.scale == 1.0:
                    actual = int(actual)
                setattr(self.detector, name, actual)
                break
        self.detector._update_ed_params()
        if name == "smooth_window":
            for ts in self.detector._tracking.values():
                ts.resize_histories(self.detector.smooth_window)
        self._save_pending = True

    def _on_method_changed(self, raw_value: int):
        methods = self.detector.get_supported_methods()
        if not (0 <= raw_value < len(methods)):
            return
        ok = self.detector.set_detect_method(methods[raw_value])
        if not ok:
            self.window.set_method_index(0)
            return
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

        self._apply_params(all_params)
        self.window.setup()
        self.detector.force_stage = 0

    def _get_active_defs(self):
        method_defs = RING_METHOD_PARAM_DEFS.get(self._current_method_key, [])
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
            self.window.update(frame=frame, result=result,
                               intermediates=intermediates,
                               ring_data=self.detector._last_ring_meta.copy())
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
        if self.detector._last_edge_preview is not None:
            steps[idx] = self.detector._last_edge_preview
            idx += 1
        if self.detector._last_mask is not None:
            steps[idx] = self.detector._last_mask
            idx += 1
        if self.detector._last_morphed is not None:
            steps[idx] = self.detector._last_morphed
            idx += 1
        if self.detector._last_alt_img is not None:
            steps[idx] = self.detector._last_alt_img
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
        for color in _COLORS:
            item = self.detector.detect_ring(frame, color)
            if item is not None and not item.is_predicted:
                cx, cy = item.coordinate
                color_bgr = {
                    Color.RED: (0, 0, 255),
                    Color.GREEN: (0, 255, 0),
                    Color.BLUE: (255, 0, 0),
                }[color]
                cv2.circle(display, (int(cx), int(cy)), 6, color_bgr, 2)
                cv2.putText(
                    display, f"{color.name} ({int(cx)},{int(cy)})",
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1,
                )
                self._draw_ring_meta_overlay(display, color, (cx, cy), color_bgr)
        return display

    def _draw_ring_meta_overlay(self, display, color, coordinate, color_bgr):
        meta = self.detector._last_ring_meta.get(color)
        if meta is None:
            return

        cx, cy = coordinate
        outer_r = meta.get('outer_radius', 0) or 0
        axes = meta.get('axes')
        angle = meta.get('angle', 0) or 0
        area = meta.get('area') or 0

        if axes is not None and axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(display, (int(cx), int(cy)),
                        (int(axes[0]), int(axes[1])), angle,
                        0, 360, color_bgr, 2)
        elif outer_r > 0:
            cv2.circle(display, (int(cx), int(cy)),
                       int(outer_r), color_bgr, 2)

        hsv_mean = self._compute_ring_hsv(display, (cx, cy), outer_r)
        text_x = int(cx) + 10
        line_h = 18

        if hsv_mean is not None:
            cv2.putText(display,
                        f"H:{hsv_mean[0]:.0f} S:{hsv_mean[1]:.0f} V:{hsv_mean[2]:.0f}",
                        (text_x, int(cy) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if area > 0:
            cv2.putText(display, f"Area: {area:.0f}",
                        (text_x, int(cy) + 22 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    @staticmethod
    def _compute_ring_hsv(frame, center, outer_r):
        if outer_r <= 0:
            return None
        cx, cy = int(center[0]), int(center[1])
        r = int(outer_r)
        pad = 4
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(frame.shape[1], cx + r + pad)
        y2 = min(frame.shape[0], cy + r + pad)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        local_cx = cx - x1
        local_cy = cy - y1
        band_w = max(4, r // 6)
        r_outer = min(int(r + band_w), min(roi.shape[0], roi.shape[1]) // 2 - 1)
        r_inner = max(int(r - band_w), 1)
        if r_inner >= r_outer:
            return None

        annulus = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(annulus, (local_cx, local_cy), r_outer, 255, -1)
        cv2.circle(annulus, (local_cx, local_cy), r_inner, 0, -1)

        mask = annulus > 0
        if not mask.any():
            return None
        hsv_values = roi_hsv[mask]
        if len(hsv_values) == 0:
            return None
        h_mean = float(np.mean(hsv_values[:, 0]))
        s_mean = float(np.mean(hsv_values[:, 1]))
        v_mean = float(np.mean(hsv_values[:, 2]))
        return (h_mean, s_mean, v_mean)

    def _cleanup(self):
        self.window.close()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
