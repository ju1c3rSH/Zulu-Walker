# -*- coding: utf-8 -*-
"""
独立检测参数调试器

独立运行入口，用于实时整定检测参数。

使用方法:
    python -m modules.zw_opencv_module.debug.detector
    或
    python modules/zw_opencv_module/debug/detector.py
"""
import os
import sys
import time
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

from modules.zw_opencv_module.detectors.circle_target_detector import CircleTargetDetector, DetectMethod
from .window import DebugWindow
from ..param_utils import load_detect_params, get_config_path, get_default_params
from ..camera_stream import CameraStream
from ..processors.circle_target_processor import CircleTargetProcessor
from .uv_window import UVDebugWindow
from .camera_window import CameraDebugWindow


class DebugDetector:
    def __init__(
        self,
        camera_source: int = 0,
        width: int = 640,
        height: int = 480,
        config_path: str = None,
        debug_uv: bool = False,
        debug_cam: bool = False
    ):
        self.config_path = config_path or get_config_path()
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.debug_uv = debug_uv
        self.debug_cam = debug_cam

        self.stream: CameraStream = None
        self.processor = CircleTargetProcessor()
        self.detector = CircleTargetDetector()

        self.debug_window: DebugWindow = None
        self.uv_debug_window: UVDebugWindow = None
        self.cam_debug_window: CameraDebugWindow = None

        self._running = False

        self._load_params()

    def _load_params(self):
        current_method, methods_params = load_detect_params(self.config_path)
        params = methods_params.get(current_method.value, {})
        self.detector.set_detect_method(current_method)
        self.detector.set_method_params(current_method, params)

    def _on_params_change(self, method_name: str, params: dict):
        try:
            new_method = DetectMethod(method_name)
            if self.detector.get_detect_method() != new_method:
                self.detector.set_detect_method(new_method)
            self.detector.set_method_params(new_method, params)
        except ValueError:
            pass

    def _on_uv_params_change(self, uv_ranges: list, uv_min_area: int, all_params: dict = None):
        self.detector.color_ranges["UV"] = uv_ranges
        self.detector.uv_min_area = uv_min_area
        if all_params:
            self.detector.uv_adaptive_enabled = bool(int(all_params.get("uv_adaptive_enabled", 0)))
            self.detector.uv_v_percentile = int(all_params.get("uv_v_percentile", 95))
            self.detector.uv_v_floor = int(all_params.get("uv_v_floor", 90))
            self.detector.uv_s_min = int(all_params.get("uv_s_min", 80))
            h_low = int(all_params.get("uv_h_low", 130))
            h_high = int(all_params.get("uv_h_high", 160))
            self.detector.uv_h_range = (h_low, h_high)
            self.detector.uv_s_gate = int(all_params.get("uv_s_gate", 80))
            self.detector.uv_contrast_ratio_min = int(all_params.get("uv_contrast_ratio_min", 115)) / 100.0
            self.detector.uv_contrast_dilate = int(all_params.get("uv_contrast_dilate", 30))

    def _on_cam_params_change(self, params: dict):
        pass

    def start(self):
        print(f"[DebugDetector] Opening camera {self.camera_source}...")
        self.stream = CameraStream(self.camera_source, self.width, self.height)

        self.debug_window = DebugWindow(
            config_path=self.config_path,
            on_params_change=self._on_params_change
        )
        self.debug_window.setup_window()

        if self.debug_uv:
            self.uv_debug_window = UVDebugWindow(
                on_params_change=self._on_uv_params_change
            )
            self.uv_debug_window.setup_window()
            print("[DebugDetector] UV debug window enabled.")

        if self.debug_cam:
            self.cam_debug_window = CameraDebugWindow(
                cap=self.stream.cap,
                on_params_change=self._on_cam_params_change
            )
            self.cam_debug_window.setup_window()
            print("[DebugDetector] Camera params debug window enabled.")

        self._running = True
        print("[DebugDetector] Started. Press 'q' or ESC to quit.")

        self._run_loop()

    def _run_loop(self):
        while self._running:
            frame = self.stream.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            targets = self.detector.detect_circle_targets(frame)

            canny_preview = self.detector.get_edge_preview(frame)

            result_frame = frame.copy()
            for target in targets.targets:
                if target.quad_points is not None:
                    cv2.polylines(result_frame, [target.quad_points], True, (255, 0, 0), 2)
                    for pt in target.quad_points:
                        cv2.circle(result_frame, tuple(pt.ravel()), 3, (255, 100, 0), -1)
                    ordered = self.processor._order_quad_points(target.quad_points) if hasattr(self.processor, '_order_quad_points') else None
                    if ordered is not None:
                        quad_center = self.processor._get_quad_center_perspective(target.quad_points) if hasattr(self.processor, '_get_quad_center_perspective') else None
                        if quad_center is not None:
                            cv2.circle(result_frame, (int(quad_center[0]), int(quad_center[1])), 8, (0, 255, 255), 2)
                            cv2.putText(
                                result_frame, "QC",
                                (int(quad_center[0]) + 10, int(quad_center[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
                            )
                if target.contour_points is not None:
                    cv2.drawContours(result_frame, [target.contour_points], -1, (0, 255, 0), 2)
                    self.processor._draw_target(result_frame, target)
                cx, cy = target.center_coordinates
                cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    result_frame, f"({cx}, {cy})",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )

            self.debug_window.update_frame(frame, canny_preview, result_frame)
            self.debug_window.update_gui()

            if self.uv_debug_window:
                self.uv_debug_window.update_frame(frame)
                self.uv_debug_window.update_gui()

            if self.cam_debug_window:
                self.cam_debug_window.update_gui()

            cv2.imshow("Detection Result", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self._running = False
                break

        self._cleanup()

    def _cleanup(self):
        print("[DebugDetector] Cleaning up...")
        if self.debug_window:
            self.debug_window.destroy_window()
        if self.uv_debug_window:
            self.uv_debug_window.destroy_window()
        if self.cam_debug_window:
            self.cam_debug_window.destroy_window()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows()
        print("[DebugDetector] Stopped.")

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="检测参数调试器")
    parser.add_argument("--camera", "-c", type=int, default=0, help="摄像头索引 (默认: 0)")
    parser.add_argument("--width", "-W", type=int, default=640, help="画面宽度 (默认: 640)")
    parser.add_argument("--height", "-H", type=int, default=480, help="画面高度 (默认: 480)")
    parser.add_argument("--config", "-f", type=str, default=None, help="配置文件路径")

    args = parser.parse_args()

    detector = DebugDetector(
        camera_source=args.camera,
        width=args.width,
        height=args.height,
        config_path=args.config
    )

    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()


if __name__ == "__main__":
    main()
