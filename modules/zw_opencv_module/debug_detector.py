# -*- coding: utf-8 -*-
"""
独立检测参数调试器

独立运行入口，用于实时整定检测参数。

使用方法:
    python -m modules.zw_opencv_module.debug_detector
    或
    python modules/zw_opencv_module/debug_detector.py

功能:
    - 实时摄像头预览
    - 边缘检测结果预览
    - 参数滑动条调整
    - 参数自动保存到 YAML
"""
import os
import sys
import time
import argparse

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

# 使用绝对导入（支持直接运行和模块运行）
from modules.zw_opencv_module.circle_target_detector import CircleTargetDetector, DetectMethod
from modules.zw_opencv_module.debug_window import DebugWindow
from modules.zw_opencv_module.param_utils import load_detect_params, save_detect_params, get_config_path, get_default_params
from modules.zw_opencv_module.camera_stream import CameraStream
from modules.zw_opencv_module.processors.circle_target_processor import CircleTargetProcessor
from modules.zw_opencv_module.uv_debug_window import UVDebugWindow

class DebugDetector:
    """
    独立检测参数调试器

    提供实时预览和参数调整功能。
    """

    def __init__(
        self,
        camera_source: int = 0,
        width: int = 640,
        height: int = 480,
        config_path: str = None,
        debug_uv: bool = False
    ):
        """
        初始化调试器

        Args:
            camera_source: 摄像头源（索引或路径）
            width: 画面宽度
            height: 画面高度
            config_path: 配置文件路径
            debug_uv: 是否启用 UV 调试面板
        """
        self.config_path = config_path or get_config_path()
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.debug_uv = debug_uv

        # 摄像头流
        self.stream: CameraStream = None
        self.processor = CircleTargetProcessor()  # 用于获取默认参数
        # 检测器
        self.detector = CircleTargetDetector()

        # 调试窗口
        self.debug_window: DebugWindow = None

        # UV 调试窗口
        self.uv_debug_window: UVDebugWindow = None

        # 运行状态
        self._running = False

        # 加载参数
        self._load_params()

    def _load_params(self):
        """从 YAML 加载参数"""
        current_method, methods_params = load_detect_params(self.config_path)
        params = methods_params.get(current_method.value, {})
        self.detector.set_detect_method(current_method)
        self.detector.set_method_params(current_method, params)

    def _on_params_change(self, method_name: str, params: dict):
        """参数变化回调"""
        try:
            new_method = DetectMethod(method_name)
            if self.detector.get_detect_method() != new_method:
                self.detector.set_detect_method(new_method)
            self.detector.set_method_params(new_method, params)
        except ValueError:
            pass

    def _on_uv_params_change(self, uv_ranges: list, uv_min_area: int):
        """UV 参数变化回调"""
        # 更新检测器的 UV 颜色范围
        self.detector.color_ranges["UV"] = uv_ranges
        # 更新 uv_min_area 参数
        self.detector.uv_min_area = uv_min_area

    def start(self):
        """启动调试器"""
        # 初始化摄像头
        print(f"[DebugDetector] Opening camera {self.camera_source}...")
        self.stream = CameraStream(self.camera_source, self.width, self.height)

        # 初始化调试窗口
        self.debug_window = DebugWindow(
            config_path=self.config_path,
            on_params_change=self._on_params_change
        )
        self.debug_window.setup_window()

        # 初始化 UV 调试窗口（如果启用）
        if self.debug_uv:
            self.uv_debug_window = UVDebugWindow(
                on_params_change=self._on_uv_params_change
            )
            self.uv_debug_window.setup_window()
            print("[DebugDetector] UV debug window enabled.")

        self._running = True
        print("[DebugDetector] Started. Press 'q' or ESC to quit.")

        self._run_loop()

    def _run_loop(self):
        """主循环"""
        while self._running:
            # 读取帧
            frame = self.stream.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 执行检测
            targets = self.detector.detect_circle_targets(frame)

            # 获取边缘预览
            canny_preview = self.detector.get_edge_preview(frame)

            # 绘制检测结果
            result_frame = frame.copy()
            for target in targets.targets:
                # 绘制四边形（蓝色）
                if target.quad_points is not None:
                    cv2.polylines(result_frame, [target.quad_points], True, (255, 0, 0), 2)
                    # 绘制四边形顶点
                    for pt in target.quad_points:
                        cv2.circle(result_frame, tuple(pt.ravel()), 3, (255, 100, 0), -1)
                    # 计算并绘制四边形透视中心（黄色）
                    # 使用透视变换计算真实中心
                    ordered = self.processor._order_quad_points(target.quad_points) if hasattr(self.processor, '_order_quad_points') else None
                    if ordered is not None:
                        quad_center = self.processor._get_quad_center_perspective(target.quad_points) if hasattr(self.processor, '_get_quad_center_perspective') else None
                        if quad_center is not None:
                            cv2.circle(result_frame, (int(quad_center[0]), int(quad_center[1])), 8, (0, 255, 255), 2)
                            cv2.putText(
                                result_frame,
                                "QC",
                                (int(quad_center[0]) + 10, int(quad_center[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 255, 255),
                                1
                            )
                # 绘制椭圆
                if target.contour_points is not None:
                    cv2.drawContours(result_frame, [target.contour_points], -1, (0, 255, 0), 2)
                    self.processor._draw_target(result_frame, target)  # 使用处理器的绘制方法
                # 绘制中心点（红色）
                cx, cy = target.center_coordinates
                cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
                # 绘制信息
                cv2.putText(
                    result_frame,
                    f"({cx}, {cy})",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

            # 更新调试窗口
            self.debug_window.update_frame(frame, canny_preview, result_frame)
            self.debug_window.update_gui()

            # 更新 UV 调试窗口（如果启用）
            if self.uv_debug_window:
                self.uv_debug_window.update_frame(frame)
                self.uv_debug_window.update_gui()

            # 显示结果窗口
            cv2.imshow("Detection Result", result_frame)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                self._running = False
                break

        self._cleanup()

    def _cleanup(self):
        """清理资源"""
        print("[DebugDetector] Cleaning up...")
        if self.debug_window:
            self.debug_window.destroy_window()
        if self.uv_debug_window:
            self.uv_debug_window.destroy_window()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows()
        print("[DebugDetector] Stopped.")

    def stop(self):
        """停止调试器"""
        self._running = False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="检测参数调试器")
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="摄像头索引 (默认: 0)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="画面宽度 (默认: 640)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="画面高度 (默认: 480)"
    )
    parser.add_argument(
        "--config", "-f",
        type=str,
        default=None,
        help="配置文件路径"
    )

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
