import cv2
import numpy as np
from typing import Optional, Tuple
from ..circle_target_detector import CircleTargetDetector, DetectMethod
from ..circle import CircleTargetItem, CircleTargets
from utils.focal_distance_util import reference_size_dict

from .base import Processor, VisionResult


class CircleTargetProcessor(Processor):
    """圆形/椭圆目标检测处理器"""

    def __init__(self, name: str = "circle_target_detect"):
        super().__init__(name)
        self.detector = CircleTargetDetector(name)
        self.target_max_radius = 10  # 最大圆半径,单位为cm
        self.target_min_radius = 2  # 最小圆半径,单位为cm
        self.target_color: Optional[str] = "Red"  # 目标颜色
        self._frame_width: int = 640
        self._frame_height: int = 480
        self._fps: float = 0.0  # 从外部获取的FPS
        cv2.ocl.setUseOpenCL(True)

    def set_target_color(self, color: Optional[str]):
        """
        设置目标颜色

        Args:
            color: 颜色名称 ('Red', 'Green', 'Blue') 或 None 表示检测所有颜色
        """
        self.target_color = color

    def set_detect_method(self, method: DetectMethod):
        """
        设置检测方法

        Args:
            method: 检测方法 (DetectMethod.CONTOUR_ELLIPSE, EDGE_CONTOUR_ELLIPSE)
        """
        self.detector.set_detect_method(method)

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """
        检测圆形/椭圆目标并计算坐标偏差

        Args:
            frame: 输入图像帧
            context: 上下文，可包含 'fps'、'focal_calculator'

        Returns:
            VisionResult: 包含检测结果和坐标偏差
        """

        if context and "fps" in context:
            self._fps = context["fps"]

        # 从 context 获取 focal_calculator
        focal_calculator = context.get("focal_calculator") if context else None

        if frame is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="Empty frame provided",
            )

        try:
            self._frame_height, self._frame_width = frame.shape[:2]

            targets = self.detector.detect_circle_targets(
                frame, target_color=self.target_color
            )

            target = self._find_target(targets)

            if target is None:
                error_msg = "Target not found"
                if self.target_color:
                    error_msg = f"{self.target_color} color target not found"
                return VisionResult(
                    task_name=self.name,
                    result_data={
                        "targets": targets,
                        "target_color": self.target_color,
                        "percent_error_x": 0,
                        "percent_error_y": 0,
                        "target_distance_mm": None,
                        "target_found": False,
                    },
                    success=False,
                    error_message=error_msg,
                )

            ordered_quad = self.detector._order_quad_points(target.quad_points)
            src_points = ordered_quad.reshape(4, 2).astype(np.float32)
            edges = [
                np.linalg.norm(src_points[i] - src_points[(i+1) % 4])
                for i in range(4)
            ]

            long_edge = max(edges)
            short_edge = min(edges)

            if self.detector.is_uv_spot_detected:
                #print("[CircleTargetProcessor] UV spot detected, target may be occluded")
                uv_center = self.detector.uv_spot_center
            else:
                uv_center = None

            # 计算坐标偏差
            percent_error_x, percent_error_y = self._calculate_position_error(target, uv_center)

            # 计算目标距离
            target_distance_mm = None
            if focal_calculator and target is not None:
                quad_real_avg = reference_size_dict["quad"][2]  # 目标平均尺寸 (mm)
                avg_edge = (long_edge + short_edge) / 2  # 平均边长像素
                target_distance_mm = focal_calculator.calculate_distance(
                    real_size_mm=quad_real_avg,
                    pixel_size=avg_edge
                )

            return VisionResult(
                task_name=self.name,
                result_data={
                    "targets": targets,
                    "target": target,
                    "is_quad_detected": self.detector.is_detected_quad,
                    "is_uv_spot_detected": self.detector.is_uv_spot_detected,
                    "target_color": self.target_color,
                    "percent_error_x": percent_error_x,
                    "percent_error_y": percent_error_y,
                    "target_found": True,
                    "target_distance_mm": target_distance_mm,
                },
                success=True,
            )

        except Exception as e:
            print(f"[CircleTargetProcessor] Error processing frame: {e}")
            import traceback

            traceback.print_exc()
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

    def _find_target(self, targets: CircleTargets) -> Optional[CircleTargetItem]:
        """
        从检测结果中找到目标

        Args:
            targets: 检测结果

        Returns:
            目标对象或None
        """
        if not targets.targets:
            return None

        # 如果没有设置目标颜色，返回半径最大的目标
        if self.target_color is None:
            return max(targets.targets, key=lambda t: t.radius or 0)

        # 根据目标颜色过滤
        matching = [t for t in targets.targets if t.color == self.target_color]
        if matching:
            return max(matching, key=lambda t: t.radius or 0)
        return None

    def _calculate_position_error(self, target: CircleTargetItem, uv_center: Optional[Tuple[int, int]] = None) -> tuple:
        """
        计算目标相对于参考点的坐标偏差

        Args:
            target: 目标对象（四边形中心）
            uv_center: UV点坐标，如果提供则计算UV点与四边形中心的偏差

        Returns:
            (percent_error_x, percent_error_y): 归一化到 [-100, 100] 的坐标偏差
        """
        # 目标中心（四边形中心）
        target_x, target_y = target.center_coordinates

        if uv_center is not None:
            # UV 点模式：计算四边形中心与 UV 点的偏差
            # 误差 = 四边形中心 - UV点位置（用于将四边形中心移向UV点）
            center_x, center_y = uv_center
        else:
            # 默认模式：计算目标与屏幕中心的偏差
            center_x = self._frame_width // 2
            center_y = self._frame_height // 2

        # 像素偏差
        pixel_error_x = target_x - center_x
        pixel_error_y = target_y - center_y

        # 归一化到 [-100, 100]
        percent_error_x = int((pixel_error_x * 200) / self._frame_width)
        percent_error_y = int((pixel_error_y * 200) / self._frame_height)

        return percent_error_x, percent_error_y

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """
        在帧上绘制检测结果：拟合椭圆和圆心

        Args:
            frame: 输入图像帧
            result: 检测结果

        Returns:
            np.ndarray: 绘制后的帧
        """
        if frame is None:
            return frame

        if result.result_data is None:
            return frame

        targets = result.result_data.get("targets")
        target = result.result_data.get("target")
        target_found = result.result_data.get("target_found", False)

        # 绘制所有检测到的目标
        if targets and targets.targets:
            for t in targets.targets:
                self._draw_target(frame, t, is_target=(t == target))

        # 绘制屏幕中心十字线
        self._draw_center_cross(frame)

        # 绘制 UV 点和连线（如果检测到）
        if self.detector.is_uv_spot_detected and self.detector.uv_spot_center is not None:
            uv_x, uv_y = self.detector.uv_spot_center
            # 绘制 UV 点（紫色圆点）
            cv2.circle(frame, (uv_x, uv_y), 4, (255, 0, 255), -1)  # 紫色 (BGR: 255,0,255)
            # 绘制 UV 点坐标文字
            cv2.putText(frame, f"UV({uv_x},{uv_y})", (uv_x + 10, uv_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # 如果有目标，绘制从四边形中心到 UV 点的连线
            if target is not None:
                quad_x, quad_y = target.center_coordinates
                cv2.line(frame, (quad_x, quad_y), (uv_x, uv_y), (255, 0, 255), 2)

        # 绘制状态信息
        self._draw_status_info(frame, result, target_found)

        return frame

    def _draw_target(
        self, frame: np.ndarray, target: CircleTargetItem, is_target: bool = False
    ):
        """
        绘制单个目标：拟合椭圆和圆心

        Args:
            frame: 输入图像帧
            target: 目标对象
            is_target: 是否为主要目标
        """
        if target.color == "Red":
            color = (0, 0, 255)
        elif target.color == "Green":
            color = (0, 255, 0)
        elif target.color == "Blue":
            color = (255, 0, 0)
        else:
            color = (128, 128, 128)

        thickness = 2 if is_target else 1

        # 绘制拟合椭圆
        if target.contour_points is not None and len(target.contour_points) >= 5:
            try:
                ellipse = cv2.fitEllipse(target.contour_points)
                cv2.ellipse(frame, ellipse, color, thickness)
            except cv2.error:
                # 如果拟合失败，绘制简单圆形
                cv2.circle(
                    frame,
                    target.center_coordinates,
                    int(target.radius),
                    color,
                    thickness,
                )
        else:
            # 轮廓点不足，绘制简单圆形
            cv2.circle(
                frame, target.center_coordinates, int(target.radius), color, thickness
            )

        # 绘制圆心（红色实心圆）
        cx, cy = target.center_coordinates
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制圆心坐标文字
        coord_text = f"({cx},{cy})"
        cv2.putText(
            frame,
            coord_text,
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 如果是主要目标，绘制从屏幕中心到目标中心的连线
        if is_target:
            center_x = self._frame_width // 2
            center_y = self._frame_height // 2
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

    def _draw_center_cross(self, frame: np.ndarray):
        """绘制屏幕中心十字线"""
        center_x = self._frame_width // 2
        center_y = self._frame_height // 2

        # 水平线
        cv2.line(
            frame,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (255, 255, 255),
            1,
        )
        # 垂直线
        cv2.line(
            frame,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (255, 255, 255),
            1,
        )


    def _draw_status_info(
        self, frame: np.ndarray, result: VisionResult, target_found: bool
    ):
        """绘制状态信息"""
        data = result.result_data

        # 直接绘制半透明背景,不需要copy和addWeighted
        #cv2.rectangle(frame, (5, 5), (280, 155), (30, 30, 30), -1)

        lines = [
            (f"FPS: {self._fps:.1f}", (10, 25), (0, 255, 255)),
            (f"Method: {self.detector.detect_method.value}", (90, 25), (200, 200, 200)),
            (f"Target: {data.get('target_color', 'Any')}", (10, 50), (255, 255, 255)),
        ]

        if target_found:
            # 显示距离信息
            distance_mm = data.get('target_distance_mm')
            if distance_mm is not None:
                distance_m = distance_mm / 1000.0
                lines.append((f"Dist: {distance_m:.2f}m", (150, 50), (0, 255, 255)))

            lines.extend(
                [
                    (
                        f"Error: X={data.get('percent_error_x', 0):+d}, Y={data.get('percent_error_y', 0):+d}",
                        (10, 75),
                        (0, 255, 0),
                    ),
                    (
                        f"Count: {data.get('targets', {}).number if data.get('targets') else 0}",
                        (10, 100),
                        (255, 255, 255),
                    ),
                ]
            )
        else:
            lines.append(("Target lost", (10, 75), (0, 0, 255)))

        for text, pos, color in lines:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def _is_target_in_vision_range(self, frame: np.ndarray) -> bool:
        """通过判定摄像头视界内是否存在一个纯黑的矩形来判断目标是否在视野范围内"""
        # TODO: 实现视界范围检测
        pass
